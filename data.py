import os
import logging
import functools
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import json
import warnings
import requests

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
load_dotenv()

logger = logging.getLogger(__name__)

DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    database=os.getenv("DB_NAME", "patents"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASS", ""),
    port=os.getenv("DB_PORT", "5432"),
)
DATABASE_URI = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

engine = create_engine(
    DATABASE_URI,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=10,
    max_overflow=20,
)

def run_query(query, params=None):
    try:
        with engine.connect() as conn:
            res = conn.execute(text(query), params) if params else conn.execute(text(query))
            return pd.DataFrame(res.fetchall(), columns=list(res.keys()))
    except Exception:
        logger.error("Erro na query: %s", query[:120], exc_info=True)
        return pd.DataFrame()

def run_write(query, params=None):
    try:
        with engine.begin() as conn:
            conn.execute(text(query), params or {})
        return True
    except Exception:
        logger.error("Erro na escrita: %s", query[:120], exc_info=True)
        return False

API_BASE_URL = os.getenv("API_BASE_URL", "https://apipatent.onrender.com").rstrip("/")
API_KEY = os.getenv("API_KEY", "")

def requests_get_with_retry(url, timeout=90, max_retries=4, delay=10):
    import time
    for attempt in range(max_retries):
        try:
            logger.info("Requisitando API (tentativa %d/%d): %s", attempt + 1, max_retries, url)
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning("API request failed (attempt %d/%d): %s. Retrying in %ds...", attempt + 1, max_retries, e, delay)
                time.sleep(delay)
            else:
                raise e

def load_patents():
    try:
        r = requests_get_with_retry(f"{API_BASE_URL}/patents")
        data = r.json()
        df = pd.DataFrame(data)
        if not df.empty and "embedding" in df.columns:
            df["embedding"] = df["embedding"].apply(
                lambda x: np.array(x, dtype=np.float32)
                if isinstance(x, list) else (
                    np.array(json.loads(x), dtype=np.float32)
                    if isinstance(x, str) and x != "" else None
                )
            )
            df = df.dropna(subset=["embedding"])
        return df.reset_index(drop=True)
    except Exception:
        logger.error("Erro ao carregar patentes da API", exc_info=True)
        return pd.DataFrame()


BG      = "#000000"
CARD    = "#0a0a0a"
BLUE    = "#2563eb"
TEXT    = "#ffffff"
MUTED   = "#6b7280"
BORDER  = "#1f2937"
PALETTE = ["#2563eb", "#e74c3c", "#22c55e", "#f39c12", "#9b59b6"]

def load_terms():
    try:
        r = requests_get_with_retry(f"{API_BASE_URL}/terms/associations")
        data = r.json()
        return pd.DataFrame(data)
    except Exception:
        logger.error("Erro ao carregar termos da API", exc_info=True)
        return pd.DataFrame()



def monthly_term_count(term, df):
    return (df[df["term"] == term]
            .groupby("year_month").size()
            .reset_index(name="count")
            .sort_values("year_month"))

# ─── cache manual para cálculos por termo ──────────────────────────────────────
# lru_cache não serve aqui porque os calc_* recebem um DataFrame (não hasheável).
# Em vez disso, cacheamos por (nome_da_função, term, id(df), len(df)) — o id(df)
# muda quando o DataFrame é recarregado (ver refresh_data), invalidando o cache
# naturalmente; len(df) é uma salvaguarda barata contra reuso de id após GC.
_CALC_CACHE = {}
_CALC_CACHE_MAXSIZE = 8192

def _cached_term_calc(func):
    @functools.wraps(func)
    def wrapper(term, df):
        key = (func.__name__, term, id(df), len(df))
        if key in _CALC_CACHE:
            return _CALC_CACHE[key]
        result = func(term, df)
        if len(_CALC_CACHE) >= _CALC_CACHE_MAXSIZE:
            _CALC_CACHE.clear()  # eviction simples — recomeça do zero
        _CALC_CACHE[key] = result
        return result
    return wrapper

def clear_calc_cache():
    """Limpa o cache de cálculos por termo. Chamado automaticamente por refresh_data()."""
    _CALC_CACHE.clear()

# ─── cache do "template" de termos usado por semantic_vector ─────────────────
# Evita recriar df["term"].unique() a cada chamada (semantic_vector é chamada
# em loop por calc_shift e ranking_table — até centenas de vezes por execução).
_UNIQUE_TERMS_CACHE = {}
_UNIQUE_TERMS_CACHE_MAXSIZE = 64

def _unique_terms_cached(df):
    key = (id(df), len(df))
    if key not in _UNIQUE_TERMS_CACHE:
        if len(_UNIQUE_TERMS_CACHE) >= _UNIQUE_TERMS_CACHE_MAXSIZE:
            _UNIQUE_TERMS_CACHE.clear()
        _UNIQUE_TERMS_CACHE[key] = df["term"].unique()
    return _UNIQUE_TERMS_CACHE[key]

def clear_unique_terms_cache():
    _UNIQUE_TERMS_CACHE.clear()

def semantic_vector(term, month, df):
    patents = df[(df["year_month"] == month) & (df["term"] == term)]["patent_id"].unique()
    co = df[df["patent_id"].isin(patents)]
    vec = co["term"].value_counts()
    full = pd.Series(0, index=_unique_terms_cached(df))
    full.update(vec)
    return full.values

@_cached_term_calc
def calc_growth(term, df):
    """
    Compara os últimos dois meses COMPLETOS disponíveis.
    Usa os 2 últimos meses do histórico (não necessariamente o mês atual).
    """
    m = monthly_term_count(term, df)
    if len(m) < 2:
        return 0.0
    # pega os dois últimos registros (já ordenados por year_month)
    p = m.iloc[-2]["count"]
    c = m.iloc[-1]["count"]
    return float(((c - p) / p) * 100) if p else 0.0

@_cached_term_calc
def calc_density(term, df):
    return len(df[df["term"] == term]["patent_id"].unique())

@_cached_term_calc
def calc_fusion(term, df):
    return df[
        (df["patent_id"].isin(df[df["term"] == term]["patent_id"])) &
        (df["term"] != term)
    ]["term"].nunique()

@_cached_term_calc
def calc_shift(term, df):
    months = sorted(df["year_month"].dropna().unique().tolist())
    if len(months) < 2:
        return 0.0
    v1 = semantic_vector(term, months[0], df)
    v2 = semantic_vector(term, months[-1], df)
    return float(cosine(v1, v2) * 100) if (v1.sum() > 0 and v2.sum() > 0) else 0.0

@_cached_term_calc
def calc_future_score(term, df):
    return round(
        0.35 * min(max(calc_growth(term, df), 0), 100) +
        0.25 * min(calc_fusion(term, df) * 5, 100) +
        0.20 * min(calc_shift(term, df), 100) +
        0.20 * min(calc_density(term, df), 100), 2)

def ranking_table(df):
    """
    Versão vetorizada — evita chamar 5 funções por termo em loop.
    Calcula todos os indicadores de uma vez aproveitando operações em DataFrame.
    """
    if df.empty:
        return pd.DataFrame()

    terms = df["term"].value_counts().index.tolist()[:300]
    df_f  = df[df["term"].isin(terms)].copy()

    # ── density: patentes únicas por termo ──
    density = df_f.groupby("term")["patent_id"].nunique().rename("density")

    # ── fusion: termos co-ocorrentes únicos ──
    co = df_f.merge(
        df_f[["patent_id", "term"]].rename(columns={"term": "other"}),
        on="patent_id"
    )
    fusion = (
        co[co["term"] != co["other"]]
        .groupby("term")["other"].nunique()
        .rename("fusion")
    )

    # ── growth: variação entre os dois últimos meses ──
    # Usa pivot para pegar último e penúltimo mês por termo,
    # evitando problemas com apply+MultiIndex no pandas 2.x
    monthly = (
        df_f.groupby(["term", "year_month"]).size()
        .reset_index(name="count")
        .sort_values(["term", "year_month"])
    )

    def _calc_growth(grp):
        if len(grp) < 2:
            return 0.0
        p = grp["count"].iloc[-2]
        c = grp["count"].iloc[-1]
        return float(((c - p) / p) * 100) if p else 0.0

    growth = (
        monthly.groupby("term")
        .apply(_calc_growth, include_groups=False)
        .rename("growth_%")
        .round(2)
    )

    # ── shift: distância cosseno entre primeiro e último mês ──
    months_sorted = sorted(df_f["year_month"].dropna().unique().tolist())
    shift_rows = {}
    if len(months_sorted) >= 2:
        m_first, m_last = months_sorted[0], months_sorted[-1]
        for t in terms:
            v1 = semantic_vector(t, m_first, df_f)
            v2 = semantic_vector(t, m_last,  df_f)
            shift_rows[t] = float(cosine(v1, v2) * 100) if (v1.sum() > 0 and v2.sum() > 0) else 0.0
    shift = pd.Series(shift_rows, name="shift_%").round(2)

    # ── monta tabela final ──
    result = (
        pd.DataFrame({"term": terms})
        .set_index("term")
        .join(density)
        .join(fusion)
        .join(growth)
        .join(shift)
        .fillna(0)
        .reset_index()
    )
    result["future_score"] = (
        0.35 * result["growth_%"].clip(0, 100) +
        0.25 * (result["fusion"] * 5).clip(0, 100) +
        0.20 * result["shift_%"].clip(0, 100) +
        0.20 * result["density"].clip(0, 100)
    ).round(2)

    return result.sort_values("future_score", ascending=False).reset_index(drop=True)

def term_correlations(term, df):
    """
    Versão vetorizada (era O(n_termos) em Python puro com sets).
    Mesma lógica e mesmas fórmulas (lift, jaccard, pmi), mas calculadas
    em bloco via pandas em vez de iterar termo a termo.
    """
    if df.empty or term not in df["term"].values:
        return pd.DataFrame()

    total = df["patent_id"].nunique()
    pa_patents = df.loc[df["term"] == term, "patent_id"].unique()
    pa_size = len(pa_patents)
    if pa_size == 0 or total == 0:
        return pd.DataFrame()

    # |pb| pré-calculado para todos os termos de uma vez
    term_sizes = df.groupby("term")["patent_id"].nunique()

    # |pa ∩ pb| para cada termo b: restringe às patentes de `term` e agrupa
    sub = df[df["patent_id"].isin(pa_patents) & (df["term"] != term)]
    inter = sub.groupby("term")["patent_id"].nunique()
    if inter.empty:
        return pd.DataFrame()

    pb_sizes = term_sizes.reindex(inter.index)

    pa_frac = pa_size / total
    pb_frac = pb_sizes / total
    pab     = inter / total
    union_size = pa_size + pb_sizes - inter

    result = pd.DataFrame({
        "term":    inter.index,
        "cooc":    inter.values,
        "lift":    (pab / (pa_frac * pb_frac)).round(4).values,
        "jaccard": (inter / union_size).round(4).values,
        "pmi":     np.log2(pab / (pa_frac * pb_frac)).round(4).values,
    })

    return result.sort_values("lift", ascending=False).reset_index(drop=True)

def build_temporal_matrix(df):
    return df.groupby(["year_month", "term"]).size().unstack(fill_value=0).sort_index()

def pearson_with_term(pivot, selected_term):
    if selected_term not in pivot.columns:
        return pd.DataFrame()
    x = pivot[selected_term].values
    if x.std() == 0 or len(x) < 3:
        return pd.DataFrame()
    rows = []
    for term in pivot.columns:
        if term == selected_term:
            continue
        y = pivot[term].values
        if y.std() == 0 or len(y) < 3:
            continue
        try:
            r, p = pearsonr(x, y)
            if not np.isnan(r):
                rows.append({"parceiro": term, "pearson_r": r, "p_value": p})
        except Exception:
            continue
    return (pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
            if rows else pd.DataFrame())

def build_graph(root_term, df, depth=3, top_n=5):
    if df.empty or root_term not in df["term"].values:
        return nx.Graph()
    G = nx.Graph()
    G.add_node(root_term, layer=0)
    frontier, visited = [(root_term, 0)], set()
    while frontier:
        curr, level = frontier.pop(0)
        if curr in visited or level >= depth:
            continue
        visited.add(curr)
        pats     = df[df["term"] == curr]["patent_id"].unique()
        co_counts = df[(df["patent_id"].isin(pats)) & (df["term"] != curr)]["term"].value_counts()
        for t, w in co_counts.head(top_n).items():
            if not G.has_node(t):
                G.add_node(t, layer=level + 1)
            G.add_edge(curr, t, weight=int(w))
            frontier.append((t, level + 1))
    return G

def similar_patents(idx, df, EMB, top_n=10):
    if len(EMB) == 0 or idx >= len(EMB):
        return pd.DataFrame()
    sims = cosine_similarity(EMB[idx].reshape(1, -1), EMB)[0]
    out  = df.copy()
    out["similarity"] = sims
    return out[out.index != idx].sort_values("similarity", ascending=False).head(top_n)

def prepare_sparse_engine(df_terms):
    df_terms             = df_terms.copy()
    df_terms["term"]     = df_terms["term"].astype("category")
    df_terms["patent_id"]= df_terms["patent_id"].astype("category")
    idx_to_term          = dict(enumerate(df_terms["term"].cat.categories))
    A = sp.csr_matrix((
        np.ones(len(df_terms)),
        (df_terms["patent_id"].cat.codes, df_terms["term"].cat.codes),
    ))
    C = A.T @ A
    C.setdiag(0)
    return C, {t: i for i, t in idx_to_term.items()}, idx_to_term

def get_sparse_opportunities(target_term, C, t_map, idx_map, top_n=20):
    if C is None or target_term not in t_map:
        return pd.DataFrame()
    idx      = t_map[target_term]
    direct   = C[idx].toarray().flatten()
    indirect = (C @ C[idx].T).toarray().flatten()
    mask     = (indirect > 0) & (direct == 0)
    mask[idx]= False
    p_idx    = np.where(mask)[0]
    if not len(p_idx):
        return pd.DataFrame()
    max_v = C.max() or 1
    return (
        pd.DataFrame([{
            "term":           idx_map[i],
            "bridge_strength": int(indirect[i]),
            "score":          round(indirect[i] / max_v, 4),
        } for i in p_idx])
        .sort_values("bridge_strength", ascending=False)
        .head(top_n)
    )

# ─── carga/recarga de dados ────────────────────────────────────────────────────
# Estas variáveis são globais do módulo, lidas por outros arquivos via
# `from data import df_patents, terms_df, ...`. refresh_data() permite
# recarregar tudo (ex.: após o processador.py adicionar patentes novas)
# sem reiniciar o processo — basta chamar data.refresh_data().
df_patents  = pd.DataFrame()
terms_df    = pd.DataFrame()
EMB         = np.array([])
C_matrix    = None
t_map       = {}
idx_map     = {}
term_list   = []
patent_opts = []

def refresh_data():
    """Recarrega patentes, termos, embeddings e estruturas derivadas do banco."""
    global df_patents, terms_df, EMB, C_matrix, t_map, idx_map, term_list, patent_opts

    logger.info("Carregando dados...")

    df_patents = load_patents()
    terms_df   = load_terms()

    # FIX: vstack protegido contra embeddings com shapes diferentes
    if not df_patents.empty and "embedding" in df_patents.columns:
        emb_list = df_patents["embedding"].tolist()
        try:
            EMB = np.vstack(emb_list)
        except ValueError:
            logger.warning("Embeddings com shapes inconsistentes, filtrando", exc_info=True)
            first_shape = emb_list[0].shape
            mask        = [e.shape == first_shape for e in emb_list]
            df_patents  = df_patents[mask].reset_index(drop=True)
            EMB         = np.vstack([e for e, ok in zip(emb_list, mask) if ok])
    else:
        EMB = np.array([])

    C_matrix, t_map, idx_map = (
        prepare_sparse_engine(terms_df) if not terms_df.empty else (None, {}, {})
    )

    term_list = (
        terms_df["term"].value_counts().index.tolist()[:500]
        if not terms_df.empty else []
    )

    patent_opts = (
        [{"label": f"{r['id']} - {str(r['title'])[:40]}", "value": i}
         for i, r in df_patents.iterrows()]
        if not df_patents.empty else []
    )

    # dados novos invalidam qualquer cálculo cacheado da geração anterior
    clear_calc_cache()
    clear_unique_terms_cache()

    if not df_patents.empty and not terms_df.empty:
        logger.info("Pronto! %d patentes, %d termos.", len(df_patents), len(term_list))
    else:
        logger.warning("DADOS VAZIOS OU PARCIAIS DETECTADOS NO STARTUP!")
        logger.warning("Iniciando diagnósticos da API...")
        logger.warning("API_BASE_URL configurada: %s", API_BASE_URL)
        
        # Testar /health
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=10)
            logger.warning("API /health status: %d | body: %s", r.status_code, r.text.strip())
        except Exception as e:
            logger.warning("API /health falhou: %s", e, exc_info=True)
            
        # Testar /patents
        try:
            r = requests.get(f"{API_BASE_URL}/patents", timeout=15)
            logger.warning("API /patents status: %d | tamanho da resposta: %d bytes", r.status_code, len(r.content))
        except Exception as e:
            logger.warning("API /patents falhou: %s", e, exc_info=True)
            
        # Testar /terms/associations
        try:
            r = requests.get(f"{API_BASE_URL}/terms/associations", timeout=15)
            logger.warning("API /terms/associations status: %d | tamanho: %d bytes", r.status_code, len(r.content))
        except Exception as e:
            logger.warning("API /terms/associations falhou: %s", e, exc_info=True)


refresh_data()