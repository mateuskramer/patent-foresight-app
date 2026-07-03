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
import time

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

# --- Configuração do Cache Local ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
CACHE_PATENTS_PATH = os.path.join(CACHE_DIR, "patents.json")
CACHE_TERMS_PATH = os.path.join(CACHE_DIR, "terms.json")
CACHE_TTL = 86400  # 24 horas (dados estáveis de patentes)
os.makedirs(CACHE_DIR, exist_ok=True)

COOLDOWN_PATH = os.path.join(CACHE_DIR, "cooldown.txt")
COOLDOWN_TIME = 60  # 1 minuto de cooldown em caso de falha da API

def _is_api_in_cooldown():
    try:
        if os.path.exists(COOLDOWN_PATH):
            if time.time() - os.path.getmtime(COOLDOWN_PATH) < COOLDOWN_TIME:
                return True
            else:
                try:
                    os.remove(COOLDOWN_PATH)
                except OSError:
                    pass
    except Exception:
        pass
    return False

def _set_api_cooldown():
    try:
        with open(COOLDOWN_PATH, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
    except Exception:
        pass

def _clear_api_cooldown():
    try:
        if os.path.exists(COOLDOWN_PATH):
            os.remove(COOLDOWN_PATH)
    except OSError:
        pass

def _read_cache(path):
    """Retorna dados do cache se existir e for recente, ou None."""
    try:
        if os.path.exists(path) and (time.time() - os.path.getmtime(path) < CACHE_TTL):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _read_cache_fallback(path):
    """Retorna dados do cache mesmo que expirado (fallback de falha)."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _write_cache(path, data):
    try:
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_path, path)  # Substituição atômica e segura contra concorrência
    except Exception as e:
        logger.warning("Erro ao salvar cache em %s: %s", path, e)

def requests_get_with_retry(url, timeout=30, max_retries=3, delay=3):
    import random
    # Adiciona pequeno jitter inicial para evitar que múltiplos workers batam no exato mesmo milissegundo
    time.sleep(random.uniform(0.1, 0.5))
    
    for attempt in range(max_retries):
        try:
            logger.info("Requisitando API (tentativa %d/%d): %s", attempt + 1, max_retries, url)
            r = requests.get(url, timeout=timeout)
            
            # Se for 429, trata com recuo (backoff) explícito
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait_time = int(retry_after) if (retry_after and retry_after.isdigit()) else (delay * (attempt + 1) * 2)
                logger.warning("API retornou 429 (Too Many Requests). Aguardando %ds antes de tentar novamente...", wait_time)
                time.sleep(wait_time)
                continue
                
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as http_err:
            if http_err.response is not None and http_err.response.status_code == 429:
                retry_after = http_err.response.headers.get("Retry-After")
                wait_time = int(retry_after) if (retry_after and retry_after.isdigit()) else (delay * (attempt + 1) * 2)
                logger.warning("API retornou 429. Aguardando %ds...", wait_time)
                time.sleep(wait_time)
                continue
            
            if attempt < max_retries - 1:
                logger.warning("API HTTP error (attempt %d/%d): %s. Retrying in %ds...", attempt + 1, max_retries, http_err, delay)
                time.sleep(delay)
            else:
                raise http_err
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning("API request failed (attempt %d/%d): %s. Retrying in %ds...", attempt + 1, max_retries, e, delay)
                time.sleep(delay)
            else:
                raise e

def load_patents():
    cached = _read_cache(CACHE_PATENTS_PATH)
    if cached is not None:
        logger.info("Carregando patentes do cache local.")
        df = pd.DataFrame(cached)
    else:
        df = pd.DataFrame()
        # Implementação de Trava de Arquivo (.lock) contra concorrência de workers no cold start
        lock_path = CACHE_PATENTS_PATH + ".lock"
        for _ in range(30):
            if os.path.exists(lock_path) and (time.time() - os.path.getmtime(lock_path) < 60):
                logger.info("Outro worker está baixando as patentes. Aguardando cache local...")
                time.sleep(2)
                cached = _read_cache(CACHE_PATENTS_PATH)
                if cached is not None:
                    logger.info("Carregando patentes do cache recém-criado por outro worker.")
                    df = pd.DataFrame(cached)
                    break
            else:
                # Se o lock foi removido ou não existia, tenta ler o cache de novo antes de ir à API
                cached = _read_cache(CACHE_PATENTS_PATH)
                if cached is not None:
                    logger.info("Carregando patentes do cache local.")
                    df = pd.DataFrame(cached)
                break

        # Se após aguardar o lock ainda estiver vazio, este worker busca na API
        if df.empty:
            # Cria o arquivo de lock
            try:
                with open(lock_path, "w", encoding="utf-8") as f:
                    f.write(str(os.getpid()))
            except Exception:
                pass
                
            try:
                if _is_api_in_cooldown():
                    logger.warning("API em cooldown devido a falha recente. Ignorando chamada de patentes.")
                    fallback = _read_cache_fallback(CACHE_PATENTS_PATH)
                    df = pd.DataFrame(fallback) if fallback else pd.DataFrame()
                else:
                    # Envia um ping leve de wake-up para o /health antes de puxar dados pesados (timeout longo 60s)
                    try:
                        logger.info("Enviando ping de wake-up para o endpoint /health da API...")
                        r_health = requests.get(f"{API_BASE_URL}/health", timeout=60)
                        if r_health.status_code == 429:
                            logger.warning("API /health retornou 429 (já acordada). Prosseguindo...")
                        else:
                            r_health.raise_for_status()
                            logger.info("API acordada com sucesso (resposta /health: %s).", r_health.text.strip())
                    except Exception as wake_err:
                        logger.warning("Falha ou timeout no ping de wake-up da API: %s. Continuando...", wake_err)

                    try:
                        r = requests_get_with_retry(f"{API_BASE_URL}/patents", timeout=30)
                        data = r.json()
                        _write_cache(CACHE_PATENTS_PATH, data)
                        _clear_api_cooldown()
                        df = pd.DataFrame(data)
                    except Exception as e:
                        logger.warning("Falha na API de patentes, ativando cooldown e usando fallback: %s", e)
                        _set_api_cooldown()
                        fallback = _read_cache_fallback(CACHE_PATENTS_PATH)
                        df = pd.DataFrame(fallback) if fallback else pd.DataFrame()
            finally:
                # Remove o arquivo de lock
                try:
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
                except Exception:
                    pass

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

BG      = "#000000"
CARD    = "#0a0a0a"
BLUE    = "#2563eb"
TEXT    = "#ffffff"
MUTED   = "#6b7280"
BORDER  = "#1f2937"
PALETTE = ["#2563eb", "#e74c3c", "#22c55e", "#f39c12", "#9b59b6"]

def load_terms():
    cached = _read_cache(CACHE_TERMS_PATH)
    if cached is not None:
        logger.info("Carregando termos do cache local.")
        return pd.DataFrame(cached)
        
    # Implementação de Trava de Arquivo (.lock) contra concorrência de workers no cold start
    lock_path = CACHE_TERMS_PATH + ".lock"
    for _ in range(30):
        if os.path.exists(lock_path) and (time.time() - os.path.getmtime(lock_path) < 60):
            logger.info("Outro worker está baixando os termos. Aguardando cache local...")
            time.sleep(2)
            cached = _read_cache(CACHE_TERMS_PATH)
            if cached is not None:
                logger.info("Carregando termos do cache recém-criado por outro worker.")
                return pd.DataFrame(cached)
        else:
            # Se o lock foi removido ou não existia, tenta ler o cache de novo antes de ir à API
            cached = _read_cache(CACHE_TERMS_PATH)
            if cached is not None:
                logger.info("Carregando termos do cache local.")
                return pd.DataFrame(cached)
            break

    # Se após aguardar o lock ainda estiver vazio, este worker busca na API
    # Cria o arquivo de lock
    try:
        with open(lock_path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass
        
    try:
        if _is_api_in_cooldown():
            logger.warning("API em cooldown devido a falha recente. Ignorando chamada de termos.")
            fallback = _read_cache_fallback(CACHE_TERMS_PATH)
            df = pd.DataFrame(fallback) if fallback else pd.DataFrame()
        else:
            try:
                r = requests_get_with_retry(f"{API_BASE_URL}/terms/associations", timeout=30)
                data = r.json()
                _write_cache(CACHE_TERMS_PATH, data)
                _clear_api_cooldown()
                df = pd.DataFrame(data)
            except Exception as e:
                logger.warning("Falha na API de termos, ativando cooldown e usando fallback: %s", e)
                _set_api_cooldown()
                fallback = _read_cache_fallback(CACHE_TERMS_PATH)
                df = pd.DataFrame(fallback) if fallback else pd.DataFrame()
    finally:
        # Remove o arquivo de lock
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception:
            pass
            
    return df




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

def similar_patents(idx, df, EMB_arg=None, top_n=10):
    current_emb = EMB_arg if (EMB_arg is not None and len(EMB_arg) > 0) else EMB
    if len(current_emb) == 0 or idx >= len(current_emb):
        return pd.DataFrame()
    sims = cosine_similarity(current_emb[idx].reshape(1, -1), current_emb)[0]
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

def get_sparse_opportunities(target_term, C=None, t_map_arg=None, idx_map_arg=None, top_n=20):
    current_C = C if C is not None else C_matrix
    current_t_map = t_map_arg if t_map_arg else t_map
    current_idx_map = idx_map_arg if idx_map_arg else idx_map

    if current_C is None or target_term not in current_t_map:
        return pd.DataFrame()
    idx      = current_t_map[target_term]
    direct   = current_C[idx].toarray().flatten()
    indirect = (current_C @ current_C[idx].T).toarray().flatten()
    mask     = (indirect > 0) & (direct == 0)
    mask[idx]= False
    p_idx    = np.where(mask)[0]
    if not len(p_idx):
        return pd.DataFrame()
    max_v = current_C.max() or 1
    return (
        pd.DataFrame([{
            "term":           current_idx_map[i],
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

    new_patents = load_patents()
    
    # Intervalo de 1.5s entre chamadas sequenciais pesadas para respeitar o rate-limiter da API
    time.sleep(1.5)
    
    new_terms   = load_terms()

    # Se estiver vazio, roda o diagnóstico e tenta carregar do cache recém-criado
    if new_patents.empty or new_terms.empty:
        logger.warning("DADOS VAZIOS OU PARCIAIS DETECTADOS NO STARTUP!")
        print("=== DADOS VAZIOS OU PARCIAIS DETECTADOS NO STARTUP! ===")
        print(f"API_BASE_URL configurada: {API_BASE_URL}")
        
        # Evita bombardear a API com diagnósticos se ela já estiver em cooldown recente (erro 429/timeout)
        if _is_api_in_cooldown():
            logger.warning("API está em cooldown recente. Pulando diagnósticos de rede para evitar sobrecarga.")
            print("API está em cooldown recente. Pulando diagnósticos de rede para evitar sobrecarga.")
        else:
            logger.warning("Iniciando diagnósticos da API...")
            logger.warning("API_BASE_URL configurada: %s", API_BASE_URL)
            
            # Testar /health
            try:
                r = requests.get(f"{API_BASE_URL}/health", timeout=10)
                logger.warning("API /health status: %d | body: %s", r.status_code, r.text.strip())
                print(f"API /health status: {r.status_code} | body: {r.text.strip()}")
            except Exception as e:
                logger.warning("API /health falhou: %s", e, exc_info=True)
                print(f"API /health falhou: {e}")
                
            # Testar /patents
            saved_patents = False
            try:
                r = requests.get(f"{API_BASE_URL}/patents", timeout=30)
                logger.warning("API /patents status: %d | tamanho da resposta: %d bytes", r.status_code, len(r.content))
                print(f"API /patents status: {r.status_code} | tamanho: {len(r.content)} bytes")
                if r.status_code == 200:
                    data = r.json()
                    _write_cache(CACHE_PATENTS_PATH, data)
                    saved_patents = True
                    logger.warning("Cache de patentes salvo via diagnóstico.")
            except Exception as e:
                logger.warning("API /patents falhou: %s", e, exc_info=True)
                print(f"API /patents falhou: {e}")
                
            # Testar /terms/associations
            saved_terms = False
            try:
                r = requests.get(f"{API_BASE_URL}/terms/associations", timeout=20)
                logger.warning("API /terms/associations status: %d | tamanho: %d bytes", r.status_code, len(r.content))
                print(f"API /terms/associations status: {r.status_code} | tamanho: {len(r.content)} bytes")
                if r.status_code == 200:
                    data = r.json()
                    _write_cache(CACHE_TERMS_PATH, data)
                    saved_terms = True
                    logger.warning("Cache de termos salvo via diagnóstico.")
            except Exception as e:
                logger.warning("API /terms/associations falhou: %s", e, exc_info=True)
                print(f"API /terms/associations falhou: {e}")

            # Se ambos os caches foram salvos com sucesso pelo diagnóstico, recarrega e limpa o cooldown
            if saved_patents and saved_terms:
                _clear_api_cooldown()
                logger.warning("Sincronização via diagnóstico concluída com sucesso. Carregando dados do cache...")
                print("Sincronização via diagnóstico concluída com sucesso. Carregando dados do cache...")
                new_patents = load_patents()
                new_terms = load_terms()

    # Atualiza df_patents in-place para manter referências em outros arquivos
    df_patents.drop(df_patents.index, inplace=True)
    for col in list(df_patents.columns):
        del df_patents[col]
    for col in new_patents.columns:
        df_patents[col] = new_patents[col]

    # Atualiza terms_df in-place para manter referências em outros arquivos
    terms_df.drop(terms_df.index, inplace=True)
    for col in list(terms_df.columns):
        del terms_df[col]
    for col in new_terms.columns:
        terms_df[col] = new_terms[col]

    # FIX: vstack protegido contra embeddings com shapes diferentes
    if not df_patents.empty and "embedding" in df_patents.columns:
        emb_list = df_patents["embedding"].tolist()
        try:
            EMB = np.vstack(emb_list)
        except ValueError:
            logger.warning("Embeddings com shapes inconsistentes, filtrando", exc_info=True)
            first_shape = emb_list[0].shape
            mask        = [e.shape == first_shape for e in emb_list]
            
            # Filtra df_patents in-place
            filtered_patents = df_patents[mask].reset_index(drop=True)
            df_patents.drop(df_patents.index, inplace=True)
            for col in list(df_patents.columns):
                del df_patents[col]
            for col in filtered_patents.columns:
                df_patents[col] = filtered_patents[col]
                
            EMB         = np.vstack([e for e, ok in zip(emb_list, mask) if ok])
    else:
        EMB = np.array([])

    new_C, new_t_map, new_idx_map = (
        prepare_sparse_engine(terms_df) if not terms_df.empty else (None, {}, {})
    )
    C_matrix = new_C
    
    # Atualiza t_map in-place
    t_map.clear()
    t_map.update(new_t_map)
    
    # Atualiza idx_map in-place
    idx_map.clear()
    idx_map.update(new_idx_map)

    new_term_list = (
        terms_df["term"].value_counts().index.tolist()[:500]
        if not terms_df.empty else []
    )
    # Atualiza term_list in-place para manter referências em outros arquivos (ex: app.py)
    term_list.clear()
    term_list.extend(new_term_list)

    new_patent_opts = (
        [{"label": f"{r['id']} - {str(r['title'])[:40]}", "value": i}
         for i, r in df_patents.iterrows()]
        if not df_patents.empty else []
    )
    # Atualiza patent_opts in-place para manter referências em outros arquivos (ex: app.py)
    patent_opts.clear()
    patent_opts.extend(new_patent_opts)

    # dados novos invalidam qualquer cálculo cacheado da geração anterior
    clear_calc_cache()
    clear_unique_terms_cache()
    
    # invalidar o snapshot de chat_sessions se importado
    try:
        from chat import invalidate_db_snapshot
        invalidate_db_snapshot()
    except Exception:
        pass

    if not df_patents.empty and not terms_df.empty:
        logger.info("Pronto! %d patentes, %d termos.", len(df_patents), len(term_list))
        print(f"=== SUCESSO NO STARTUP: {len(df_patents)} patentes, {len(term_list)} termos carregados. ===")
    else:
        logger.warning("DADOS VAZIOS OU PARCIAIS DETECTADOS NO STARTUP!")
        print("=== DADOS VAZIOS OU PARCIAIS DETECTADOS NO STARTUP! ===")
        print(f"API_BASE_URL configurada: {API_BASE_URL}")

refresh_data()