import uuid
import time
import logging
import pandas as pd

from llm import create_chat

from data import (
    run_query, run_write, engine,
    terms_df, monthly_term_count,
    calc_growth, calc_density, calc_fusion, calc_shift, calc_future_score,
    ranking_table, term_correlations, get_sparse_opportunities,
    C_matrix, t_map, idx_map,
)

logger = logging.getLogger(__name__)

def _ensure_table():
    """Cria a tabela chat_sessions se não existir."""
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id         SERIAL PRIMARY KEY,
                    session_id TEXT        NOT NULL,
                    role       TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
                    content    TEXT        NOT NULL,
                    created_at TIMESTAMP   DEFAULT NOW()
                );
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id
                ON chat_sessions(session_id);
            """))
        logger.info("Tabela chat_sessions OK")
    except Exception:
        logger.error("Erro ao criar tabela chat_sessions", exc_info=True)

_ensure_table()


def new_session_id() -> str:
    return str(uuid.uuid4())


def load_history(session_id: str) -> list[dict]:
    """Carrega histórico de mensagens do banco."""
    df = run_query(
        "SELECT role, content FROM chat_sessions "
        "WHERE session_id = :sid ORDER BY created_at ASC",
        {"sid": session_id},
    )
    if df.empty:
        return []
    return df.to_dict("records")


def save_message(session_id: str, role: str, content: str):
    """Salva uma mensagem no banco."""
    run_write(
        "INSERT INTO chat_sessions (session_id, role, content) "
        "VALUES (:sid, :role, :content)",
        {"sid": session_id, "role": role, "content": content},
    )


def list_sessions() -> pd.DataFrame:
    """Lista todas as sessões com preview da primeira mensagem."""
    return run_query("""
        SELECT
            session_id,
            MIN(created_at)  AS started_at,
            COUNT(*)         AS messages,
            MIN(CASE WHEN role = 'user' THEN content END) AS preview
        FROM chat_sessions
        GROUP BY session_id
        ORDER BY MIN(created_at) DESC
        LIMIT 50
    """)


def delete_session(session_id: str):
    run_write(
        "DELETE FROM chat_sessions WHERE session_id = :sid",
        {"sid": session_id},
    )


# ─── cache com TTL do snapshot do banco ───────────────────────────────────────
# build_db_snapshot() roda ranking_table() (pesado) — sem cache isso acontece
# a cada mensagem enviada no chat. Mesmo padrão usado em dashboard.py.
_SNAPSHOT_TTL = 300  # segundos
_snapshot_cache = {"value": None, "ts": 0.0}

def _build_db_snapshot_raw() -> str:
    """
    Monta um snapshot textual dos dados do banco para
    dar ao modelo contexto sobre o que existe no sistema.
    """
    lines = ["## Live Database Snapshot\n"]

    # top 20 termos por future score
    try:
        rk = ranking_table(terms_df).head(20)
        if not rk.empty:
            lines.append("### Top 20 Terms by Future Score")
            for _, row in rk.iterrows():
                lines.append(
                    f"- **{row['term']}**: score={row['future_score']}, "
                    f"growth={row['growth_%']}%, density={row['density']}, "
                    f"fusion={row['fusion']}, shift={row['shift_%']}%"
                )
    except Exception as e:
        lines.append(f"(ranking unavailable: {e})")
        logger.error("build_db_snapshot: ranking_table falhou", exc_info=True)

    # estatísticas gerais
    try:
        n_patents  = run_query("SELECT COUNT(*) AS n FROM patents").iloc[0]["n"]
        n_terms    = run_query("SELECT COUNT(*) AS n FROM term_dictionary").iloc[0]["n"]
        date_range = run_query(
            "SELECT MIN(year_month) AS mn, MAX(year_month) AS mx "
            "FROM patents WHERE year_month IS NOT NULL"
        )
        mn = date_range.iloc[0]["mn"]
        mx = date_range.iloc[0]["mx"]
        lines.append("\n### General Stats")
        lines.append(f"- Total patents: {n_patents}")
        lines.append(f"- Terms in dictionary: {n_terms}")
        lines.append(f"- Date range: {mn} → {mx}")
    except Exception as e:
        lines.append(f"(stats unavailable: {e})")
        logger.error("build_db_snapshot: stats gerais falharam", exc_info=True)



    return "\n".join(lines)

def build_db_snapshot(force_refresh: bool = False) -> str:
    """Retorna o snapshot do banco, cacheado por _SNAPSHOT_TTL segundos."""
    now = time.time()
    if (not force_refresh
            and _snapshot_cache["value"] is not None
            and (now - _snapshot_cache["ts"]) < _SNAPSHOT_TTL):
        return _snapshot_cache["value"]
    value = _build_db_snapshot_raw()
    _snapshot_cache["value"] = value
    _snapshot_cache["ts"] = now
    return value

def invalidate_db_snapshot():
    """Força recálculo do snapshot na próxima chamada (ex.: após refresh_data())."""
    _snapshot_cache["value"] = None
    _snapshot_cache["ts"] = 0.0


def build_term_detail(term: str) -> str:
    """Dados detalhados de um termo específico para injetar no contexto."""
    lines = [f"## Detail for term: '{term}'\n"]
    try:
        hist = monthly_term_count(term, terms_df)
        if hist.empty:
            return f"No data found for term '{term}'."

        h = hist.copy()
        h["year_month"] = pd.to_datetime(h["year_month"], errors="coerce")
        h = h.dropna(subset=["year_month"]).sort_values("year_month")

        lines.append(f"- Total patents: {int(h['count'].sum())}")
        lines.append(f"- Period: {h['year_month'].min().strftime('%Y-%m')} → {h['year_month'].max().strftime('%Y-%m')}")
        lines.append(f"- Monthly average: {h['count'].mean():.1f}")
        lines.append(f"- Peak: {int(h['count'].max())} in {h.loc[h['count'].idxmax(), 'year_month'].strftime('%Y-%m')}")
        lines.append(f"- Growth (last 2 months): {calc_growth(term, terms_df):.1f}%")
        lines.append(f"- Density: {calc_density(term, terms_df)}")
        lines.append(f"- Fusion (co-terms): {calc_fusion(term, terms_df)}")
        lines.append(f"- Semantic shift: {calc_shift(term, terms_df):.1f}%")
        lines.append(f"- Future Score: {calc_future_score(term, terms_df):.2f}")

        # últimos 6 meses
        last6 = h.tail(6)
        lines.append("\nLast 6 months:")
        for _, row in last6.iterrows():
            lines.append(f"  {row['year_month'].strftime('%Y-%m')}: {int(row['count'])} patents")

        # top correlações
        corr = term_correlations(term, terms_df).head(5)
        if not corr.empty:
            lines.append("\nTop co-occurring terms (by lift):")
            for _, row in corr.iterrows():
                lines.append(f"  - {row['term']}: lift={row['lift']}, jaccard={row['jaccard']}")

        # oportunidades esparsas
        opp = get_sparse_opportunities(term, C_matrix, t_map, idx_map, top_n=5)
        if not opp.empty:
            lines.append("\nTop sparse opportunities:")
            for _, row in opp.iterrows():
                lines.append(f"  - {row['term']}: bridge_strength={row['bridge_strength']}")

    except Exception as e:
        lines.append(f"(error loading detail: {e})")
        logger.error("build_term_detail falhou (term=%r)", term, exc_info=True)

    return "\n".join(lines)


def detect_mentioned_terms(user_message: str) -> list[str]:
    """
    Verifica se o usuário mencionou algum termo do dicionário na mensagem.
    Retorna lista de termos encontrados (case-insensitive).
    """
    msg_lower = user_message.lower()
    return [
        t for t in (terms_df["term"].unique() if not terms_df.empty else [])
        if t.lower() in msg_lower
    ]


SYSTEM_PROMPT = """You are an expert patent intelligence analyst assistant embedded in the Patent Foresight Lab platform.

## Your role
You have real-time access to patent filing data, term analytics, technology indicators, and forecasts from the platform's database. Your job is to help R&D Directors, Innovation Managers, and Technology Strategists understand patent trends, competitive landscapes, and innovation opportunities.

## What you CAN help with
- Patent filing trends and evolution of specific technology terms
- Technology lifecycle assessment (emerging, growing, mature, declining)
- Competitive intelligence and IP strategy
- Innovation opportunities and white-space analysis
- Interpretation of indicators (growth, density, fusion, shift, future score)
- Comparison of multiple technology terms
- Forecast interpretation
- Strategic R&D and IP recommendations based on patent data

## What you CANNOT help with
Anything unrelated to patents, technology intelligence, innovation strategy, or R&D. If asked about unrelated topics, politely decline and redirect to patent/technology analysis.

## How to respond
- Be analytical and evidence-based — always cite specific numbers from the data provided
- Be concise but complete — answer the actual question, then add one strategic implication
- Use the live data provided in the context — it reflects the current state of the database
- If a specific term is mentioned, use its detailed data to give precise answers
- Respond in the same language as the user (Portuguese if they write in Portuguese)
- Never make up data — if information is not in the context, say so clearly

## Tone
Professional, direct, strategic. Like a senior analyst presenting to a board — confident, data-driven, no fluff.
"""

def build_messages_for_api(history: list[dict], user_message: str) -> list[dict]:
    """
    Monta a lista de mensagens para a API do Gemini.
    ATENÇÃO: o Gemini usa role='model' (não 'assistant').
    """
    db_context = build_db_snapshot()

    mentioned = detect_mentioned_terms(user_message)
    term_details = ""
    if mentioned:
        term_details = "\n\n" + "\n\n".join(
            build_term_detail(t) for t in mentioned[:3]
        )

    context_block = db_context + term_details

    messages = []
    # FIX: converte role 'assistant' → 'model' para a API do Gemini
    for msg in history[-20:]:
        gemini_role = "model" if msg["role"] == "assistant" else "user"
        messages.append({
            "role": gemini_role,
            "parts": [{"text": msg["content"]}],
        })

    # mensagem atual com contexto injetado
    full_user_content = (
        f"{user_message}\n\n"
        f"---\n"
        f"[LIVE CONTEXT — use this data to answer]\n"
        f"{context_block}"
    )
    messages.append({
        "role": "user",
        "parts": [{"text": full_user_content}],
    })

    return messages


def chat(session_id: str, user_message: str) -> str:
    """
    Recebe mensagem do usuário, consulta banco, chama Gemini,
    salva e retorna resposta.
    """
    # salva mensagem do usuário primeiro
    save_message(session_id, "user", user_message)

    # carrega histórico e remove a última mensagem (que acabamos de salvar)
    # para não duplicar na construção das mensagens da API
    history = load_history(session_id)
    # FIX: remove apenas a última entrada (a que acabou de ser salva),
    # não todas as ocorrências da mesma mensagem
    history_without_last = history[:-1] if history else []

    # monta mensagens com contexto
    messages = build_messages_for_api(history_without_last, user_message)

    # separa histórico anterior da mensagem atual
    api_history = messages[:-1]
    current_msg = messages[-1]["parts"][0]["text"]

    # chama Gemini via llm.py (client centralizado; cuida de chave ausente)
    try:
        chat_obj = create_chat(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=2000,
            history=api_history,
        )
        if chat_obj is None:
            reply = "⚠️ GEMINI_API_KEY not set in .env"
        else:
            response = chat_obj.send_message(current_msg)
            reply = response.text.strip()
    except Exception as e:
        reply = f"⚠️ Erro ao consultar IA: {e}"
        logger.error("chat(): falha ao consultar Gemini (session=%s)", session_id, exc_info=True)

    # salva resposta do assistente
    save_message(session_id, "assistant", reply)
    return reply