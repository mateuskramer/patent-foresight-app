"""
dashboard.py  — v4
Chat iterativo + controles inline + drill-down

Mudanças v4:
- Gemini centralizado em llm.py (generate_json com schema validado)
- Validação de `terms` contra term_list antes de renderizar
- Cache com TTL para _db_context() (evita rodar ranking_table a cada mensagem)
- logging no lugar de print
- correção de truncamento em logs
"""

import json, uuid, time, logging
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import pearsonr
from llm import generate_json
import dash
from dash import html, dcc, Input, Output, State, callback_context, ALL, ctx
import dash_bootstrap_components as dbc

from data import (
    BG, CARD, BLUE, TEXT, MUTED, BORDER, PALETTE,

    run_query, run_write,
    terms_df, monthly_term_count,
    calc_growth, calc_density, calc_fusion, calc_shift, calc_future_score,
    ranking_table, term_correlations, build_temporal_matrix, pearson_with_term,
    get_sparse_opportunities, C_matrix, t_map, idx_map,
    term_list,
)

logger = logging.getLogger(__name__)

# conjunto para checagem O(1) de termos válidos
_VALID_TERMS = set(term_list)

# tipos de gráfico suportados
_VALID_TYPES = {"line", "bar", "scatter", "heatmap", "waterfall", "ranking"}

# ─── Banco ────────────────────────────────────────────────────────────────────
def _ensure_tables():
    run_write("""
        CREATE TABLE IF NOT EXISTS dashboard_sessions (
            id         SERIAL PRIMARY KEY,
            session_id TEXT      NOT NULL UNIQUE,
            title      TEXT      NOT NULL DEFAULT 'Dashboard',
            spec_json  TEXT      NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_dash_sess ON dashboard_sessions(session_id);
    """)

_ensure_tables()

def save_dashboard(sid, title, spec):
    return run_write(
        "INSERT INTO dashboard_sessions (session_id, title, spec_json) "
        "VALUES (:sid, :title, :spec) "
        "ON CONFLICT (session_id) DO UPDATE SET title=:title, spec_json=:spec, created_at=NOW()",
        {"sid": sid, "title": title, "spec": json.dumps(spec, ensure_ascii=False)},
    )

def load_dashboard(sid):
    df = run_query("SELECT spec_json, title FROM dashboard_sessions WHERE session_id=:sid", {"sid": sid})
    if df.empty:
        return None
    return {"spec": json.loads(df.iloc[0]["spec_json"]), "title": df.iloc[0]["title"]}

def list_dashboards():
    return run_query(
        "SELECT session_id, title, created_at FROM dashboard_sessions ORDER BY created_at DESC LIMIT 40"
    )

def delete_dashboard(sid):
    return run_write("DELETE FROM dashboard_sessions WHERE session_id=:sid", {"sid": sid})

def rename_dashboard(sid, new_title):
    return run_write(
        "UPDATE dashboard_sessions SET title=:title WHERE session_id=:sid",
        {"sid": sid, "title": new_title}
    )



# ─── Gemini prompt ────────────────────────────────────────────────────────────
_SYSTEM = """You are a data visualization expert inside Patent Foresight Lab.
Given the CURRENT dashboard spec (JSON) and a USER REQUEST, return an UPDATED spec.

RULES:
- Return ONLY valid JSON. No markdown, no explanation.
- If current spec is {}, build from scratch.
- If current spec has charts, EDIT it: add/remove/modify as requested.
- Preserve charts NOT mentioned by the user.
- Each chart must have a UNIQUE "id" string.
- "terms" values must come EXACTLY from the AVAILABLE TERMS list.

SPEC SHAPE:
{
  "title": "string (max 60 chars)",
  "charts": [
    {
      "id":           "unique_string",
      "type":         "line"|"bar"|"scatter"|"heatmap"|"waterfall"|"ranking",
      "title":        "string",
      "terms":        ["term1", "term2"],
      "metric":       "count"|"growth"|"future_score"|"density"|"fusion"|"shift",
      "width":        6|12,
      "show_trend":   true|false,
      "compare_mode": true|false,
      "time_range":   "all"|"12m"|"24m"|"6m"
    }
  ]
}

CHART TYPE GUIDE:
- line/bar   -> time-series count of >=1 terms
- scatter    -> month-by-month correlation between exactly 2 terms
- heatmap    -> Pearson matrix for >=2 terms (use width:12)
- waterfall  -> monthly deltas for 1 term
- ranking    -> horizontal bar sorted by a scalar metric

On unsupported request return: {"title":"Unsupported","charts":[],"error":"reason"}
"""

# ─── cache com TTL do contexto do banco ───────────────────────────────────────
_DB_CTX_TTL = 300  # segundos
_db_ctx_cache = {"value": None, "ts": 0.0}

def _build_db_context():
    """Versão crua (sem cache) — monta o snapshot textual do banco."""
    try:
        n  = run_query("SELECT COUNT(*) AS n FROM patents").iloc[0]["n"]
        dr = run_query("SELECT MIN(year_month) mn, MAX(year_month) mx FROM patents WHERE year_month IS NOT NULL")
        stats = f"Patents: {n} | Period: {dr.iloc[0]['mn']} -> {dr.iloc[0]['mx']}"
    except Exception:
        stats = "Stats unavailable"
    ranking_block = ""
    try:
        rk = ranking_table(terms_df).head(30)
        if not rk.empty:
            lines = ["TOP 30 TERMS BY FUTURE SCORE (use these exact strings in 'terms'):"]
            for _, row in rk.iterrows():
                lines.append(
                    f'  "{row["term"]}": future_score={row["future_score"]}, '
                    f'growth={row["growth_%"]}%, density={row["density"]}, '
                    f'fusion={row["fusion"]}, shift={row["shift_%"]}%'
                )
            ranking_block = "\n".join(lines)
    except Exception as e:
        ranking_block = f"Ranking unavailable: {e}"

    all_terms = ", ".join(f'"{t}"' for t in term_list[:80])
    extra = max(0, len(term_list) - 80)
    terms_block = "ALL AVAILABLE TERMS: " + all_terms + (f" ... and {extra} more." if extra else "")

    return f"{stats}\n\n{ranking_block}\n\n{terms_block}"

def _db_context(force_refresh: bool = False):
    """Retorna o contexto do banco, usando cache com TTL de _DB_CTX_TTL segundos."""
    now = time.time()
    if (not force_refresh
            and _db_ctx_cache["value"] is not None
            and (now - _db_ctx_cache["ts"]) < _DB_CTX_TTL):
        return _db_ctx_cache["value"]
    value = _build_db_context()
    _db_ctx_cache["value"] = value
    _db_ctx_cache["ts"] = now
    return value

def invalidate_db_context():
    """Força recálculo do contexto na próxima chamada (use após ingestão de dados)."""
    _db_ctx_cache["value"] = None
    _db_ctx_cache["ts"] = 0.0


# ─── sanitização da spec ──────────────────────────────────────────────────────
def _sanitize_spec(spec: dict) -> dict:
    """
    Valida e limpa a spec retornada pelo modelo:
    - filtra termos que não existem em term_list
    - marca charts cujo tipo é inválido ou que ficaram sem termos válidos
    Não levanta exceção; anota problemas em chart["_warning"] para a UI mostrar.
    """
    if not isinstance(spec, dict):
        return {"title": "Error", "charts": [], "error": "spec inválida"}

    charts = spec.get("charts", [])
    if not isinstance(charts, list):
        return {"title": spec.get("title", "Dashboard"), "charts": [], "error": "charts inválido"}

    clean_charts = []
    dropped_terms = set()

    for ch in charts:
        if not isinstance(ch, dict):
            continue

        # tipo
        ctype = ch.get("type", "line")
        if ctype not in _VALID_TYPES:
            ch["_warning"] = f"tipo '{ctype}' não suportado"
            ch["type"] = "line"

        # termos — ranking pode não depender de terms específicos
        raw_terms = ch.get("terms", []) or []
        valid = [t for t in raw_terms if t in _VALID_TERMS]
        invalid = [t for t in raw_terms if t not in _VALID_TERMS]
        if invalid:
            dropped_terms.update(invalid)
            ch["_warning"] = (
                (ch.get("_warning", "") + " | " if ch.get("_warning") else "")
                + f"termos ignorados: {', '.join(invalid)}"
            )
        ch["terms"] = valid

        # charts que exigem termos e ficaram vazios (ranking é exceção)
        if not valid and ch.get("type") != "ranking":
            ch["_warning"] = (
                (ch.get("_warning", "") + " | " if ch.get("_warning") else "")
                + "nenhum termo válido"
            )

        clean_charts.append(ch)

    spec["charts"] = clean_charts
    if dropped_terms:
        logger.warning("Termos inválidos descartados da spec: %s", ", ".join(sorted(dropped_terms)))
    return spec


def call_gemini(current_spec, user_request):
    db_ctx = _db_context()

    prompt = "\n\n".join([
        _SYSTEM,
        f"## DATABASE\n{db_ctx}",
        f"## CURRENT SPEC\n{json.dumps(current_spec, ensure_ascii=False)}",
        f"## USER REQUEST\n{user_request}",
        "Return the updated JSON spec now:",
    ])

    logger.info("Gemini call | prompt=%d chars | request=%r", len(prompt), user_request[:80])

    data, result = generate_json(prompt, temperature=0.2, max_output_tokens=4096)
    if data is None:
        logger.error("Gemini retornou erro: %s", result.error)
        return {"title": "Error", "charts": [], "error": str(result)}

    spec = _sanitize_spec(data)
    logger.info("Gemini OK | %d charts", len(spec.get("charts", [])))
    return spec

# ─── Data helpers ─────────────────────────────────────────────────────────────
def _filter_time(df, time_range):
    if df.empty or time_range == "all":
        return df
    df = df.copy()
    df["year_month"] = pd.to_datetime(df["year_month"], errors="coerce")
    df = df.dropna(subset=["year_month"])
    months = int(time_range.replace("m", ""))
    cutoff = df["year_month"].max() - pd.DateOffset(months=months)
    return df[df["year_month"] >= cutoff]

def _time_series(terms, time_range="all"):
    frames = []
    for t in terms:
        h = monthly_term_count(t, terms_df)
        if h.empty:
            continue
        h = _filter_time(h, time_range)
        h["term"] = t
        frames.append(h)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["year_month"] = pd.to_datetime(out["year_month"], errors="coerce")
    return out.sort_values("year_month")

def _trendline(fig, x, y, color, name):
    try:
        coeffs = np.polyfit(range(len(x)), np.array(y, dtype=float), 1)
        trend  = np.poly1d(coeffs)(range(len(x)))
        fig.add_trace(go.Scatter(
            x=x, y=trend, name=f"{name} trend", mode="lines",
            line=dict(color=color, dash="dot", width=1.5), opacity=0.6,
        ))
    except Exception:
        pass

def _no_data(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(color=MUTED, size=13))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
        margin=dict(l=40, r=16, t=36, b=32),
    )
    return fig

# ─── base layout dict SEM hovermode (cada renderer define o seu) ──────────────
_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter,sans-serif", size=12),
    margin=dict(l=40, r=16, t=36, b=32),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=11)),
)

# ─── Chart renderers ──────────────────────────────────────────────────────────
def render_line_bar(spec):
    terms  = spec.get("terms", [])[:6]
    ctype  = spec.get("type", "line")
    tr     = spec.get("time_range", "all")
    trend  = spec.get("show_trend", False)
    data   = _time_series(terms, tr)
    fig    = go.Figure()
    if data.empty:
        return _no_data("No data for selected terms")
    for i, term in enumerate(terms):
        sub = data[data["term"] == term]
        if sub.empty:
            continue
        col = PALETTE[i % len(PALETTE)]
        x, y = sub["year_month"], sub["count"]
        if ctype == "bar":
            fig.add_trace(go.Bar(
                x=x, y=y, name=term, marker_color=col, opacity=0.85,
                customdata=[[term]] * len(x),
                hovertemplate="<b>%{x|%Y-%m}</b><br>" + term + ": %{y}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y, name=term, mode="lines+markers",
                line=dict(color=col, width=2), marker=dict(size=5),
                customdata=[[term]] * len(x),
                hovertemplate="<b>%{x|%Y-%m}</b><br>" + term + ": %{y}<extra></extra>",
            ))
            if trend:
                _trendline(fig, x, y.values, col, term)
    if ctype == "bar" and spec.get("compare_mode", True):
        fig.update_layout(barmode="group")
    fig.update_layout(**_BASE, hovermode="x unified",
                      xaxis_title="Month", yaxis_title="Patent count")
    return fig

def render_scatter(spec):
    terms = spec.get("terms", [])[:2]
    tr    = spec.get("time_range", "all")
    trend = spec.get("show_trend", False)
    if len(terms) < 2:
        return _no_data("Scatter needs 2 terms")
    h1 = _filter_time(monthly_term_count(terms[0], terms_df), tr)
    h2 = _filter_time(monthly_term_count(terms[1], terms_df), tr)
    if h1.empty or h2.empty:
        return _no_data("Insufficient data")
    m = h1.merge(h2, on="year_month", suffixes=("_x", "_y"))
    if m.empty:
        return _no_data()
    x_vals, y_vals = m["count_x"], m["count_y"]
    try:
        r, p = pearsonr(x_vals, y_vals)
        corr = f"r={r:+.3f}  p={p:.3f}"
    except Exception:
        corr = ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="markers",
        marker=dict(
            color=list(range(len(m))), colorscale="Blues", size=9, showscale=True,
            colorbar=dict(title="Time", thickness=10,
                          tickvals=[0, len(m) - 1],
                          ticktext=[str(m["year_month"].iloc[0])[:7],
                                    str(m["year_month"].iloc[-1])[:7]]),
        ),
        text=m["year_month"].astype(str),
        customdata=[[terms[0]]] * len(m),
        hovertemplate="<b>%{text}</b><br>" + terms[0] + ": %{x}<br>" + terms[1] + ": %{y}<extra></extra>",
    ))
    if trend:
        try:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            xl = np.linspace(x_vals.min(), x_vals.max(), 80)
            fig.add_trace(go.Scatter(
                x=xl, y=np.poly1d(coeffs)(xl), mode="lines",
                line=dict(color="#f39c12", dash="dash", width=2), name="Regression",
            ))
        except Exception:
            pass
    sub = f"<br><sub>{corr}</sub>" if corr else ""
    fig.update_layout(
        **_BASE, hovermode="closest",
        xaxis_title=terms[0], yaxis_title=terms[1],
        title=dict(text=spec.get("title", "") + sub, font=dict(size=12)),
    )
    return fig

def render_heatmap(spec):
    terms = spec.get("terms", [])[:12]
    valid = [t for t in terms if t in terms_df["term"].values]
    if len(valid) < 2:
        return _no_data("Need >=2 valid terms")
    sub   = terms_df[terms_df["term"].isin(valid)]
    pivot = build_temporal_matrix(sub)[valid]
    n     = len(valid)
    mat   = np.zeros((n, n))
    for i, t1 in enumerate(valid):
        for j, t2 in enumerate(valid):
            if i == j:
                mat[i][j] = 1.0
            else:
                try:
                    r, _ = pearsonr(pivot[t1], pivot[t2])
                    mat[i][j] = 0.0 if np.isnan(r) else round(r, 3)
                except Exception:
                    pass
    fig = go.Figure(go.Heatmap(
        z=mat, x=valid, y=valid,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in mat],
        texttemplate="%{text}",
        colorbar=dict(title="r", thickness=12, tickvals=[-1, -0.5, 0, 0.5, 1]),
        customdata=[[t] * n for t in valid],
        hovertemplate="<b>%{y} x %{x}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **_BASE, hovermode="closest",
        xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig

def render_waterfall(spec):
    terms = spec.get("terms", [])[:1]
    tr    = spec.get("time_range", "all")
    if not terms:
        return _no_data("Select a term")
    h = _filter_time(monthly_term_count(terms[0], terms_df), tr)
    if h.empty or len(h) < 2:
        return _no_data("Insufficient data")
    h["year_month"] = pd.to_datetime(h["year_month"], errors="coerce")
    h = h.sort_values("year_month")
    deltas   = h["count"].diff().fillna(h["count"].iloc[0]).tolist()
    labels   = h["year_month"].dt.strftime("%Y-%m").tolist()
    measures = ["absolute"] + ["relative"] * (len(deltas) - 1)
    fig = go.Figure(go.Waterfall(
        x=labels, y=deltas, measure=measures,
        connector=dict(line=dict(color="rgba(255,255,255,0.1)", width=1)),
        increasing=dict(marker_color="#22c55e"),
        decreasing=dict(marker_color="#e74c3c"),
        totals=dict(marker_color=BLUE),
        text=[f"{'+' if d > 0 else ''}{int(d)}" for d in deltas],
        textposition="outside", textfont=dict(size=10, color=TEXT),
        customdata=[[terms[0]]] * len(labels),
        hovertemplate="<b>%{x}</b><br>delta: %{y}<extra></extra>",
    ))
    fig.update_layout(**_BASE, hovermode="x unified",
                      xaxis_tickangle=-35, showlegend=False)
    return fig

def render_ranking(spec):
    metric = spec.get("metric", "future_score")
    rk     = ranking_table(terms_df)
    if rk.empty:
        return _no_data()
    req = spec.get("terms", [])
    if req:
        rk = rk[rk["term"].isin(req)]
    valid_metrics = {"future_score", "growth_%", "density", "fusion", "shift_%"}
    col = metric if metric in valid_metrics else "future_score"
    top = rk.nlargest(15, col).sort_values(col)
    med = top[col].median()
    colors = [BLUE if v >= med else MUTED for v in top[col]]
    fig = go.Figure(go.Bar(
        x=top[col], y=top["term"], orientation="h",
        marker_color=colors,
        text=top[col].round(2), textposition="outside", textfont=dict(size=11),
        customdata=[[t] for t in top["term"]],
        hovertemplate="<b>%{y}</b><br>" + col + ": %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_BASE, hovermode="y unified",
        xaxis_title=col.replace("_", " ").title(),
        height=max(300, len(top) * 28), showlegend=False,
    )
    return fig

def render_chart(spec):
    t = spec.get("type", "line")
    if t in ("line", "bar"): return render_line_bar(spec)
    if t == "scatter":       return render_scatter(spec)
    if t == "heatmap":       return render_heatmap(spec)
    if t == "waterfall":     return render_waterfall(spec)
    if t == "ranking":       return render_ranking(spec)
    return _no_data(f"Unknown type: {t}")

# ─── Feature 2 — controles inline ────────────────────────────────────────────
def _ctrl_btn(label, btn_id, active=False):
    return html.Button(label, id=btn_id, n_clicks=0, style={
        "background":   "rgba(37,99,235,0.35)" if active else "rgba(255,255,255,0.05)",
        "border":       "1px solid rgba(37,99,235,0.5)" if active else "1px solid rgba(255,255,255,0.1)",
        "color":        "#93c5fd" if active else MUTED,
        "borderRadius": "6px", "padding": "2px 8px",
        "fontSize":     "11px", "cursor": "pointer",
        "fontWeight":   "600" if active else "400",
        "transition":   "all 0.15s",
    })

def _chart_card(ch, card_idx):
    ctype = ch.get("type", "line")
    tr    = ch.get("time_range", "all")
    trend = ch.get("show_trend", False)
    w     = ch.get("width", 6)
    cid   = ch.get("id", f"chart_{card_idx}")
    warning = ch.get("_warning")

    try:
        fig = render_chart(ch)
    except Exception as e:
        logger.error("Erro ao renderizar chart %s: %s", cid, e, exc_info=True)
        fig = _no_data(f"Render error: {e}")

    graph = dcc.Graph(
        id={"type": "db-graph", "index": cid},
        figure=fig,
        config={"displayModeBar": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        style={"height": "268px"},
        clear_on_unhover=True,
    )

    controls = []
    if ctype in ("line", "bar"):
        controls += [
            _ctrl_btn("Line", {"type": "db-ctrl-type", "index": cid, "val": "line"}, active=(ctype == "line")),
            _ctrl_btn("Bar",  {"type": "db-ctrl-type", "index": cid, "val": "bar"},  active=(ctype == "bar")),
        ]
    if ctype in ("line", "bar", "scatter", "waterfall"):
        controls += [
            _ctrl_btn("6m",  {"type": "db-ctrl-time", "index": cid, "val": "6m"},  active=(tr == "6m")),
            _ctrl_btn("12m", {"type": "db-ctrl-time", "index": cid, "val": "12m"}, active=(tr == "12m")),
            _ctrl_btn("24m", {"type": "db-ctrl-time", "index": cid, "val": "24m"}, active=(tr == "24m")),
            _ctrl_btn("All", {"type": "db-ctrl-time", "index": cid, "val": "all"}, active=(tr == "all")),
        ]
    if ctype in ("line", "scatter"):
        controls.append(
            _ctrl_btn("Trend~", {"type": "db-ctrl-trend", "index": cid, "val": "toggle"}, active=trend)
        )

    ctrl_bar = html.Div(controls, style={
        "display": "flex", "gap": "4px", "flexWrap": "wrap",
        "marginBottom": "8px", "alignItems": "center",
    }) if controls else html.Div()

    badge = html.Span(ctype.upper(), style={
        "fontSize": "9px", "color": BLUE, "letterSpacing": "1.5px",
        "fontWeight": "700", "marginLeft": "8px",
        "background": "rgba(37,99,235,0.12)", "padding": "2px 6px", "borderRadius": "4px",
    })

    # banner de aviso quando termos foram ignorados / tipo trocado
    warn_el = html.Div()
    if warning:
        warn_el = html.Div([
            html.I(className="fas fa-triangle-exclamation me-1",
                   style={"fontSize": "10px"}),
            warning,
        ], style={
            "fontSize": "10.5px", "color": "#fbbf24",
            "background": "rgba(251,191,36,0.08)",
            "border": "1px solid rgba(251,191,36,0.25)",
            "borderRadius": "6px", "padding": "4px 8px", "marginBottom": "8px",
        })

    card = dbc.Card(dbc.CardBody([
        html.Div([
            html.Span(ch.get("title", "Chart"),
                      style={"fontSize": "13px", "fontWeight": "600", "color": TEXT}),
            badge,
        ], style={"marginBottom": "8px", "display": "flex", "alignItems": "center"}),
        warn_el,
        ctrl_bar,
        graph,
    ]), style={
        "background": "rgba(12,12,18,0.97)",
        "border":     f"1px solid {BORDER}",
        "borderRadius": "14px",
    })

    return dbc.Col(card, md=w, style={"marginBottom": "16px"})

# ─── Feature 3 — drill-down panel ────────────────────────────────────────────
def _drill_panel(term):
    h = monthly_term_count(term, terms_df)
    h["year_month"] = pd.to_datetime(h["year_month"], errors="coerce")
    h = h.dropna(subset=["year_month"]).sort_values("year_month")

    spark = go.Figure()
    if not h.empty:
        spark.add_trace(go.Scatter(
            x=h["year_month"], y=h["count"], mode="lines",
            line=dict(color=BLUE, width=2),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.12)",
        ))
    spark.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), height=80,
    )

    g   = calc_growth(term, terms_df)
    den = calc_density(term, terms_df)
    fus = calc_fusion(term, terms_df)
    sh  = calc_shift(term, terms_df)
    fs  = calc_future_score(term, terms_df)
    g_color = "#22c55e" if g > 0 else "#e74c3c" if g < 0 else MUTED

    def kpi(label, val, color=TEXT):
        return html.Div([
            html.Div(label, style={"fontSize": "9px", "color": MUTED,
                                   "letterSpacing": "1px", "marginBottom": "2px"}),
            html.Div(val,   style={"fontSize": "15px", "fontWeight": "700", "color": color}),
        ], style={"background": "rgba(255,255,255,0.03)", "borderRadius": "8px",
                  "padding": "7px 8px", "flex": "1", "textAlign": "center"})

    kpi_row = html.Div([
        kpi("GROWTH",  f"{g:+.1f}%", g_color),
        kpi("DENSITY", str(den)),
        kpi("FUSION",  str(fus)),
        kpi("SHIFT",   f"{sh:.1f}%"),
        kpi("SCORE",   f"{fs:.2f}", BLUE),
    ], style={"display": "flex", "gap": "5px", "flexWrap": "wrap", "marginBottom": "14px"})

    corr_df  = term_correlations(term, terms_df).head(5)
    corr_els = []
    if not corr_df.empty:
        for _, row in corr_df.iterrows():
            corr_els.append(html.Div([
                html.Span(row["term"], style={"fontSize": "12px", "color": TEXT, "flex": "1"}),
                html.Span(f"lift {row['lift']:.2f}", style={"fontSize": "11px", "color": BLUE}),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "padding": "5px 0", "borderBottom": f"1px solid {BORDER}"}))

    opp_df  = get_sparse_opportunities(term, C_matrix, t_map, idx_map, top_n=3)
    opp_els = []
    if not opp_df.empty:
        for _, row in opp_df.iterrows():
            opp_els.append(html.Div([
                html.Span(row["term"], style={"fontSize": "12px", "color": TEXT, "flex": "1"}),
                html.Span(f"bridge {row['bridge_strength']}", style={"fontSize": "11px", "color": "#a78bfa"}),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "padding": "5px 0", "borderBottom": f"1px solid {BORDER}"}))

    return html.Div([
        html.Div([
            html.Div([
                html.Div("DRILL-DOWN", style={"fontSize": "9px", "color": MUTED,
                                              "letterSpacing": "2px", "marginBottom": "2px"}),
                html.Div(term, style={"fontSize": "14px", "fontWeight": "700", "color": TEXT}),
            ]),
            html.Button("✕", id="db-drill-close", n_clicks=0, style={
                "background": "transparent", "border": "none", "color": MUTED,
                "fontSize": "18px", "cursor": "pointer", "padding": "4px 8px",
            }),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "flex-start", "marginBottom": "12px"}),
        dcc.Graph(figure=spark, config={"displayModeBar": False}, style={"marginBottom": "12px"}),
        kpi_row,
        html.Div("TOP CO-OCORRÊNCIAS", style={"fontSize": "9px", "color": MUTED,
                                              "letterSpacing": "1.5px", "marginBottom": "6px"}),
        html.Div(corr_els or [html.Div("—", style={"color": MUTED, "fontSize": "12px"})],
                 style={"marginBottom": "14px"}),
        html.Div("OPORTUNIDADES", style={"fontSize": "9px", "color": "#a78bfa",
                                         "letterSpacing": "1.5px", "marginBottom": "6px"}),
        html.Div(opp_els or [html.Div("—", style={"color": MUTED, "fontSize": "12px"})]),
    ], style={
        "position": "absolute", "top": "0", "right": "0", "bottom": "0", "width": "300px",
        "background": "#08080f", "borderLeft": f"1px solid {BORDER}",
        "padding": "18px 16px", "overflowY": "auto", "zIndex": "10",
        "animation": "slideInRight 0.22s ease",
    })

# ─── Grid e helpers de UI ─────────────────────────────────────────────────────
def _empty_canvas(msg="Descreva o dashboard no chat →"):
    return html.Div([
        html.I(className="fas fa-chart-area",
               style={"fontSize": "44px", "color": "#1f2937", "marginBottom": "14px"}),
        html.P(msg, style={"color": MUTED, "fontSize": "14px", "textAlign": "center",
                           "maxWidth": "300px", "lineHeight": "1.6"}),
    ], style={"display": "flex", "flexDirection": "column", "alignItems": "center",
              "justifyContent": "center", "height": "100%", "padding": "60px 20px"})

def _spec_to_grid(spec):
    charts = spec.get("charts", [])
    if not charts:
        return _empty_canvas(f"⚠️ {spec.get('error', 'Nenhum gráfico gerado.')}")
    cols = [_chart_card(ch, i) for i, ch in enumerate(charts)]
    return html.Div(dbc.Row(cols, className="g-3"))

def _history_item(sid, title, created_at, active=False):
    try:    date = pd.to_datetime(created_at).strftime("%d/%m %H:%M")
    except: date = str(created_at)[:16]
    return html.Div([
        html.Div(title[:30] + ("…" if len(title) > 30 else ""),
                 style={"fontSize": "12px", "color": "white" if active else "#9ca3af",
                        "fontWeight": "600" if active else "400", "marginBottom": "1px",
                        "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
        html.Div(date, style={"fontSize": "10px", "color": MUTED}),
    ], id={"type": "db-history-item", "index": sid}, n_clicks=0, style={
        "padding": "9px 10px", "borderRadius": "10px", "cursor": "pointer", "marginBottom": "3px",
        "background": "rgba(37,99,235,0.18)" if active else "transparent",
        "border": f"1px solid {'rgba(37,99,235,0.45)' if active else 'transparent'}",
        "transition": "all 0.18s",
    })

def _meta(spec):
    charts = spec.get("charts", [])
    if not charts:
        return "Nenhum gráfico"
    types = ", ".join(sorted({c.get("type", "?") for c in charts}))
    return f"{len(charts)} gráfico(s) • {types}"

def _bubble(role, text):
    is_user = role == "user"
    return html.Div(text, style={
        "background":   "rgba(37,99,235,0.85)" if is_user else "rgba(37,99,235,0.08)",
        "border":       "none" if is_user else "1px solid rgba(37,99,235,0.2)",
        "borderRadius": "12px 4px 12px 12px" if is_user else "4px 12px 12px 12px",
        "padding":      "9px 13px", "fontSize": "12.5px", "lineHeight": "1.6",
        "color":        "white" if is_user else "#d1d5db",
        "alignSelf":    "flex-end" if is_user else "flex-start",
        "maxWidth":     "92%", "whiteSpace": "pre-wrap",
    })

EXAMPLES = [
    "Tendência dos 3 termos com maior future score",
    "Correlação electric vehicle vs battery — últimos 24 meses",
    "Heatmap dos top 8 termos",
    "Ranking por growth%",
    "Waterfall mensal do termo mais popular",
    "Compare AI vs machine learning vs deep learning",
]

# ─── Layout ───────────────────────────────────────────────────────────────────
def dashboard_builder_layout(search=None):
    # se veio ?sid=xxx da galeria, carrega a spec já no layout
    preload_sid, preload_spec, preload_title = None, {}, "Dashboard"
    if search:
        import urllib.parse
        qs = urllib.parse.parse_qs(search.lstrip("?"))
        sid = qs.get("sid", [None])[0]
        if sid:
            data = load_dashboard(sid)
            if data:
                preload_sid   = sid
                preload_spec  = data["spec"]
                preload_title = data["title"]

    initial_canvas = _spec_to_grid(preload_spec) if preload_spec.get("charts") else _empty_canvas()
    initial_meta   = _meta(preload_spec) if preload_spec.get("charts") else "Nenhum dashboard carregado"

    return html.Div([
        dcc.Store(id="db-current-session", data=preload_sid),
        dcc.Store(id="db-current-spec",    data=preload_spec),
        dcc.Store(id="db-drill-term",      data=None),
        dcc.Store(id="db-sidebar-collapsed", data=False),
        dcc.Interval(id="db-history-refresh", interval=500, n_intervals=0, max_intervals=1),

        # Modal de Renomear
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Renomear Dashboard")),
            dbc.ModalBody([
                html.Label("Novo título:", style={"color": MUTED, "fontSize": "12px", "marginBottom": "6px"}),
                dbc.Input(id="db-rename-input", type="text", placeholder="Ex: Meu Dashboard", style={
                    "background": "#111827", "color": TEXT, "borderColor": BORDER
                }),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancelar", id="db-rename-cancel", className="ms-auto", n_clicks=0, style={
                    "background": "rgba(255,255,255,0.05)", "border": f"1px solid {BORDER}", "color": MUTED, "borderRadius": "8px"
                }),
                dbc.Button("Salvar", id="db-rename-save", n_clicks=0, style={
                    "background": BLUE, "border": "none", "borderRadius": "8px"
                }),
            ])
        ], id="db-rename-modal", is_open=False),

        html.Div(style={"display": "flex", "height": "calc(100vh - 80px)", "position": "relative"}, children=[

            # COL 1 — histórico
            html.Div(id="db-sidebar-container", style={
                "width": "220px", "flexShrink": "0",
                "background": "#050505", "borderRight": f"1px solid {BORDER}",
                "display": "flex", "flexDirection": "column", "overflow": "hidden",
                "transition": "width 0.2s ease-in-out, border 0.2s ease-in-out",
            }, children=[
                html.Div([
                    html.Div([
                        html.Div("DASHBOARDS", style={"fontSize": "10px", "fontWeight": "700",
                                                       "letterSpacing": "2px", "color": MUTED, "marginBottom": "0"}),
                        html.Button([html.I(className="fas fa-angle-left")], id="db-sidebar-toggle", n_clicks=0, style={
                            "background": "transparent", "border": "none", "color": MUTED, "cursor": "pointer",
                            "fontSize": "14px", "padding": "2px 6px", "outline": "none"
                        }),
                    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"}),
                    html.Button([html.I(className="fas fa-plus me-1"), "Novo"],
                                id="db-new-btn", n_clicks=0, style={
                        "background": "rgba(37,99,235,0.15)", "border": f"1px solid {BLUE}",
                        "color": "#93c5fd", "borderRadius": "8px", "padding": "6px 12px",
                        "fontSize": "12px", "cursor": "pointer", "width": "100%",
                        "marginBottom": "10px", "fontWeight": "500",
                    }),
                ], style={"padding": "16px 12px 8px"}),
                html.Div(id="db-history-list",
                         children=[],
                         style={"overflowY": "auto", "flex": "1", "padding": "0 8px 8px"}),
                html.Div([
                    html.Div([
                        html.Button([html.I(className="fas fa-edit me-1"), "Renomear"],
                                    id="db-rename-btn", n_clicks=0, style={
                            "background": "rgba(255,255,255,0.05)", "border": f"1px solid {BORDER}",
                            "color": TEXT, "borderRadius": "8px", "padding": "6px 10px",
                            "fontSize": "11px", "cursor": "pointer", "flex": "1",
                        }),
                        html.Button([html.I(className="fas fa-trash-alt me-1"), "Apagar"],
                                    id="db-delete-btn", n_clicks=0, style={
                            "background": "rgba(231,76,60,0.08)", "border": "1px solid rgba(231,76,60,0.3)",
                            "color": "#f87171", "borderRadius": "8px", "padding": "6px 10px",
                            "fontSize": "11px", "cursor": "pointer", "flex": "1",
                        }),
                    ], style={"display": "flex", "gap": "6px"}),
                ], style={"padding": "8px 12px 16px"}),
            ]),

            # COL 2 — canvas
            html.Div(style={
                "flex": "1", "display": "flex", "flexDirection": "column",
                "overflow": "hidden", "background": BG, "position": "relative",
            }, children=[
                html.Div([
                    html.Button([html.I(className="fas fa-angle-right")], id="db-sidebar-expand", n_clicks=0, style={
                        "background": "rgba(255,255,255,0.05)", "border": f"1px solid {BORDER}",
                        "color": TEXT, "borderRadius": "6px", "padding": "4px 8px", "marginRight": "12px",
                        "cursor": "pointer", "display": "none"
                    }),
                    html.Div([
                        html.Div(id="db-canvas-title", children=preload_title,
                                 style={"fontSize": "15px", "fontWeight": "700", "color": TEXT}),
                        html.Div(id="db-canvas-meta", children=initial_meta,
                                 style={"fontSize": "11px", "color": MUTED, "marginTop": "2px"}),
                    ], style={"flex": "1"}),
                    html.Button(
                        [html.I(className="fas fa-star me-2"), "Salvar na Galeria"],
                        id="db-pin-btn", n_clicks=0,
                        style={
                            "background": "rgba(52,211,153,0.12)", "border": "1px solid rgba(52,211,153,0.4)",
                            "color": "#34d399", "borderRadius": "8px", "padding": "6px 14px",
                            "fontSize": "12px", "cursor": "pointer", "fontWeight": "500",
                            "transition": "all 0.2s", "whiteSpace": "nowrap",
                        },
                    ),
                    html.Div(id="db-pin-msg", style={"fontSize": "11px", "color": "#34d399",
                                                      "marginLeft": "10px", "alignSelf": "center"}),
                ], style={"padding": "12px 18px", "borderBottom": f"1px solid {BORDER}",
                           "background": "#050505", "display": "flex", "alignItems": "center", "gap": "12px"}),
                html.Div(style={"flex": "1", "position": "relative", "overflow": "hidden"}, children=[
                    dcc.Loading(
                        html.Div(id="db-canvas-area", style={
                            "height": "100%", "overflowY": "auto", "padding": "18px",
                        }, children=[initial_canvas]),
                        type="dot", color=BLUE,
                    ),
                    html.Div(id="db-drill-panel", style={"display": "none"}),
                ]),
            ]),

            # COL 3 — chat
            html.Div(style={
                "width": "340px", "flexShrink": "0",
                "background": "#050505", "borderLeft": f"1px solid {BORDER}",
                "display": "flex", "flexDirection": "column", "overflow": "hidden",
            }, children=[
                html.Div([
                    html.Div(style={
                        "width": "28px", "height": "28px", "borderRadius": "50%",
                        "background": "rgba(37,99,235,0.2)", "display": "flex",
                        "alignItems": "center", "justifyContent": "center", "marginRight": "10px",
                    }, children=[html.I(className="fas fa-wand-magic-sparkles",
                                        style={"color": BLUE, "fontSize": "12px"})]),
                    html.Div([
                        html.Div("Dashboard AI",
                                 style={"fontSize": "13px", "fontWeight": "600", "color": TEXT}),
                        html.Div("Iterativo — edita o dashboard atual",
                                 style={"fontSize": "11px", "color": MUTED}),
                    ]),
                ], style={"padding": "13px 16px", "borderBottom": f"1px solid {BORDER}",
                           "display": "flex", "alignItems": "center"}),

                html.Div(id="db-chat-messages", style={
                    "flex": "1", "overflowY": "auto", "padding": "14px 16px",
                    "display": "flex", "flexDirection": "column", "gap": "8px",
                }, children=[
                    html.Div(
                        "Olá! Descreva o dashboard. Posso criar, editar e expandir gráficos "
                        "mantendo o contexto entre mensagens.",
                        style={"background": "rgba(37,99,235,0.08)",
                               "border": "1px solid rgba(37,99,235,0.2)",
                               "borderRadius": "4px 12px 12px 12px",
                               "padding": "10px 14px", "fontSize": "12.5px",
                               "lineHeight": "1.7", "color": "#d1d5db"},
                    ),
                ]),

                html.Div([
                    html.Div("Sugestões:", style={"fontSize": "10px", "color": MUTED,
                                                   "marginBottom": "5px", "letterSpacing": "1px"}),
                    html.Div([
                        html.Button(q, id={"type": "db-example", "index": i}, n_clicks=0, style={
                            "background": "rgba(37,99,235,0.07)",
                            "border": "1px solid rgba(37,99,235,0.2)",
                            "color": "#93c5fd", "borderRadius": "14px",
                            "padding": "3px 9px", "fontSize": "11px",
                            "cursor": "pointer", "margin": "2px 2px 0 0", "whiteSpace": "nowrap",
                        })
                        for i, q in enumerate(EXAMPLES)
                    ], style={"display": "flex", "flexWrap": "wrap"}),
                ], style={"padding": "8px 16px", "borderTop": f"1px solid {BORDER}"}),

                html.Div(id="db-typing-indicator", children="⟳ Atualizando dashboard…",
                         style={"padding": "0 16px 4px", "fontSize": "11px",
                                "color": BLUE, "display": "none"}),

                html.Div([
                    dcc.Textarea(id="db-chat-input",
                        placeholder="Ex: adiciona heatmap dos top 6 termos, muda o 1º gráfico pra 12 meses…",
                        style={
                            "flex": "1", "background": "#111827",
                            "border": f"1px solid {BORDER}", "borderRadius": "10px",
                            "color": TEXT, "padding": "10px 12px", "fontSize": "12.5px",
                            "resize": "none", "height": "68px",
                            "fontFamily": "Inter,sans-serif", "outline": "none", "lineHeight": "1.5",
                        },
                    ),
                    html.Button(html.I(className="fas fa-paper-plane"),
                                id="db-send-btn", n_clicks=0, style={
                        "background": BLUE, "border": "none", "color": "white",
                        "borderRadius": "10px", "width": "40px", "height": "68px",
                        "cursor": "pointer", "fontSize": "15px", "flexShrink": "0",
                    }),
                ], style={"padding": "8px 16px 12px", "display": "flex", "gap": "8px", "alignItems": "stretch"}),
            ]),
        ]),
    ], style={"padding": "0", "margin": "-40px", "height": "calc(100vh - 80px)", "overflow": "hidden"})

# ─── Callbacks ────────────────────────────────────────────────────────────────
def register_dashboard_callbacks(app):

    def _render_history(active_sid=None):
        df = list_dashboards()
        if df.empty:
            return [html.Div("Nenhum dashboard.", style={"color": MUTED, "fontSize": "12px", "padding": "8px 4px"})]
        return [_history_item(r["session_id"], r["title"], r["created_at"],
                              active=(r["session_id"] == active_sid))
                for _, r in df.iterrows()]

    # exemplos
    @app.callback(
        Output("db-chat-input", "value", allow_duplicate=True),
        Input({"type": "db-example", "index": ALL}, "n_clicks"),
        State({"type": "db-example", "index": ALL}, "children"),
        prevent_initial_call=True,
    )
    def fill_example(clicks, labels):
        if not ctx.triggered or not any(clicks):
            return dash.no_update
        try:
            idx = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
            return labels[idx]
        except Exception:
            return dash.no_update

    # novo
    @app.callback(
        Output("db-current-session", "data",     allow_duplicate=True),
        Output("db-current-spec",    "data",     allow_duplicate=True),
        Output("db-canvas-area",     "children", allow_duplicate=True),
        Output("db-canvas-title",    "children", allow_duplicate=True),
        Output("db-canvas-meta",     "children", allow_duplicate=True),
        Output("db-history-list",    "children", allow_duplicate=True),
        Output("db-chat-messages",   "children", allow_duplicate=True),
        Output("db-drill-panel",     "style",    allow_duplicate=True),
        Input("db-new-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def new_dashboard(n):
        if not n:
            return [dash.no_update] * 8
        welcome = [_bubble("assistant", "Novo dashboard. Descreva o que quer visualizar →")]
        return (None, {}, _empty_canvas(), "Novo Dashboard", "Nenhum gráfico ainda",
                _render_history(), welcome, {"display": "none"})

    # apagar
    @app.callback(
        Output("db-current-session", "data",     allow_duplicate=True),
        Output("db-current-spec",    "data",     allow_duplicate=True),
        Output("db-canvas-area",     "children", allow_duplicate=True),
        Output("db-canvas-title",    "children", allow_duplicate=True),
        Output("db-canvas-meta",     "children", allow_duplicate=True),
        Output("db-history-list",    "children", allow_duplicate=True),
        Output("db-drill-panel",     "style",    allow_duplicate=True),
        Input("db-delete-btn", "n_clicks"),
        State("db-current-session", "data"),
        prevent_initial_call=True,
    )
    def delete_db(n, sid):
        if not n:
            return [dash.no_update] * 7
        if sid:
            delete_dashboard(sid)
        return (None, {}, _empty_canvas("Apagado. Crie um novo →"),
                "Dashboard", "—", _render_history(), {"display": "none"})

    # carregar histórico
    @app.callback(
        Output("db-current-session", "data",     allow_duplicate=True),
        Output("db-current-spec",    "data",     allow_duplicate=True),
        Output("db-canvas-area",     "children", allow_duplicate=True),
        Output("db-canvas-title",    "children", allow_duplicate=True),
        Output("db-canvas-meta",     "children", allow_duplicate=True),
        Output("db-history-list",    "children", allow_duplicate=True),
        Output("db-drill-panel",     "style",    allow_duplicate=True),
        Input({"type": "db-history-item", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def load_from_history(clicks):
        if not ctx.triggered or not any(clicks):
            return [dash.no_update] * 7
        try:
            sid = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
        except Exception:
            return [dash.no_update] * 7
        data = load_dashboard(sid)
        if not data:
            return [dash.no_update] * 7
        spec = data["spec"]
        return (sid, spec, _spec_to_grid(spec), data["title"],
                _meta(spec), _render_history(sid), {"display": "none"})

    # FEATURE 1 — enviar mensagem iterativo
    @app.callback(
        Output("db-canvas-area",     "children", allow_duplicate=True),
        Output("db-canvas-title",    "children", allow_duplicate=True),
        Output("db-canvas-meta",     "children", allow_duplicate=True),
        Output("db-current-session", "data",     allow_duplicate=True),
        Output("db-current-spec",    "data",     allow_duplicate=True),
        Output("db-history-list",    "children", allow_duplicate=True),
        Output("db-chat-messages",   "children", allow_duplicate=True),
        Output("db-chat-input",      "value",    allow_duplicate=True),
        Output("db-drill-panel",     "style",    allow_duplicate=True),
        Input("db-send-btn", "n_clicks"),
        State("db-chat-input",     "value"),
        State("db-current-session", "data"),
        State("db-current-spec",    "data"),
        State("db-chat-messages",   "children"),
        prevent_initial_call=True,
        running=[
            (Output("db-send-btn", "disabled"), True, False),
            (Output("db-typing-indicator", "style"),
             {"padding": "0 16px 4px", "fontSize": "11px", "color": BLUE, "display": "block"},
             {"padding": "0 16px 4px", "fontSize": "11px", "color": BLUE, "display": "none"}),
        ],
    )
    def generate_dashboard(n_clicks, user_text, current_sid, current_spec, current_msgs):
        logger.info("generate_dashboard | clicks=%s text=%r", n_clicks, (user_text or "")[:80])
        if not n_clicks or not user_text or not user_text.strip():
            return [dash.no_update] * 9

        user_text    = user_text.strip()
        current_spec = current_spec or {}
        msgs         = list(current_msgs or [])
        msgs.append(_bubble("user", user_text))

        try:
            spec = call_gemini(current_spec, user_text)
        except Exception as e:
            logger.error("call_gemini falhou: %s", e, exc_info=True)
            spec = {"title": "Error", "charts": [], "error": str(e)}

        logger.info("spec | error=%s charts=%d", spec.get("error"), len(spec.get("charts", [])))

        if spec.get("error"):
            grid     = _spec_to_grid(current_spec) if current_spec.get("charts") else _empty_canvas(f"⚠️ {spec['error']}")
            title    = current_spec.get("title", "Dashboard")
            ai_reply = f"⚠️ {spec['error']}"
            new_spec = current_spec
        else:
            try:
                grid = _spec_to_grid(spec)
            except Exception as e:
                logger.error("_spec_to_grid falhou: %s", e, exc_info=True)
                grid = _empty_canvas(f"⚠️ Erro ao renderizar: {e}")
            title      = spec.get("title", "Dashboard")
            n_ch       = len(spec.get("charts", []))
            types      = ", ".join(sorted({c.get("type", "?") for c in spec.get("charts", [])}))
            terms_used = ", ".join(sorted({t for c in spec.get("charts", []) for t in c.get("terms", [])}))
            # avisos de sanitização agregados
            warns = [c["_warning"] for c in spec.get("charts", []) if c.get("_warning")]
            warn_line = ("\n⚠️ " + " ; ".join(warns)) if warns else ""
            ai_reply   = f"✅ {title}\n{n_ch} gráfico(s): {types}\nTermos: {terms_used}{warn_line}"
            new_spec   = spec

        sid = current_sid or str(uuid.uuid4())
        if not spec.get("error"):
            saved = save_dashboard(sid, title, new_spec)
            if saved:
                logger.info("dashboard salvo sid=%s", sid)
            else:
                # run_write não levanta exceção em falha — precisamos checar
                # o retorno explicitamente, senão a falha passa despercebida.
                logger.error("save_dashboard retornou False (sid=%s) — ver log de 'Erro na escrita' acima", sid)
                ai_reply += "\n⚠️ Falha ao salvar no banco — veja o log do servidor para detalhes."

        msgs.append(_bubble("assistant", ai_reply))

        return (grid, title, _meta(new_spec), sid, new_spec,
                _render_history(sid), msgs, "", {"display": "none"})

    # FEATURE 2a — tipo line/bar
    @app.callback(
        Output("db-canvas-area",  "children", allow_duplicate=True),
        Output("db-current-spec", "data",     allow_duplicate=True),
        Input({"type": "db-ctrl-type", "index": ALL, "val": ALL}, "n_clicks"),
        State("db-current-spec", "data"),
        prevent_initial_call=True,
    )
    def ctrl_type(clicks, spec):
        if not ctx.triggered or not any(c for c in clicks if c):
            return [dash.no_update] * 2
        try:
            prop = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
            cid, val = prop["index"], prop["val"]
        except Exception:
            return [dash.no_update] * 2
        spec = spec or {}
        for ch in spec.get("charts", []):
            if ch.get("id") == cid:
                ch["type"] = val
                break
        return _spec_to_grid(spec), spec

    # FEATURE 2b — período
    @app.callback(
        Output("db-canvas-area",  "children", allow_duplicate=True),
        Output("db-current-spec", "data",     allow_duplicate=True),
        Input({"type": "db-ctrl-time", "index": ALL, "val": ALL}, "n_clicks"),
        State("db-current-spec", "data"),
        prevent_initial_call=True,
    )
    def ctrl_time(clicks, spec):
        if not ctx.triggered or not any(c for c in clicks if c):
            return [dash.no_update] * 2
        try:
            prop = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
            cid, val = prop["index"], prop["val"]
        except Exception:
            return [dash.no_update] * 2
        spec = spec or {}
        for ch in spec.get("charts", []):
            if ch.get("id") == cid:
                ch["time_range"] = val
                break
        return _spec_to_grid(spec), spec

    # FEATURE 2c — trendline
    @app.callback(
        Output("db-canvas-area",  "children", allow_duplicate=True),
        Output("db-current-spec", "data",     allow_duplicate=True),
        Input({"type": "db-ctrl-trend", "index": ALL, "val": ALL}, "n_clicks"),
        State("db-current-spec", "data"),
        prevent_initial_call=True,
    )
    def ctrl_trend(clicks, spec):
        if not ctx.triggered or not any(c for c in clicks if c):
            return [dash.no_update] * 2
        try:
            prop = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
            cid  = prop["index"]
        except Exception:
            return [dash.no_update] * 2
        spec = spec or {}
        for ch in spec.get("charts", []):
            if ch.get("id") == cid:
                ch["show_trend"] = not ch.get("show_trend", False)
                break
        return _spec_to_grid(spec), spec

    # FEATURE 3 — drill-down por clique
    @app.callback(
        Output("db-drill-panel", "children", allow_duplicate=True),
        Output("db-drill-panel", "style",    allow_duplicate=True),
        Output("db-drill-term",  "data",     allow_duplicate=True),
        Input({"type": "db-graph", "index": ALL}, "clickData"),
        prevent_initial_call=True,
    )
    def open_drill(click_data_list):
        if not ctx.triggered:
            return [dash.no_update] * 3
        term = None
        for cd in click_data_list:
            if not cd or not cd.get("points"):
                continue
            pt = cd["points"][0]
            if pt.get("customdata"):
                term = pt["customdata"][0]
                break
            if pt.get("y") and isinstance(pt["y"], str):
                term = pt["y"]
                break
        if not term or term not in terms_df["term"].values:
            return [dash.no_update] * 3
        panel = _drill_panel(term)
        style = {"position": "absolute", "top": "0", "right": "0",
                 "bottom": "0", "width": "300px", "display": "block"}
        return panel, style, term

    # fechar drill
    @app.callback(
        Output("db-drill-panel", "style", allow_duplicate=True),
        Output("db-drill-term",  "data",  allow_duplicate=True),
        Input("db-drill-close", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_drill(n):
        if not n:
            return [dash.no_update] * 2
        return {"display": "none"}, None

    # salvar na galeria
    @app.callback(
        Output("db-pin-msg", "children"),
        Input("db-pin-btn",  "n_clicks"),
        State("db-current-session", "data"),
        State("db-canvas-title",    "children"),
        prevent_initial_call=True,
    )
    def pin_current(n, sid, title):
        if not n or not sid:
            return ""
        ok = pin_dashboard(sid)
        if ok:
            return "✓ Salvo!"
        logger.error("pin_dashboard retornou False (sid=%s)", sid)
        return "❌ Erro ao salvar — veja o log do servidor."

    # init histórico — apenas popula a lista lateral
    @app.callback(
        Output("db-history-list",   "children"),
        Input("db-history-refresh", "n_intervals"),
        State("db-current-session", "data"),
        prevent_initial_call=False,
    )
    def init_history(n, active_sid):
        return _render_history(active_sid)

    # Minimizar / Expandir Sidebar
    @app.callback(
        Output("db-sidebar-collapsed", "data"),
        Output("db-sidebar-container", "style"),
        Output("db-sidebar-expand", "style"),
        Input("db-sidebar-toggle", "n_clicks"),
        Input("db-sidebar-expand", "n_clicks"),
        State("db-sidebar-collapsed", "data"),
        prevent_initial_call=True,
    )
    def toggle_sidebar(toggle_clicks, expand_clicks, collapsed):
        triggered_id = ctx.triggered_id
        if triggered_id == "db-sidebar-toggle":
            collapsed = True
        elif triggered_id == "db-sidebar-expand":
            collapsed = False
        
        if collapsed:
            container_style = {
                "width": "0px", "flexShrink": "0",
                "background": "#050505", "borderRight": "0px solid rgba(0,0,0,0)",
                "display": "flex", "flexDirection": "column", "overflow": "hidden",
                "transition": "width 0.2s ease-in-out, border 0.2s ease-in-out"
            }
            expand_style = {
                "background": "rgba(255,255,255,0.05)", "border": f"1px solid {BORDER}",
                "color": TEXT, "borderRadius": "6px", "padding": "4px 8px", "marginRight": "12px",
                "cursor": "pointer", "display": "block"
            }
        else:
            container_style = {
                "width": "220px", "flexShrink": "0",
                "background": "#050505", "borderRight": f"1px solid {BORDER}",
                "display": "flex", "flexDirection": "column", "overflow": "hidden",
                "transition": "width 0.2s ease-in-out, border 0.2s ease-in-out"
            }
            expand_style = {
                "background": "rgba(255,255,255,0.05)", "border": f"1px solid {BORDER}",
                "color": TEXT, "borderRadius": "6px", "padding": "4px 8px", "marginRight": "12px",
                "cursor": "pointer", "display": "none"
            }
        return collapsed, container_style, expand_style

    # Callback do Modal de Renomear
    @app.callback(
        Output("db-rename-modal",   "is_open"),
        Output("db-rename-input",   "value"),
        Output("db-canvas-title",    "children", allow_duplicate=True),
        Output("db-history-list",    "children", allow_duplicate=True),
        Input("db-rename-btn",      "n_clicks"),
        Input("db-rename-cancel",   "n_clicks"),
        Input("db-rename-save",     "n_clicks"),
        State("db-rename-input",    "value"),
        State("db-current-session", "data"),
        State("db-canvas-title",    "children"),
        prevent_initial_call=True,
    )
    def handle_rename_modal(n_open, n_cancel, n_save, new_title_val, active_sid, current_title):
        triggered_id = ctx.triggered_id
        
        if triggered_id == "db-rename-btn":
            if not active_sid:
                return dash.no_update
            return True, current_title, dash.no_update, dash.no_update
            
        elif triggered_id == "db-rename-cancel":
            return False, "", dash.no_update, dash.no_update
            
        elif triggered_id == "db-rename-save":
            if not active_sid or not new_title_val or not new_title_val.strip():
                return False, "", dash.no_update, dash.no_update
            
            new_title = new_title_val.strip()
            ok = rename_dashboard(active_sid, new_title)
            if ok:
                return False, "", new_title, _render_history(active_sid)
            else:
                return False, "", dash.no_update, dash.no_update
                
        return dash.no_update

    # Ativar/Desativar botões da barra lateral dependendo se há uma sessão ativa
    @app.callback(
        Output("db-rename-btn", "disabled"),
        Output("db-delete-btn", "disabled"),
        Input("db-current-session", "data")
    )
    def toggle_sidebar_buttons(active_sid):
        has_session = active_sid is not None
        return not has_session, not has_session

    _register_gallery_callbacks(app)


# ═══════════════════════════════════════════════════════════════════════════════
# GALERIA — página /dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_gallery_col():
    """Adiciona coluna pinned à tabela se ainda não existir."""
    run_write("""
        ALTER TABLE dashboard_sessions
        ADD COLUMN IF NOT EXISTS pinned BOOLEAN DEFAULT FALSE;
    """)

_ensure_gallery_col()

def pin_dashboard(sid: str):
    return run_write("UPDATE dashboard_sessions SET pinned=TRUE WHERE session_id=:sid", {"sid": sid})

def unpin_dashboard(sid: str):
    return run_write("UPDATE dashboard_sessions SET pinned=FALSE WHERE session_id=:sid", {"sid": sid})

def list_pinned() -> pd.DataFrame:
    return run_query(
        "SELECT session_id, title, created_at FROM dashboard_sessions "
        "WHERE pinned=TRUE ORDER BY created_at DESC"
    )




def _gallery_card(sid: str, title: str, spec: dict) -> dbc.Col:
    """Card de preview na galeria com mini-gráficos e botão remover."""
    charts = spec.get("charts", [])
    previews = []
    for ch in charts[:2]:           # máx 2 mini-gráficos por card
        try:
            fig = render_chart(ch)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=110,
            )
            previews.append(dcc.Graph(
                figure=fig,
                config={"displayModeBar": False},
                style={"marginBottom": "6px"},
            ))
        except Exception:
            pass

    n_charts = len(charts)
    types    = " · ".join(sorted({c.get("type","?") for c in charts}))

    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.Div([
                html.Div(title, style={"fontSize":"14px","fontWeight":"700","color":"var(--text)","flex":"1"}),
                html.Button("✕", id={"type":"gallery-unpin","index":sid}, n_clicks=0, style={
                    "background":"transparent","border":"none","color":"var(--muted)",
                    "fontSize":"16px","cursor":"pointer","padding":"0 4px","lineHeight":"1",
                }),
            ], style={"display":"flex","alignItems":"center","marginBottom":"10px"}),
            html.Div(previews or [html.Div("Sem preview", style={"color":"var(--muted)","fontSize":"12px","padding":"20px 0","textAlign":"center"})]),
            html.Div([
                html.Span(f"{n_charts} gráfico(s)", style={"fontSize":"11px","color":"var(--muted)"}),
                html.Span(types, style={"fontSize":"11px","color":BLUE,"marginLeft":"8px"}),
            ], style={"marginTop":"10px","display":"flex","alignItems":"center"}),
            html.A(
                [html.I(className="fas fa-external-link-alt me-1"), "Abrir no Builder"],
                href=f"/dashboards?sid={sid}",
                style={
                    "display":"block","textAlign":"center","textDecoration":"none",
                    "background":"rgba(37,99,235,0.15)","border":"1px solid rgba(37,99,235,0.4)",
                    "color":"#93c5fd","borderRadius":"8px","marginTop":"10px",
                    "fontSize":"12px","width":"100%","padding":"6px 0",
                },
            ),
        ]), style={
            "background":"var(--card)","border":"1px solid var(--border)",
            "borderRadius":"16px","height":"100%",
            "transition":"box-shadow 0.2s",
        }),
        md=4, style={"marginBottom":"20px"},
    )


def gallery_layout():
    pinned = list_pinned()
    if pinned.empty:
        body = html.Div([
            html.I(className="fas fa-table-columns", style={"fontSize":"48px","color":"var(--border)","marginBottom":"16px"}),
            html.P("Nenhum dashboard salvo ainda.", style={"color":"var(--muted)","fontSize":"15px"}),
            html.P("Gere um dashboard no Builder e clique em ★ Salvar na Galeria.", style={"color":"var(--muted)","fontSize":"13px"}),
            dbc.Button([html.I(className="fas fa-chart-area me-2"), "Ir para o Builder"],
                       href="/dashboards", style={"background":BLUE,"border":"none","borderRadius":"12px","padding":"10px 24px","marginTop":"8px"}),
        ], style={"display":"flex","flexDirection":"column","alignItems":"center","justifyContent":"center","padding":"80px 20px"})
    else:
        cards = []
        for _, row in pinned.iterrows():
            data = load_dashboard(row["session_id"])
            if data:
                cards.append(_gallery_card(row["session_id"], row["title"], data["spec"]))
        body = dbc.Row(cards, className="g-3")

    return html.Div([
        dcc.Store(id="gallery-refresh", data=0),
        html.Div([
            html.Div([
                html.H1("Dashboard", style={"marginBottom": "4px"}),
                html.P("Dashboards salvos na galeria — gerados pelo Builder.", style={"color": "var(--muted)", "marginBottom": "0"}),
            ], style={"flex": "1"}),
            dbc.Button([html.I(className="fas fa-plus me-2"), "Nova Conversa / Dashboard"],
                       href="/dashboards",
                       style={"background": BLUE, "border": "none", "borderRadius": "12px",
                              "padding": "10px 20px", "fontSize": "13px", "fontWeight": "600"}),
        ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "marginBottom": "30px"}),
        html.Div(id="gallery-body", children=body),
    ])


# ─── callback: remover da galeria ─────────────────────────────────────────────
def _register_gallery_callbacks(app):
    from dash import Input, Output, ALL, ctx
    import dash

    @app.callback(
        Output("gallery-body", "children"),
        Input({"type": "gallery-unpin", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def unpin_card(clicks):
        if not ctx.triggered or not any(clicks):
            return dash.no_update
        try:
            sid = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
            unpin_dashboard(sid)
        except Exception:
            pass
        # re-renderiza galeria
        pinned = list_pinned()
        if pinned.empty:
            return html.Div([
                html.I(className="fas fa-table-columns", style={"fontSize":"48px","color":"var(--border)","marginBottom":"16px"}),
                html.P("Nenhum dashboard salvo ainda.", style={"color":"var(--muted)","fontSize":"15px"}),
                dbc.Button([html.I(className="fas fa-chart-area me-2"), "Ir para o Builder"],
                           href="/dashboards", style={"background":BLUE,"border":"none","borderRadius":"12px","padding":"10px 24px","marginTop":"8px"}),
            ], style={"display":"flex","flexDirection":"column","alignItems":"center","padding":"80px 20px"})
        cards = []
        for _, row in pinned.iterrows():
            data = load_dashboard(row["session_id"])
            if data:
                cards.append(_gallery_card(row["session_id"], row["title"], data["spec"]))
        return dbc.Row(cards, className="g-3")