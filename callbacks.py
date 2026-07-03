import json
import time
import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import dash
from dash import Input, Output, State, callback_context, html, ALL
import dash_bootstrap_components as dbc

import requests
from processador import start_processing, stop_processing, get_logs, is_running
from chat import chat, new_session_id, load_history, list_sessions, delete_session

from data import (
    BG, CARD, BLUE, TEXT, MUTED, BORDER, PALETTE,
    API_BASE_URL, API_KEY,
    terms_df, df_patents, EMB, C_matrix, t_map, idx_map,
    monthly_term_count, similar_patents, build_graph,
    calc_growth, calc_density, calc_fusion, calc_shift, calc_future_score,
    ranking_table, semantic_vector,
    term_correlations, build_temporal_matrix, pearson_with_term,
    get_sparse_opportunities,
)

def get_predictions_df(term: str) -> pd.DataFrame:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        r = requests.get(f"{API_BASE_URL}/predictions/{term}", headers=headers, timeout=10)
        if r.status_code == 404:
            return pd.DataFrame()
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data.get("predictions", []))
    except Exception as e:
        logger.error("Failed to get predictions for %s: %s", term, e)
        return pd.DataFrame()

def get_dictionary_df() -> pd.DataFrame:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        r = requests.get(f"{API_BASE_URL}/dictionary", headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data)
    except Exception as e:
        logger.error("Failed to get dictionary: %s", e)
        return pd.DataFrame()

def add_dictionary_term(term: str) -> bool:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        r = requests.post(f"{API_BASE_URL}/dictionary", json={"term": term}, headers=headers, timeout=10)
        r.raise_for_status()
        res = r.json()
        return res.get("status") in ("ok", "duplicate")
    except Exception as e:
        logger.error("Failed to add dictionary term %s: %s", term, e)
        return False

def delete_dictionary_term(term_id: int) -> bool:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        r = requests.delete(f"{API_BASE_URL}/dictionary/{term_id}", headers=headers, timeout=10)
        r.raise_for_status()
        res = r.json()
        return res.get("status") == "ok"
    except Exception as e:
        logger.error("Failed to delete dictionary term %d: %s", term_id, e)
        return False

from gemini import (
    build_term_context, build_indicators_context, build_comparison_context,
    build_correlation_context, build_opportunities_context,
    call_gemini, render_analysis,
    prompt_trend_single, prompt_trend_comparison, prompt_forecast,
    prompt_indicators, prompt_correlation, prompt_opportunities,
)

from scipy.spatial.distance import cosine as scipy_cosine

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def create_dark_table(df, hover=True, striped=False, style_extra=None):
    if df is None or df.empty:
        return html.P("Sem dados para exibir.", style={"color": MUTED, "padding": "20px"})
    classes = ["table", "table-dark", "table-borderless"]
    if hover:   classes.append("table-hover")
    if striped: classes.append("table-striped")
    base = {"color": TEXT, "background": "transparent"}
    if style_extra: base.update(style_extra)
    return dbc.Table.from_dataframe(df, className=" ".join(classes), style=base, responsive=True)


# ─────────────────────────────────────────────
# CACHE DE IA NO SERVIDOR (com TTL)
# ─────────────────────────────────────────────
# O dcc.Store já cacheia por sessão de navegador (instantâneo, mas se perde
# com refresh ou em outra aba). Este cache complementa isso no lado do
# servidor: mesma chave (tipo de análise + termo) evita uma nova chamada
# ao Gemini mesmo entre sessões diferentes, até expirar o TTL.
_AI_CACHE = {}
_AI_CACHE_TTL = 3600  # 1 hora

def _ai_cache_get(key: str):
    entry = _AI_CACHE.get(key)
    if not entry:
        return None
    value, ts = entry
    if time.time() - ts > _AI_CACHE_TTL:
        del _AI_CACHE[key]
        return None
    return value

def _ai_cache_set(key: str, value: str):
    _AI_CACHE[key] = (value, time.time())


def api_waiting_layout(pathname):
    return html.Div([
        # Insere um iframe invisível que faz a chamada à API nativamente pelo navegador, acordando o Render em segundo plano.
        html.Iframe(src=f"{API_BASE_URL}/health", style={"display": "none"}),
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-server fa-spin me-3", style={"fontSize": "40px", "color": BLUE}),
                    html.H2("Inicializando Servidor de Dados (Plano Gratuito)", style={"margin": 0, "color": TEXT, "fontWeight": "600"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px"}),
                html.P(
                    "A API de patentes está sendo inicializada no Render. Como estamos utilizando o plano gratuito, "
                    "o servidor de dados entra em modo de suspensão após inatividade e pode levar de 1 a 2 minutos para acordar.",
                    style={"color": MUTED, "fontSize": "16px", "lineHeight": "1.6"}
                ),
                html.P(
                    "Por favor, aguarde alguns instantes e pressione F5 para carregar a plataforma. Se a página carregar incompleta (apenas com o fundo e a barra lateral em branco), basta atualizar o navegador (F5) mais uma vez para sincronizar os dados.",
                    style={"color": "#4b5563", "fontSize": "14px", "marginBottom": "24px"}
                ),
                dbc.Button([
                    html.I(className="fas fa-sync-alt me-2"), "Verificar Conexão e Recarregar"
                ], href=pathname, external_link=True, color="primary", size="lg",
                   style={"borderRadius": "12px", "fontWeight": "600", "padding": "12px 24px", "background": BLUE, "border": "none"})
            ]),
            style={
                "background": CARD,
                "border": f"1px solid {BORDER}",
                "borderRadius": "24px",
                "padding": "30px",
                "maxWidth": "650px",
                "margin": "80px auto 0",
                "boxShadow": "0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3)"
            }
        )
    ], style={"padding": "20px"})

def register_callbacks(app, page_routes):

    # =========================================================================
    # NAVEGAÇÃO
    # =========================================================================
    @app.callback(Output("page-content", "children"),
                  Input("url", "pathname"), Input("url", "search"))
    def render_page(pathname, search):
        try:
            import data
            
            # Se os dados estão vazios (por exemplo, devido ao cold start da API no Render), tenta recarregar imediatamente
            if data.df_patents.empty or data.terms_df.empty:
                logger.warning("Dados vazios na renderização da página %s. Tentando recarregar da API...", pathname)
                try:
                    data.refresh_data()
                except Exception as ex:
                    logger.error("Falha ao recarregar dados na renderização da página: %s", ex)

            # Se mesmo após a tentativa de refresh os dados ainda estiverem vazios, exibe a tela de carregamento amigável
            if data.df_patents.empty or data.terms_df.empty:
                return api_waiting_layout(pathname)

            layout_fn = page_routes.get(pathname, page_routes["/"])
            # passa search só para o builder (que sabe ler ?sid=)
            if pathname == "/dashboards":
                return layout_fn(search)
            return layout_fn()
        except Exception as e:
            logger.error("Erro ao renderizar página %s: %s", pathname, e, exc_info=True)
            return html.Div(f"Erro: {e}", style={"color": "#e74c3c", "padding": "40px"})

    @app.callback(
        Output("sidebar", "className"), Output("page-content", "className"),
        Output("sidebar-state", "data"),
        Input("toggle-sidebar", "n_clicks"), Input("url", "pathname"),
        State("sidebar-state", "data"),
    )
    def toggle_sidebar(n, path, state):
        trigger = callback_context.triggered_id
        if trigger == "toggle-sidebar": state = "closed" if state == "open" else "open"
        elif trigger == "url":          state = "closed"
        if state == "open": return "sidebar sidebar-open", "content content-open", "open"
        return "sidebar sidebar-closed", "content content-closed", "closed"

    # =========================================================================
    # TENDÊNCIAS — GRÁFICO INDIVIDUAL
    # =========================================================================
    @app.callback(
        Output("tendencia-graph", "figure"),
        Input("master-tendencia-dropdown", "value"), Input("tend-chart-type", "value"),
        prevent_initial_call=True,
    )
    def update_tend(term, chart_type):
        try:
            if not term: return go.Figure()
            hist = monthly_term_count(term, terms_df)
            if hist.empty: return go.Figure()
            if chart_type == "bar":
                fig = px.bar(hist, x="year_month", y="count", title=f"Frequência: {term}", template="plotly_dark")
                fig.update_traces(marker_color=BLUE)
            else:
                fig = px.line(hist, x="year_month", y="count", markers=True, title=f"Frequência: {term}", template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT)
            return fig
        except Exception as e:
            logger.error("update_tend falhou (term=%r): %s", term, e, exc_info=True)
            return go.Figure()

    # =========================================================================
    # TENDÊNCIAS — COMPARAÇÃO
    # =========================================================================
    @app.callback(
        Output("tendencia-comp-graph", "figure"),
        Input("tendencia-comp-1", "value"), Input("tendencia-comp-2", "value"),
        Input("tendencia-comp-3", "value"), Input("comp-chart-type", "value"),
        prevent_initial_call=True,
    )
    def update_comp(t1, t2, t3, chart_type):
        try:
            terms = [t for t in [t1, t2, t3] if t]
            if not terms:
                return go.Figure().update_layout(template="plotly_dark", title="Selecione pelo menos um termo",
                                                 font_color=TEXT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            fig = go.Figure()
            for i, t in enumerate(terms):
                hist = monthly_term_count(t, terms_df)
                if hist.empty: continue
                col = PALETTE[i % len(PALETTE)]
                if chart_type == "bar":
                    fig.add_trace(go.Bar(x=hist["year_month"], y=hist["count"], name=t, marker_color=col))
                else:
                    fig.add_trace(go.Scatter(x=hist["year_month"], y=hist["count"], name=t,
                                             mode="lines+markers", line=dict(color=col, width=2)))
            if chart_type == "bar": fig.update_layout(barmode="group")
            fig.update_layout(template="plotly_dark", title="Comparação de Tendências",
                              xaxis_title="Mês", yaxis_title="Ocorrências", hovermode="x unified",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            return fig
        except Exception as e:
            logger.error("update_comp falhou (t1=%r t2=%r t3=%r): %s", t1, t2, t3, e, exc_info=True)
            return go.Figure()

    # =========================================================================
    # TENDÊNCIAS — PREDIÇÃO TFT
    # =========================================================================
    @app.callback(
        Output("pred-graph", "figure"),
        Output("pred-last-real", "children"), Output("pred-next", "children"),
        Output("pred-next", "style"),         Output("pred-delta", "children"),
        Output("pred-delta", "style"),        Output("pred-pess", "children"),
        Output("pred-opt", "children"),
        Input("master-tendencia-dropdown", "value"), prevent_initial_call=True,
    )
    def update_pred(term):
        try:
            if not term:
                fig = go.Figure()
                fig.add_annotation(text="Selecione um termo", xref="paper", yref="paper",
                                   x=0.5, y=0.5, showarrow=False, font=dict(color=MUTED, size=14))
                fig.update_layout(template="plotly_dark", height=450,
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT)
                return fig, "—", "—", {"color": BLUE}, "—", {"color": MUTED}, "—", "—"

            hist = monthly_term_count(term, terms_df)
            if not hist.empty:
                hist["year_month"] = pd.to_datetime(hist["year_month"], errors="coerce")
            pred = get_predictions_df(term)
            if not pred.empty and "target_year_month" in pred.columns:
                pred["target_year_month"] = pd.to_datetime(pred["target_year_month"], errors="coerce")


            fig = go.Figure()
            if not hist.empty:
                fig.add_trace(go.Scatter(x=hist["year_month"], y=hist["count"], name="Real",
                                         line=dict(color="#3498db", width=3), mode="lines+markers"))
            if not pred.empty and "optimistic_count" in pred.columns and "pessimistic_count" in pred.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([pred["target_year_month"], pred["target_year_month"].iloc[::-1]]),
                    y=pd.concat([pred["optimistic_count"], pred["pessimistic_count"].iloc[::-1]]),
                    fill="toself", fillcolor="rgba(46,204,113,0.15)",
                    line_color="rgba(255,255,255,0)", name="Incerteza (q10–q90)", hoverinfo="skip"))
            if not pred.empty and "predicted_count" in pred.columns:
                fig.add_trace(go.Scatter(x=pred["target_year_month"], y=pred["predicted_count"],
                                         name="Previsão", line=dict(color="#2ecc71", dash="dash"),
                                         mode="lines+markers"))
            if not any(t.name == "Previsão" for t in fig.data):
                fig.add_annotation(text="Sem dados de previsão.<br>Execute patent_tft_pipeline.py.",
                                   xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                                   font=dict(color=MUTED, size=13), align="center",
                                   bordercolor=BORDER, borderwidth=1, borderpad=10, bgcolor=CARD)
            fig.update_layout(template="plotly_dark", height=450,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT,
                              hovermode="x unified", xaxis_title="Mês", yaxis_title="Ocorrências",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

            last = hist["count"].iloc[-1] if not hist.empty else 0
            next_val = None
            if not pred.empty and "predicted_count" in pred.columns and len(pred) > 0:
                if not hist.empty:
                    fp = pred[pred["target_year_month"] > hist["year_month"].max()]
                    if not fp.empty: next_val = fp.iloc[0]["predicted_count"]
                else:
                    next_val = pred.iloc[0]["predicted_count"]

            last_str = f"{last:.0f} pat." if pd.notna(last) else "0"
            next_str = f"{next_val:.1f}" if pd.notna(next_val) else "—"
            if pd.notna(last) and last > 0 and pd.notna(next_val):
                delta = f"{((next_val-last)/last*100):+.1f}%"
                delta_color = "#22c55e" if next_val > last else "#e74c3c" if next_val < last else TEXT
            else:
                delta, delta_color = "—", MUTED
            pess = opt = "—"
            if not pred.empty and "pessimistic_count" in pred.columns:
                v = pred["pessimistic_count"].iloc[0]
                pess = f"{v:.1f}" if pd.notna(v) else "—"
            if not pred.empty and "optimistic_count" in pred.columns:
                v = pred["optimistic_count"].iloc[-1]
                opt = f"{v:.1f}" if pd.notna(v) else "—"
            return fig, last_str, next_str, {"color": BLUE}, delta, {"color": delta_color}, pess, opt

        except Exception as e:
            logger.error("update_pred falhou (term=%r): %s", term, e, exc_info=True)
            fig = go.Figure()
            fig.add_annotation(text=f"Erro: {e}", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font=dict(color="#e74c3c", size=12))
            fig.update_layout(template="plotly_dark", height=450,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT)
            return fig, "—", "—", {"color": MUTED}, "—", {"color": MUTED}, "—", "—"

    # =========================================================================
    # REDE
    # =========================================================================
    @app.callback(Output("network-graph", "figure"),
                  Input("termo-rede-dropdown", "value"), Input("rede-depth-slider", "value"))
    def update_net(term, depth):
        try:
            if not term or terms_df.empty: return go.Figure()
            G = build_graph(term, terms_df, depth=int(depth or 3), top_n=5)
            if not G.nodes(): return go.Figure()
            pos = nx.spring_layout(G, k=0.8, seed=42, iterations=50)
            edges = [go.Scatter(x=[pos[u][0],pos[v][0],None], y=[pos[u][1],pos[v][1],None],
                                line=dict(width=2,color="rgba(136,136,136,0.4)"),
                                hoverinfo="none", mode="lines") for u,v in G.edges()]
            nl = list(G.nodes()); pal = ["#FF4B4B","#FFA500","#1E90FF","#00FF7F","#808080"]
            nodes = go.Scatter(
                x=[pos[n][0] for n in nl], y=[pos[n][1] for n in nl],
                mode="markers+text", text=nl, textposition="top center",
                marker=dict(color=[pal[min(G.nodes[n].get("layer",0),4)] for n in nl],
                            size=[15+G.degree(n)*3 for n in nl]),
                hovertext=[f"{n} ({G.degree(n)} conexões)" for n in nl], hoverinfo="text")
            fig = go.Figure(data=edges+[nodes])
            fig.update_layout(template="plotly_dark", showlegend=False, height=600,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT,
                              xaxis=dict(visible=False), yaxis=dict(visible=False),
                              title=f"Rede: {term} | Camadas: {depth}")
            return fig
        except Exception as e:
            logger.error("update_net falhou (term=%r depth=%r): %s", term, depth, e, exc_info=True)
            return go.Figure()

    # =========================================================================
    # INDICADORES
    # =========================================================================
    @app.callback(
        Output("ind-growth","children"), Output("ind-growth","style"),
        Output("ind-density","children"), Output("ind-fusion","children"),
        Output("ind-shift","children"),   Output("ind-future","children"),
        Input("master-indicadores-dropdown","value"), prevent_initial_call=True,
    )
    def update_ind(term):
        try:
            if not term: return "0.00%", {"color": MUTED}, "0", "0", "0.00%", "0.00"
            g = calc_growth(term, terms_df)
            return (f"{g:.2f}%", {"color": "#22c55e" if g>0 else "#e74c3c" if g<0 else TEXT},
                    f"{calc_density(term,terms_df)}", f"{calc_fusion(term,terms_df)}",
                    f"{calc_shift(term,terms_df):.2f}%", f"{calc_future_score(term,terms_df):.2f}")
        except Exception as e:
            logger.error("update_ind falhou (term=%r): %s", term, e, exc_info=True)
            return "—", {"color": "#e74c3c"}, "—", "—", "—", "—"

    @app.callback(Output("evol-sim-val","children"), Output("evol-shift-val","children"),
                  Input("master-indicadores-dropdown","value"), prevent_initial_call=True)
    def update_evol(term):
        try:
            if not term: return "0.0000", "0.00%"
            months = sorted(terms_df["year_month"].dropna().unique().tolist())
            if len(months) < 2: return "0.0000", "0.00%"
            v1 = semantic_vector(term, months[0],  terms_df)
            v2 = semantic_vector(term, months[-1], terms_df)
            sim = 1 - scipy_cosine(v1, v2) if (v1.sum()>0 and v2.sum()>0) else 0
            return f"{sim:.4f}", f"{(1-sim)*100:.2f}%"
        except Exception as e:
            logger.error("update_evol falhou (term=%r): %s", term, e, exc_info=True)
            return "—", "—"

    @app.callback(Output("indicadores-bar-chart","figure"),
                  Input("master-indicadores-dropdown","value"), Input("ind-chart-type","value"))
    def update_ind_bar(selected_term, chart_type):
        try:
            rk = ranking_table(terms_df)
            if rk.empty:
                return go.Figure().update_layout(template="plotly_dark", title="Sem dados",
                                                 font_color=TEXT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            top15 = rk.head(15)
            colors = [BLUE if t == selected_term else MUTED for t in top15["term"]]
            if chart_type == "bar":
                fig = go.Figure(go.Bar(x=top15["term"], y=top15["future_score"], marker_color=colors))
            else:
                fig = go.Figure(go.Scatter(x=top15["term"], y=top15["future_score"],
                                           mode="lines+markers", line=dict(color=BLUE, width=2),
                                           marker=dict(color=colors, size=9)))
            fig.update_layout(template="plotly_dark", title="Top 15 por Future Score",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT,
                              xaxis_title="Termo", yaxis_title="Future Score", xaxis_tickangle=-45)
            return fig
        except Exception as e:
            logger.error("update_ind_bar falhou: %s", e, exc_info=True)
            return go.Figure()

    # =========================================================================
    # SIMILARIDADE
    # =========================================================================
    @app.callback(Output("similarity-results","children"),
                  Input("patent-dropdown","value"), prevent_initial_call=True)
    def update_sim(idx):
        try:
            if idx is None: return html.P("Selecione uma patente.", style={"color": MUTED})
            res = similar_patents(int(idx), df_patents, EMB)
            return (create_dark_table(res[["id","title","year_month","similarity"]].head(10).round(4))
                    if not res.empty else html.P("Sem similares.", style={"color": MUTED}))
        except Exception as e:
            logger.error("update_sim falhou (idx=%r): %s", idx, e, exc_info=True)
            return html.P("Erro.", style={"color": "#e74c3c"})

    # =========================================================================
    # CORRELAÇÃO
    # =========================================================================
    @app.callback(Output("correlacao-table","children"),
                  Input("termo-correlacao-dropdown","value"), prevent_initial_call=True)
    def update_corr(term):
        try:
            res = term_correlations(term, terms_df) if term else None
            return (create_dark_table(res.head(20)) if res is not None and not res.empty
                    else html.P("Sem correlações.", style={"color": MUTED}))
        except Exception as e:
            logger.error("update_corr falhou (term=%r): %s", term, e, exc_info=True)
            return html.P("Erro.", style={"color": "#e74c3c"})

    @app.callback(
        Output("pearson-bar-chart","figure"),
        Output("pearson-partner-dropdown","options"), Output("pearson-partner-dropdown","value"),
        Input("termo-correlacao-dropdown","value"), Input("corr-topn-slider","value"),
        Input("corr-threshold-slider","value"),     Input("corr-chart-type","value"),
        prevent_initial_call=True,
    )
    def update_pearson_ranking(selected_term, top_n, threshold, chart_type):
        empty = go.Figure().update_layout(template="plotly_dark", font_color=TEXT,
                                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        try:
            if not selected_term or terms_df.empty:
                return empty.update_layout(title="Selecione um termo"), [], None
            freq = terms_df.groupby("term")["patent_id"].nunique()
            dff  = terms_df[terms_df["term"].isin(freq[freq >= 2].index)].copy()
            if dff.empty: return empty, [], None
            pivot      = build_temporal_matrix(dff)
            pearson_df = pearson_with_term(pivot, selected_term)
            if pearson_df.empty:
                return empty.update_layout(title=f"Sem dados para '{selected_term}'"), [], None
            df_filt = pearson_df[pearson_df["pearson_r"].abs() >= threshold].head(top_n)
            partner_opts = [{"label": t, "value": t} for t in pearson_df["parceiro"]]
            default_p    = pearson_df.iloc[0]["parceiro"] if not pearson_df.empty else None
            if df_filt.empty:
                return empty.update_layout(title="Nenhum acima do filtro", height=300), partner_opts, default_p
            df_plot = df_filt.sort_values("pearson_r")
            colors  = ["#e74c3c" if r < 0 else "#2ecc71" for r in df_plot["pearson_r"]]
            if chart_type == "bar":
                fig = go.Figure(go.Bar(
                    x=df_plot["pearson_r"], y=df_plot["parceiro"], orientation="h",
                    marker_color=colors, text=df_plot["pearson_r"].apply(lambda v: f"{v:+.3f}"),
                    textposition="outside", customdata=df_plot["p_value"],
                    hovertemplate="<b>%{y}</b><br>r=%{x:.4f}<br>p=%{customdata:.4f}<extra></extra>"))
            else:
                fig = go.Figure(go.Scatter(
                    x=df_plot["pearson_r"], y=df_plot["parceiro"], mode="markers+text",
                    text=df_plot["pearson_r"].apply(lambda v: f"{v:+.3f}"),
                    textposition="middle right", marker=dict(color=colors, size=10)))
            fig.update_layout(template="plotly_dark", title=f"Pearson com '{selected_term}'",
                              xaxis=dict(title="Pearson r", range=[-1.25,1.25], zeroline=True,
                                         zerolinecolor="rgba(255,255,255,0.3)"),
                              height=max(350, len(df_plot)*32),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color=TEXT, margin=dict(l=60,r=20,t=40,b=20))
            return fig, partner_opts, default_p
        except Exception as e:
            logger.error("update_pearson_ranking falhou (term=%r): %s", selected_term, e, exc_info=True)
            return empty.update_layout(title=f"Erro: {e}"), [], None

    @app.callback(
        Output("pearson-line-chart","figure"),
        Output("pearson-r-metric","children"), Output("pearson-r-metric","style"),
        Output("pearson-p-metric","children"), Output("pearson-p-metric","style"),
        Input("termo-correlacao-dropdown","value"), Input("pearson-partner-dropdown","value"),
        Input("pearson-line-chart-type","value"), prevent_initial_call=True,
    )
    def update_pearson_comparison(selected_term, partner_term, chart_type):
        empty = (go.Figure().update_layout(template="plotly_dark", title="Selecione dois termos",
                                           font_color=TEXT, paper_bgcolor="rgba(0,0,0,0)",
                                           plot_bgcolor="rgba(0,0,0,0)", height=400),
                 "—", {"color": MUTED}, "—", {"color": MUTED})
        try:
            if not selected_term or not partner_term or terms_df.empty: return empty
            freq = terms_df.groupby("term")["patent_id"].nunique()
            dff  = terms_df[terms_df["term"].isin(freq[freq >= 2].index)].copy()
            if dff.empty or selected_term not in dff["term"].values or partner_term not in dff["term"].values:
                return empty
            pivot      = build_temporal_matrix(dff)
            pearson_df = pearson_with_term(pivot, selected_term)
            if pearson_df.empty or partner_term not in pearson_df["parceiro"].values: return empty
            row = pearson_df[pearson_df["parceiro"] == partner_term].iloc[0]
            r_val, p_val = row["pearson_r"], row["p_value"]
            r_color = "#2ecc71" if r_val > 0.7 else "#f39c12" if r_val > 0.4 else "#e74c3c" if r_val < 0 else TEXT
            p_color = "#22c55e" if p_val < 0.05 else MUTED
            if chart_type == "bar":
                fig = go.Figure()
                fig.add_trace(go.Bar(x=pivot.index, y=pivot[selected_term], name=selected_term, marker_color=PALETTE[0]))
                fig.add_trace(go.Bar(x=pivot.index, y=pivot[partner_term],  name=partner_term,  marker_color=PALETTE[1]))
                fig.update_layout(barmode="group")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pivot.index, y=pivot[selected_term], name=selected_term,
                                         mode="lines+markers", line=dict(color=PALETTE[0], width=2), marker=dict(size=6)))
                fig.add_trace(go.Scatter(x=pivot.index, y=pivot[partner_term],  name=partner_term,
                                         mode="lines+markers", line=dict(color=PALETTE[1], width=2, dash="dot"),
                                         marker=dict(size=6, symbol="diamond")))
            fig.update_layout(template="plotly_dark", title=f"'{selected_term}' vs '{partner_term}'",
                              xaxis_title="Mês", yaxis_title="Ocorrências", hovermode="x unified", height=400,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT)
            return fig, f"{r_val:+.4f}", {"color": r_color}, f"{p_val:.2e}", {"color": p_color}
        except Exception as e:
            logger.error("update_pearson_comparison falhou (term=%r partner=%r): %s",
                         selected_term, partner_term, e, exc_info=True)
            return (go.Figure().update_layout(template="plotly_dark", title=f"Erro: {e}",
                                              font_color=TEXT, paper_bgcolor="rgba(0,0,0,0)",
                                              plot_bgcolor="rgba(0,0,0,0)", height=400),
                    "—", {"color": "#e74c3c"}, "—", {"color": "#e74c3c"})

    @app.callback(Output("pearson-full-table","children"),
                  Input("termo-correlacao-dropdown","value"), prevent_initial_call=True)
    def update_pearson_table(selected_term):
        try:
            if not selected_term or terms_df.empty: return html.P("Selecione um termo.", style={"color": MUTED})
            freq = terms_df.groupby("term")["patent_id"].nunique()
            dff  = terms_df[terms_df["term"].isin(freq[freq >= 2].index)].copy()
            if dff.empty: return html.P("Sem dados.", style={"color": MUTED})
            pivot      = build_temporal_matrix(dff)
            pearson_df = pearson_with_term(pivot, selected_term)
            if pearson_df.empty: return html.P("Sem correlações.", style={"color": MUTED})
            display = pearson_df.copy(); display.columns = ["Termo","Pearson r","p-value"]
            display["Pearson r"] = display["Pearson r"].round(4)
            display["p-value"]   = display["p-value"].round(4)
            display["Significativo"] = display["p-value"].apply(lambda p: "✅" if p < 0.05 else "—")
            return create_dark_table(display.reset_index(drop=True), style_extra={"fontSize": "13px"})
        except Exception as e:
            logger.error("update_pearson_table falhou (term=%r): %s", selected_term, e, exc_info=True)
            return html.P(f"Erro: {e}", style={"color": "#e74c3c"})

    # =========================================================================
    # RANKING
    # =========================================================================
    @app.callback(Output("ranking-table","children"), Input("ranking-table","id"))
    def update_rank(_):
        try:
            rk = ranking_table(terms_df)
            return create_dark_table(rk.head(30)) if not rk.empty else html.P("Sem dados.", style={"color": MUTED})
        except Exception as e:
            logger.error("update_rank falhou: %s", e, exc_info=True)
            return html.P("Erro.", style={"color": "#e74c3c"})

    # =========================================================================
    # OPORTUNIDADES
    # =========================================================================
    @app.callback(Output("sparse-table","children"), Output("sparse-graph","figure"),
                  Input("termo-oportunidades-dropdown","value"), Input("opp-chart-type","value"),
                  prevent_initial_call=True)
    def update_opp(term, chart_type):
        try:
            res = get_sparse_opportunities(term, C_matrix, t_map, idx_map) if term else None
            if res is None or res.empty:
                return html.P("Sem oportunidades.", style={"color": MUTED}), go.Figure()
            top = res.head(15)
            if chart_type == "bar":
                fig = px.bar(top, x="bridge_strength", y="term", orientation="h",
                             color="bridge_strength", template="plotly_dark", title="Bridge Strength")
            else:
                fig = go.Figure(go.Scatter(x=top["bridge_strength"], y=top["term"],
                                           mode="markers+lines", line=dict(color=BLUE),
                                           marker=dict(color=BLUE, size=9)))
                fig.update_layout(template="plotly_dark", title="Bridge Strength")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color=TEXT, xaxis_title="Força")
            return create_dark_table(res.round(3)), fig
        except Exception as e:
            logger.error("update_opp falhou (term=%r): %s", term, e, exc_info=True)
            return html.P("Erro.", style={"color": "#e74c3c"}), go.Figure()

    # =========================================================================
    # DICIONÁRIO
    # =========================================================================
    @app.callback(
        Output("dict-table","children"), Output("delete-term-dropdown","options"),
        Input("refresh-dict-btn","n_clicks"), Input("add-term-btn","n_clicks"),
        Input("delete-term-btn","n_clicks"), prevent_initial_call=False,
    )
    def update_dict_view(n_r, n_a, n_d):
        try:
            dict_df = get_dictionary_df()
            if dict_df.empty: return html.P("Dicionário vazio.", style={"color": MUTED}), []
            opts    = [{"label": str(r["term"]), "value": r["id"]} for _, r in dict_df.iterrows()]
            display = dict_df[["id","term"]].copy(); display.columns = ["ID","Termo"]
            return create_dark_table(display, hover=True), opts
        except Exception as e:
            logger.error("update_dict_view falhou: %s", e, exc_info=True)
            return html.P("Erro.", style={"color": "#e74c3c"}), []

    @app.callback(Output("add-term-msg","children"),
                  Input("add-term-btn","n_clicks"), State("new-term-input","value"),
                  prevent_initial_call=True)
    def handle_add_term(n, term):
        if not term or not term.strip(): return html.P("Digite um termo válido.", style={"color": "#f39c12"})
        ok = add_dictionary_term(term.strip())
        return html.P("✅ Adicionado!" if ok else "❌ Erro.", style={"color": "#22c55e" if ok else "#e74c3c"})

    @app.callback(Output("delete-term-msg","children"),
                  Input("delete-term-btn","n_clicks"), State("delete-term-dropdown","value"),
                  prevent_initial_call=True)
    def handle_delete_term(n, term_id):
        if not term_id: return html.P("Selecione um termo.", style={"color": "#f39c12"})
        ok = delete_dictionary_term(term_id)
        return html.P("✅ Removido!" if ok else "❌ Erro.", style={"color": "#22c55e" if ok else "#e74c3c"})


    # =========================================================================
    # PIPELINE
    # =========================================================================
    @app.callback(Output("process-log","children"),
                  Input("start-process-btn","n_clicks"), Input("stop-process-btn","n_clicks"),
                  prevent_initial_call=True)
    def handle_processing(start, stop):
        if not callback_context.triggered: return dash.no_update
        if callback_context.triggered_id == "start-process-btn": return [html.Div(start_processing(), style={"color": "#22c55e"})]
        return [html.Div(stop_processing(), style={"color": "#f39c12"})]

    @app.callback(
        Output("process-log","children",allow_duplicate=True),
        Output("pipeline-status","children"), Output("pipeline-status","style"),
        Input("process-log-interval","n_intervals"), prevent_initial_call=True,
    )
    def update_logs(_):
        running = is_running()
        return ([html.Div(l, style={"marginBottom": "8px", "color": "#e2e8f0"}) for l in get_logs()],
                "● Processando..." if running else "● Sistema pronto",
                {"color": "#22c55e" if running else "#9ca3af", "fontSize": "15px", "marginTop": "20px"})

    # =========================================================================
    # GEMINI — 6 CALLBACKS
    # =========================================================================
    # Cada callback segue o mesmo padrão de 3 camadas de cache antes de
    # chamar o Gemini de fato:
    #   1. dcc.Store (cache da sessão do navegador — instantâneo)
    #   2. _AI_CACHE no servidor (sobrevive a refresh / outras sessões, TTL)
    #   3. chamada real ao Gemini (só se as duas anteriores não tiverem)

    @app.callback(Output("ai-out-tend","children"), Output("ai-cache-tend","data"),
                  Input("btn-ai-tend","n_clicks"), State("master-tendencia-dropdown","value"),
                  State("ai-cache-tend","data"), prevent_initial_call=True)
    def ai_tend(n, term, cache):
        if not n or not term: return "", cache
        if term in cache: return render_analysis(cache[term], "Trend Analysis"), cache
        cache_key = f"tend:{term}"
        cached = _ai_cache_get(cache_key)
        if cached is not None:
            cache[term] = cached
            return render_analysis(cached, "Trend Analysis"), cache
        hist = monthly_term_count(term, terms_df)
        text = call_gemini(prompt_trend_single(build_term_context(term, hist)))
        cache[term] = text
        _ai_cache_set(cache_key, text)
        return render_analysis(text, "Trend Analysis"), cache

    @app.callback(Output("ai-out-comp","children"), Output("ai-cache-comp","data"),
                  Input("btn-ai-comp","n_clicks"), State("tendencia-comp-1","value"),
                  State("tendencia-comp-2","value"), State("tendencia-comp-3","value"),
                  State("ai-cache-comp","data"), prevent_initial_call=True)
    def ai_comp(n, t1, t2, t3, cache):
        if not n: return "", cache
        terms = [t for t in [t1, t2, t3] if t]
        if not terms: return html.P("Selecione pelo menos um termo.", style={"color": MUTED}), cache
        key = "|".join(sorted(terms))
        if key in cache: return render_analysis(cache[key], "Comparative Analysis"), cache
        cache_key = f"comp:{key}"
        cached = _ai_cache_get(cache_key)
        if cached is not None:
            cache[key] = cached
            return render_analysis(cached, "Comparative Analysis"), cache
        text = call_gemini(prompt_trend_comparison(build_comparison_context(terms, terms_df)))
        cache[key] = text
        _ai_cache_set(cache_key, text)
        return render_analysis(text, "Comparative Analysis"), cache

    @app.callback(Output("ai-out-pred","children"), Output("ai-cache-pred","data"),
                  Input("btn-ai-pred","n_clicks"), State("master-tendencia-dropdown","value"),
                  State("ai-cache-pred","data"), prevent_initial_call=True)
    def ai_pred(n, term, cache):
        if not n or not term: return "", cache
        if term in cache: return render_analysis(cache[term], "Forecast Intelligence"), cache
        cache_key = f"pred:{term}"
        cached = _ai_cache_get(cache_key)
        if cached is not None:
            cache[term] = cached
            return render_analysis(cached, "Forecast Intelligence"), cache
        hist = monthly_term_count(term, terms_df)
        if not hist.empty:
            hist = hist.copy()
        pred = get_predictions_df(term)
        if not pred.empty and "target_year_month" in pred.columns:
            pred["target_year_month"] = pd.to_datetime(pred["target_year_month"], errors="coerce")

        ctx  = build_term_context(term, hist, pred if not pred.empty else None)
        text = call_gemini(prompt_forecast(ctx))
        cache[term] = text
        _ai_cache_set(cache_key, text)
        return render_analysis(text, "Forecast Intelligence"), cache

    @app.callback(Output("ai-out-ind","children"), Output("ai-cache-ind","data"),
                  Input("btn-ai-ind","n_clicks"), State("master-indicadores-dropdown","value"),
                  State("ai-cache-ind","data"), prevent_initial_call=True)
    def ai_ind(n, term, cache):
        if not n or not term: return "", cache
        if term in cache: return render_analysis(cache[term], "Indicator Analysis"), cache
        cache_key = f"ind:{term}"
        cached = _ai_cache_get(cache_key)
        if cached is not None:
            cache[term] = cached
            return render_analysis(cached, "Indicator Analysis"), cache
        text = call_gemini(prompt_indicators(build_indicators_context(term, terms_df)))
        cache[term] = text
        _ai_cache_set(cache_key, text)
        return render_analysis(text, "Indicator Analysis"), cache

    @app.callback(Output("ai-out-corr","children"), Output("ai-cache-corr","data"),
                  Input("btn-ai-corr","n_clicks"), State("termo-correlacao-dropdown","value"),
                  State("ai-cache-corr","data"), prevent_initial_call=True)
    def ai_corr(n, term, cache):
        if not n or not term: return "", cache
        if term in cache: return render_analysis(cache[term], "Correlation Intelligence"), cache
        cache_key = f"corr:{term}"
        cached = _ai_cache_get(cache_key)
        if cached is not None:
            cache[term] = cached
            return render_analysis(cached, "Correlation Intelligence"), cache
        corr_df = term_correlations(term, terms_df)
        freq    = terms_df.groupby("term")["patent_id"].nunique()
        dff     = terms_df[terms_df["term"].isin(freq[freq >= 2].index)].copy()
        pivot   = build_temporal_matrix(dff) if not dff.empty else None
        p_df    = pearson_with_term(pivot, term) if pivot is not None else pd.DataFrame()
        text    = call_gemini(prompt_correlation(build_correlation_context(term, corr_df, p_df)))
        cache[term] = text
        _ai_cache_set(cache_key, text)
        return render_analysis(text, "Correlation Intelligence"), cache

    @app.callback(Output("ai-out-opp","children"), Output("ai-cache-opp","data"),
                  Input("btn-ai-opp","n_clicks"), State("termo-oportunidades-dropdown","value"),
                  State("ai-cache-opp","data"), prevent_initial_call=True)
    def ai_opp(n, term, cache):
        if not n or not term: return "", cache
        if term in cache: return render_analysis(cache[term], "Opportunity Intelligence"), cache
        cache_key = f"opp:{term}"
        cached = _ai_cache_get(cache_key)
        if cached is not None:
            cache[term] = cached
            return render_analysis(cached, "Opportunity Intelligence"), cache
        sparse_df = get_sparse_opportunities(term, C_matrix, t_map, idx_map)
        text      = call_gemini(prompt_opportunities(build_opportunities_context(term, sparse_df)))
        cache[term] = text
        _ai_cache_set(cache_key, text)
        return render_analysis(text, "Opportunity Intelligence"), cache

    # =========================================================================
    # CHAT
    # =========================================================================

    def _render_message(role: str, content: str):
        is_user = role == "user"
        return html.Div(style={
            "display": "flex",
            "flexDirection": "row-reverse" if is_user else "row",
            "gap": "10px", "alignItems": "flex-start",
        }, children=[
            html.Div(style={
                "width": "32px", "height": "32px", "borderRadius": "50%",
                "background": "rgba(37,99,235,0.85)" if is_user else "rgba(37,99,235,0.2)",
                "flexShrink": "0",
                "display": "flex", "alignItems": "center", "justifyContent": "center",
            }, children=[
                html.I(className="fas fa-user" if is_user else "fas fa-robot",
                       style={"color": "white" if is_user else BLUE, "fontSize": "13px"}),
            ]),
            html.Div(style={
                "background": "rgba(37,99,235,0.85)" if is_user else "rgba(37,99,235,0.08)",
                "border": "none" if is_user else "1px solid rgba(37,99,235,0.2)",
                "borderRadius": ("16px 4px 16px 16px" if is_user else "4px 16px 16px 16px"),
                "padding": "12px 16px", "maxWidth": "75%",
                "fontSize": "13.5px", "lineHeight": "1.7",
                "color": "white" if is_user else "#d1d5db",
                "whiteSpace": "pre-wrap",
            }, children=content),
        ])

    def _render_history(history: list) -> list:
        if not history:
            return []
        return [_render_message(m["role"], m["content"]) for m in history]

    def _build_session_list(active_session_id):
        sessions = list_sessions()
        items = []
        if not sessions.empty:
            for _, row in sessions.iterrows():
                sid       = row["session_id"]
                label     = str(row.get("preview", ""))[:40] or "Conversa"
                count     = row.get("messages", 0)
                is_active = sid == active_session_id
                items.append(html.Div(
                    [
                        html.Div(label, style={"fontSize": "13px", "color": "white" if is_active else "#9ca3af"}),
                        html.Div(f"{count} msgs", style={"fontSize": "11px", "color": MUTED}),
                    ],
                    id={"type": "chat-session-item", "index": sid},
                    className=f"chat-session-item {'active' if is_active else ''}".strip(),
                    n_clicks=0,
                ))
        return items

    # Toggle sidebar (clientside)
    app.clientside_callback(
        """
        function(n, is_open) {
            if (!n) return window.dash_clientside.no_update;
            var sidebar = document.getElementById('chat-sidebar');
            var icon    = document.getElementById('chat-sidebar-icon');
            var label   = document.getElementById('chat-sidebar-label');
            var newLabel= document.getElementById('chat-new-label');
            var list    = document.getElementById('chat-session-list');
            if (!sidebar) return window.dash_clientside.no_update;
            if (is_open) {
                sidebar.style.width   = '56px';
                sidebar.style.padding = '12px 8px';
                sidebar.style.alignItems = 'center';
                if (icon)     icon.className = 'fas fa-chevron-right';
                if (label)    label.style.display = 'none';
                if (newLabel) newLabel.style.display = 'none';
                if (list)     list.style.display = 'none';
                return false;
            } else {
                sidebar.style.width   = '260px';
                sidebar.style.padding = '16px';
                sidebar.style.alignItems = '';
                if (icon)     icon.className = 'fas fa-chevron-left';
                if (label)    label.style.display = '';
                if (newLabel) newLabel.style.display = '';
                if (list)     list.style.display = '';
                return true;
            }
        }
        """,
        Output("chat-sidebar-open", "data"),
        Input("chat-sidebar-toggle", "n_clicks"),
        State("chat-sidebar-open", "data"),
        prevent_initial_call=True,
    )

    # ── init_chat SEM prevent_initial_call ────────────────────────────────
    @app.callback(
        Output("chat-session-id", "data"),
        Output("chat-session-list", "children"),
        Input("url", "pathname"),
        State("chat-session-id", "data"),
    )
    def init_chat(pathname, current_session):
        if pathname != "/chat":
            return current_session or dash.no_update, []
        session_id = current_session or new_session_id()
        logger.debug("init_chat | pathname=%s session=%s", pathname, session_id)
        return session_id, _build_session_list(session_id)

    # Nova conversa
    @app.callback(
        Output("chat-session-id", "data", allow_duplicate=True),
        Output("chat-messages", "children", allow_duplicate=True),
        Output("chat-session-list", "children", allow_duplicate=True),
        Input("chat-new-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def new_chat(n):
        if not n:
            return dash.no_update, dash.no_update, dash.no_update
        session_id = new_session_id()
        welcome = [_render_message("assistant",
            "Nova conversa iniciada! Como posso ajudar com sua análise de patentes?")]
        return session_id, welcome, _build_session_list(session_id)

    # Apagar conversa
    @app.callback(
        Output("chat-session-id", "data", allow_duplicate=True),
        Output("chat-messages", "children", allow_duplicate=True),
        Output("chat-session-list", "children", allow_duplicate=True),
        Input("chat-delete-btn", "n_clicks"),
        State("chat-session-id", "data"),
        prevent_initial_call=True,
    )
    def delete_chat(n, session_id):
        if not n or not session_id:
            return dash.no_update, dash.no_update, dash.no_update
        delete_session(session_id)
        new_sid = new_session_id()
        welcome = [_render_message("assistant", "Conversa apagada. Começando nova sessão!")]
        return new_sid, welcome, _build_session_list(new_sid)

    # Carregar sessão existente
    @app.callback(
        Output("chat-session-id", "data", allow_duplicate=True),
        Output("chat-messages", "children", allow_duplicate=True),
        Input({"type": "chat-session-item", "index": ALL}, "n_clicks"),
        State({"type": "chat-session-item", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def load_session(n_clicks_list, id_list):
        if not callback_context.triggered or not any(n_clicks_list):
            return dash.no_update, dash.no_update
        try:
            clicked_id = json.loads(callback_context.triggered[0]["prop_id"].split(".")[0])["index"]
        except Exception as e:
            logger.error("load_session: falha ao parsear id clicado: %s", e, exc_info=True)
            return dash.no_update, dash.no_update
        history  = load_history(clicked_id)
        messages = _render_history(history) if history else [
            _render_message("assistant", "Conversa vazia.")
        ]
        return clicked_id, messages

    # ── Clique nos exemplos preenche o input ──────────────────────────────
    @app.callback(
        Output("chat-input", "value", allow_duplicate=True),
        Input({"type": "chat-example", "index": ALL}, "n_clicks"),
        State({"type": "chat-example", "index": ALL}, "children"),
        prevent_initial_call=True,
    )
    def fill_example(n_clicks_list, labels):
        if not callback_context.triggered or not any(n_clicks_list):
            return dash.no_update
        triggered_idx = callback_context.triggered[0]["prop_id"]
        try:
            idx = json.loads(triggered_idx.split(".")[0])["index"]
            return labels[idx]
        except Exception as e:
            logger.error("fill_example: falha ao parsear índice: %s", e, exc_info=True)
            return dash.no_update

    # ── Enviar mensagem ────────────────────────────────────────────────────
    @app.callback(
        Output("chat-messages", "children", allow_duplicate=True),
        Output("chat-input", "value"),
        Output("chat-session-list", "children", allow_duplicate=True),
        Input("chat-send-btn", "n_clicks"),
        Input("chat-input", "n_submit"),
        State("chat-input", "value"),
        State("chat-session-id", "data"),
        State("chat-messages", "children"),
        prevent_initial_call=True,
        running=[
            (Output("chat-send-btn", "disabled"), True, False),
            (Output("chat-typing-indicator", "style"),
             {"padding": "0 20px 8px", "fontSize": "12px", "color": MUTED, "display": "block"},
             {"padding": "0 20px 8px", "fontSize": "12px", "color": MUTED, "display": "none"}),
        ],
    )
    def send_message(n_clicks, n_submit, user_text, session_id, current_messages):
        logger.debug("send_message | clicks=%s submit=%s text=%r session=%s",
                     n_clicks, n_submit, (user_text or "")[:80], session_id)

        if not callback_context.triggered:
            return dash.no_update, dash.no_update, dash.no_update

        if not user_text or not user_text.strip():
            return dash.no_update, dash.no_update, dash.no_update

        if not session_id:
            session_id = new_session_id()
            logger.debug("send_message: sessão criada na hora: %s", session_id)

        user_text = user_text.strip()
        current   = list(current_messages or []) + [_render_message("user", user_text)]

        try:
            reply = chat(session_id, user_text)
        except Exception as e:
            reply = f"⚠️ Erro ao processar mensagem: {e}"
            logger.error("send_message: chat() falhou (session=%s): %s", session_id, e, exc_info=True)

        current = current + [_render_message("assistant", reply)]
        return current, "", _build_session_list(session_id)