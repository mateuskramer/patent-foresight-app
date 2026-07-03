import os
import logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from data import (
    BG, CARD, BLUE, TEXT, MUTED, BORDER,
    term_list, patent_opts,
)
from callbacks import register_callbacks, create_dark_table
from dashboard import dashboard_builder_layout, gallery_layout, register_dashboard_callbacks

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css",
        dbc.themes.BOOTSTRAP,
    ],
    suppress_callback_exceptions=True,
)
app.title = "Patent Foresight Lab"

# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
CHART_OPTS = [{"label": "📈 Linha", "value": "line"}, {"label": "📊 Barra", "value": "bar"}]

def chart_type_radio(radio_id, label="Tipo de gráfico"):
    return html.Div([
        html.Label(label, style={"fontSize": "12px", "color": MUTED, "letterSpacing": "1px",
                                  "marginBottom": "6px", "display": "block"}),
        dcc.RadioItems(id=radio_id, options=CHART_OPTS, value="line", inline=True,
                       style={"color": TEXT},
                       labelStyle={"marginRight": "20px", "cursor": "pointer", "fontSize": "13px"}),
    ], style={"marginBottom": "16px"})

def gemini_panel(btn_id, output_id, cache_id):
    return html.Div([
        html.Div(style={"height": "1px",
                        "background": "linear-gradient(90deg, transparent, #1f2937, transparent)",
                        "margin": "22px 0 18px"}),
        html.Div([
            html.Button(
                [html.I(className="fas fa-wand-magic-sparkles me-2", style={"fontSize": "13px"}),
                 html.Span("Analyze with AI")],
                id=btn_id, n_clicks=0,
                style={"background": "rgba(37,99,235,0.12)", "border": "1px solid rgba(37,99,235,0.45)",
                       "color": "#93c5fd", "borderRadius": "8px", "padding": "7px 18px",
                       "fontSize": "13px", "fontWeight": "500", "cursor": "pointer",
                       "letterSpacing": "0.3px", "transition": "all 0.2s ease"},
            ),
            html.Span("Powered by Gemini 2.5 Flash",
                      style={"fontSize": "11px", "color": "#374151", "marginLeft": "12px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        dcc.Loading(html.Div(id=output_id), type="dot", color=BLUE, style={"marginTop": "12px"}),
        dcc.Store(id=cache_id, data={}),
    ])

def info_box(title, content_text, icon="fas fa-info-circle"):
    return dbc.Card(dbc.CardBody([
        html.Div([html.I(className=f"{icon} me-2", style={"color": BLUE}),
                  html.Strong(title, style={"color": TEXT})],
                 style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        html.P(content_text, style={"color": MUTED, "fontSize": "14px", "lineHeight": "1.5", "margin": 0}),
    ]), style={"background": "rgba(37,99,235,0.08)", "border": f"1px solid {BORDER}",
               "borderRadius": "12px", "marginBottom": "20px"})

# ─────────────────────────────────────────────
# LAYOUTS
# ─────────────────────────────────────────────
def home_layout():
    return html.Div([html.Div([
        html.Div("PATENT FORESIGHT LAB", style={"color": BLUE, "fontSize": "14px", "letterSpacing": "4px", "marginBottom": "24px", "fontWeight": "600"}),
        html.H1(["Inteligência semântica", html.Br(), "para evolução tecnológica"],
                style={"fontSize": "72px", "fontWeight": "700", "lineHeight": "1.05", "marginBottom": "34px", "maxWidth": "1000px"}),
        html.P("Plataforma de análise de patentes baseada em embeddings vetoriais, redes tecnológicas, convergência semântica e predição temporal com IA.",
               style={"fontSize": "22px", "lineHeight": "1.8", "color": MUTED, "maxWidth": "900px"}),
        html.Br(), html.Br(),
        dbc.Button([html.I(className="fas fa-arrow-right", style={"marginRight": "10px"}), "Explorar Plataforma"],
                   href="/similaridade",
                   style={"background": BLUE, "border": "none", "padding": "18px 34px",
                          "borderRadius": "18px", "fontSize": "16px", "fontWeight": "600"}),
    ], style={"paddingTop": "120px", "paddingLeft": "40px"})])


def dicionario_layout():
    return html.Div([
        html.H1("Dicionário", style={"marginBottom": "30px"}),
        info_box("Pipeline de Processamento", "Controle do pipeline local de embeddings e extração semântica."),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("STATUS", style={"fontSize": "12px", "letterSpacing": "2px", "color": BLUE, "marginBottom": "10px"}),
                html.H3("Pipeline Local", style={"fontWeight": "700"}),
                html.Div(id="pipeline-status", children="● Sistema pronto", style={"color": "#22c55e", "fontSize": "15px", "marginTop": "20px"}),
                html.Br(),
                dbc.Button([html.I(className="fas fa-play", style={"marginRight": "10px"}), "Iniciar"], id="start-process-btn", color="primary", style={"height": "48px", "borderRadius": "14px", "fontWeight": "600", "width": "100%"}),
                html.Br(),
                dbc.Button([html.I(className="fas fa-stop", style={"marginRight": "10px"}), "Parar"], id="stop-process-btn", color="danger", style={"height": "48px", "borderRadius": "14px", "fontWeight": "600", "width": "100%"}),
            ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "24px", "height": "100%"}), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("LOG", style={"fontSize": "13px", "letterSpacing": "2px", "color": BLUE, "marginBottom": "20px"}),
                html.Div(id="process-log", children=[html.Div("Aguardando...", style={"color": "#9ca3af"})],
                         style={"height": "300px", "overflowY": "auto", "background": "#030303",
                                "border": "1px solid #111827", "borderRadius": "18px", "padding": "22px",
                                "fontFamily": "monospace", "fontSize": "14px"}),
            ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "24px"}), md=8),
        ], style={"marginBottom": "40px"}),
        html.Hr(style={"borderColor": BORDER, "margin": "40px 0"}),
        dbc.Card(dbc.CardBody([
            html.H4("Adicionar Novo Termo", style={"marginBottom": "15px"}),
            dbc.Row([
                dbc.Col(dcc.Input(id="new-term-input", type="text", placeholder="Digite o novo termo...", style={"color": "#000", "borderRadius": "8px", "height": "40px", "width": "100%"}), md=8),
                dbc.Col(dbc.Button("Adicionar", id="add-term-btn", color="success", className="w-100", style={"height": "40px"}), md=4),
            ]),
            html.Div(id="add-term-msg", style={"marginTop": "10px"}),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px", "marginBottom": "20px"}),
        dbc.Card(dbc.CardBody([
            html.H4("Remover Termo", style={"marginBottom": "15px"}),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id="delete-term-dropdown", options=[], value=None, placeholder="Selecione...", style={"color": "#000"}), md=8),
                dbc.Col(dbc.Button("Remover", id="delete-term-btn", color="danger", className="w-100", style={"height": "40px"}), md=4),
            ]),
            html.Div(id="delete-term-msg", style={"marginTop": "10px"}),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px", "marginBottom": "20px"}),
        html.H4("Termos Atuais", style={"marginBottom": "15px"}),
        dbc.Card(dbc.CardBody([dcc.Loading(dbc.Table(id="dict-table"), type="circle", color=BLUE)]),
                 style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px", "marginBottom": "20px"}),
        dbc.Button("Atualizar Lista", id="refresh-dict-btn", color="secondary", outline=True),
    ])


def similarity_layout():
    return html.Div([
        html.H1("Similaridade Vetorial", style={"marginBottom": "30px"}),
        info_box("Similaridade Vetorial", "Cada patente é representada como ponto em espaço multidimensional via embeddings."),
        dbc.Card(dbc.CardBody([
            html.Label("Patente alvo"), html.Br(),
            dcc.Dropdown(id="patent-dropdown", options=patent_opts, value=0 if patent_opts else None, style={"color": "#000"}),
            html.Br(),
            dcc.Loading(dbc.Table(id="similarity-results"), type="circle", color=BLUE),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px"}),
    ])


def tendencias_layout():
    opts = [{"label": t, "value": t} for t in term_list]
    default = term_list[0] if term_list else None
    return html.Div([
        html.H1("Tendências", style={"marginBottom": "30px"}),
        info_box("O que mostra este gráfico?", "Evolução temporal da frequência de um termo técnico nas patentes."),
        dbc.Card(dbc.CardBody([
            html.Label("Selecione o Termo (aplicado a todos os gráficos desta página)"),
            dcc.Dropdown(id="master-tendencia-dropdown", options=opts, value=default, placeholder="Selecione", style={"color": "#000"}),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px", "marginBottom": "30px"}),
        dbc.Card(dbc.CardBody([
            html.H4("Gráfico de Tendência", style={"marginBottom": "16px"}),
            chart_type_radio("tend-chart-type"),
            dcc.Loading(dcc.Graph(id="tendencia-graph"), type="circle", color=BLUE),
            gemini_panel("btn-ai-tend", "ai-out-tend", "ai-cache-tend"),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px", "marginBottom": "30px"}),
        html.Hr(style={"borderColor": BORDER, "margin": "30px 0"}),
        html.H3("Comparação de Múltiplos Termos", style={"marginBottom": "20px", "color": TEXT}),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([html.Label("Termo 1"), dcc.Dropdown(id="tendencia-comp-1", options=opts, value=None, placeholder="Nenhum", style={"color": "#000"})], md=4),
                dbc.Col([html.Label("Termo 2"), dcc.Dropdown(id="tendencia-comp-2", options=opts, value=None, placeholder="Nenhum", style={"color": "#000"})], md=4),
                dbc.Col([html.Label("Termo 3"), dcc.Dropdown(id="tendencia-comp-3", options=opts, value=None, placeholder="Nenhum", style={"color": "#000"})], md=4),
            ], style={"marginBottom": "20px"}),
            chart_type_radio("comp-chart-type", "Tipo de gráfico (comparação)"),
            dcc.Loading(dcc.Graph(id="tendencia-comp-graph"), type="circle", color=BLUE),
            gemini_panel("btn-ai-comp", "ai-out-comp", "ai-cache-comp"),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px", "marginBottom": "40px"}),
        html.Hr(style={"borderColor": BORDER, "margin": "40px 0"}),
        html.H2("Inteligência Preditiva (TFT)", style={"marginBottom": "20px", "color": TEXT}),
        info_box("Como funciona a Predição com TFT?", "O Temporal Fusion Transformer aprende padrões temporais complexos."),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Último real", style={"color": MUTED, "fontSize": "12px"}), html.H3(id="pred-last-real", children="0 pat.")]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "16px"}), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Próximo mês (q50)", style={"color": MUTED, "fontSize": "12px"}), html.H3(id="pred-next", children="0.0", style={"color": BLUE}), html.Small(id="pred-delta", children="—", style={"color": "#22c55e"})])), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Pessimista (q10)", style={"color": MUTED, "fontSize": "12px"}), html.H3(id="pred-pess", children="—")]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "16px"}), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Otimista (q90)", style={"color": MUTED, "fontSize": "12px"}), html.H3(id="pred-opt", children="—")]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "16px"}), md=3),
        ], style={"marginBottom": "20px"}),
        dbc.Card(dbc.CardBody([
            html.H4("Gráfico de Predição", style={"marginBottom": "20px"}),
            html.Div([html.Span("🔵 Histórico"), html.Span("🟢 Previsão q50", style={"marginLeft": "20px"}), html.Span("🟦 Banda q10–q90", style={"marginLeft": "20px"})],
                     style={"fontSize": "13px", "marginBottom": "12px", "color": MUTED}),
            dcc.Loading(dcc.Graph(id="pred-graph"), type="circle", color=BLUE),
            gemini_panel("btn-ai-pred", "ai-out-pred", "ai-cache-pred"),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px"}),
    ])


def network_layout():
    return html.Div([
        html.H1("Rede Tecnológica", style={"marginBottom": "30px"}),
        info_box("Rede de Co-ocorrência", "Mostra como termos técnicos aparecem juntos nas mesmas patentes."),
        dbc.Card(dbc.CardBody([
            html.Label("Termo raiz"),
            dcc.Dropdown(id="termo-rede-dropdown", options=[{"label": t, "value": t} for t in term_list], value=term_list[0] if term_list else None, style={"color": "#000"}),
            html.Br(), html.Label("Camadas"),
            dcc.Slider(id="rede-depth-slider", min=1, max=5, step=1, value=3, marks={i: str(i) for i in range(1, 6)}, tooltip={"placement": "bottom", "always_visible": True}),
            html.Br(),
            dcc.Loading(dcc.Graph(id="network-graph"), type="circle", color=BLUE),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px"}),
    ])


def indicadores_layout():
    opts = [{"label": t, "value": t} for t in term_list]
    default = term_list[0] if term_list else None
    return html.Div([
        html.H1("Indicadores", style={"marginBottom": "30px"}),
        info_box("Indicadores", "Growth % | Density | Fusion | Shift % | Future Score"),
        dbc.Card(dbc.CardBody([
            html.Label("Selecione o Termo"),
            dcc.Dropdown(id="master-indicadores-dropdown", options=opts, value=default, placeholder="Selecione", style={"color": "#000"}),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px", "marginBottom": "30px"}),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Growth %", style={"color": MUTED, "fontSize": "12px"}), html.H3(id="ind-growth", children="0.00%", style={"color": "#22c55e"})]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "16px"}), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Density", style={"color": MUTED, "fontSize": "12px"}), html.H3(id="ind-density", children="0")]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "16px"}), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Fusion", style={"color": MUTED, "fontSize": "12px"}), html.H3(id="ind-fusion", children="0")]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "16px"}), md=4),
        ], style={"marginBottom": "20px"}),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Shift %", style={"color": MUTED, "fontSize": "12px"}), html.H3(id="ind-shift", children="0.00%")]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "16px"}), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Future Score", style={"color": MUTED, "fontSize": "12px"}), html.H3(id="ind-future", children="0.00", style={"color": BLUE, "fontSize": "36px"})])), md=6),
        ], style={"marginBottom": "20px"}),
        html.Hr(style={"borderColor": BORDER, "margin": "40px 0"}),
        html.H3("Ranking Top 15 por Future Score", style={"marginBottom": "20px", "color": TEXT}),
        dbc.Card(dbc.CardBody([
            chart_type_radio("ind-chart-type"),
            dcc.Loading(dcc.Graph(id="indicadores-bar-chart"), type="circle", color=BLUE),
            gemini_panel("btn-ai-ind", "ai-out-ind", "ai-cache-ind"),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px"}),
        html.Hr(style={"borderColor": BORDER, "margin": "40px 0"}),
        html.H2("Evolução Semântica", style={"marginBottom": "20px", "color": TEXT}),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("Similaridade Contextual"), html.P("1.0 = idêntico.", style={"fontSize": "12px", "color": MUTED}), html.H2(id="evol-sim-val", children="0.0000", style={"color": BLUE})]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px"}), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("Shift % (Mutação)"), html.P("Percentual de mudança semântica.", style={"fontSize": "12px", "color": MUTED}), html.H2(id="evol-shift-val", children="0.00%", style={"color": "#e74c3c"})]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px"}), md=6),
        ]),
    ])


def correlacao_layout():
    return html.Div([
        html.H1("Correlação", style={"marginBottom": "30px"}),
        info_box("Lift, Jaccard e PMI", "Lift >1 = co-ocorrência acima do acaso | Jaccard: proporção | PMI: logarítmico"),
        dbc.Card(dbc.CardBody([
            html.Label("Termo base"),
            dcc.Dropdown(id="termo-correlacao-dropdown", options=[{"label": t, "value": t} for t in term_list], value=term_list[0] if term_list else None, style={"color": "#000"}),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px", "marginBottom": "30px"}),
        dbc.Card(dbc.CardBody([
            html.H4("Correlações de Co-ocorrência", style={"marginBottom": "20px"}),
            dcc.Loading(dbc.Table(id="correlacao-table"), type="circle", color=BLUE),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "16px", "marginBottom": "30px"}),
        html.Div([
            html.Hr(style={"borderColor": BORDER, "margin": "30px 0"}),
            html.H3("Correlação Temporal (Pearson)", style={"marginBottom": "20px", "color": TEXT}),
            dbc.Row([
                dbc.Col([html.Label("Quantos termos exibir"), dcc.Slider(id="corr-topn-slider", min=5, max=40, step=5, value=15, marks={i: str(i) for i in range(5,41,5)}, tooltip={"placement": "bottom", "always_visible": True})], md=6),
                dbc.Col([html.Label("Filtro mínimo |r|"), dcc.Slider(id="corr-threshold-slider", min=0.0, max=0.9, step=0.05, value=0.0, marks={i/10: str(i/10) for i in range(0,10,2)}, tooltip={"placement": "bottom", "always_visible": True})], md=6),
            ], style={"marginBottom": "20px"}),
            dbc.Card(dbc.CardBody([
                html.H5("Ranking de Correlação", style={"marginBottom": "16px"}),
                chart_type_radio("corr-chart-type"),
                dcc.Loading(dcc.Graph(id="pearson-bar-chart"), type="circle", color=BLUE),
                html.P("Verde = crescem juntos | Vermelho = anticorrelação", style={"fontSize": "12px", "color": MUTED, "textAlign": "center", "marginTop": "10px"}),
                gemini_panel("btn-ai-corr", "ai-out-corr", "ai-cache-corr"),
            ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "16px", "marginBottom": "20px"}),
            dbc.Card(dbc.CardBody([
                html.H5("Comparar Séries Temporais", style={"marginBottom": "16px"}),
                dcc.Dropdown(id="pearson-partner-dropdown", options=[], value=None, placeholder="Selecione um termo para comparar", style={"color": "#000", "marginBottom": "16px"}),
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([html.Small("Pearson r", style={"color": MUTED, "fontSize": "12px"}), html.H4(id="pearson-r-metric", children="—", style={"color": BLUE, "margin": 0})]), style={"background": "rgba(37,99,235,0.1)", "borderRadius": "12px", "textAlign": "center"}), md=6),
                    dbc.Col(dbc.Card(dbc.CardBody([html.Small("p-value", style={"color": MUTED, "fontSize": "12px"}), html.H4(id="pearson-p-metric", children="—", style={"color": TEXT, "margin": 0})]), style={"background": "rgba(107,114,128,0.1)", "borderRadius": "12px", "textAlign": "center"}), md=6),
                ], style={"marginBottom": "20px"}),
                chart_type_radio("pearson-line-chart-type", "Tipo de gráfico (comparação temporal)"),
                dcc.Loading(dcc.Graph(id="pearson-line-chart"), type="circle", color=BLUE),
            ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "16px", "marginBottom": "20px"}),
            dbc.Accordion([
                dbc.AccordionItem([dbc.Table(id="pearson-full-table", className="table table-dark table-borderless table-hover", responsive=True, style={"color": TEXT})], title="Ver tabela completa de correlações"),
            ], start_collapsed=True, flush=True, style={"background": "transparent"}),
        ], style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px", "padding": "20px"}),
    ])


def ranking_layout():
    return html.Div([
        html.H1("Ranking", style={"marginBottom": "30px"}),
        info_box("Future Score", "Growth (35%) + Fusion (25%) + Shift (20%) + Density (20%)"),
        dbc.Card(dbc.CardBody([
            html.H4("Top 30 por Future Score", style={"marginBottom": "20px"}),
            dcc.Loading(dbc.Table(id="ranking-table"), type="circle", color=BLUE),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px"}),
    ])


def oportunidades_layout():
    return html.Div([
        html.H1("Oportunidades", style={"marginBottom": "30px"}),
        info_box("Oportunidades Esparsas", "Termos que nunca co-ocorreram mas compartilham vizinhos — potencial para inovação cruzada."),
        dbc.Card(dbc.CardBody([
            html.Label("Termo âncora"),
            dcc.Dropdown(id="termo-oportunidades-dropdown", options=[{"label": t, "value": t} for t in term_list], value=term_list[0] if term_list else None, style={"color": "#000"}),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px", "marginBottom": "30px"}),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("Oportunidades Esparsas", style={"marginBottom": "20px"}), dcc.Loading(dbc.Table(id="sparse-table"), type="circle", color=BLUE)]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px"}), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("Bridge Strength", style={"marginBottom": "20px"}), chart_type_radio("opp-chart-type"), dcc.Loading(dcc.Graph(id="sparse-graph"), type="circle", color=BLUE), gemini_panel("btn-ai-opp", "ai-out-opp", "ai-cache-opp")]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "20px"}), md=6),
        ]),
    ])


def chat_layout():
    return html.Div([
        dcc.Store(id="chat-session-id", data=None),
        dcc.Store(id="chat-sidebar-open", data=True),
        html.Div(style={"display": "flex", "height": "calc(100vh - 80px)", "gap": "20px"}, children=[
            html.Div(id="chat-sidebar", style={"width": "260px", "flexShrink": "0", "background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "16px", "padding": "16px", "display": "flex", "flexDirection": "column", "gap": "8px", "transition": "all 0.3s ease", "overflow": "hidden"}, children=[
                html.Div(style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "marginBottom": "8px"}, children=[
                    html.Div("CONVERSAS", id="chat-sidebar-label", style={"fontSize": "11px", "fontWeight": "700", "letterSpacing": "2px", "color": MUTED}),
                    html.Button(html.I(className="fas fa-chevron-left", id="chat-sidebar-icon"), id="chat-sidebar-toggle", n_clicks=0,
                                style={"background": "transparent", "border": "none", "color": MUTED, "cursor": "pointer", "fontSize": "13px", "padding": "4px 6px", "borderRadius": "6px", "transition": "all 0.2s"}),
                ]),
                html.Button([html.I(className="fas fa-plus me-2", id="chat-new-icon"), html.Span("Nova conversa", id="chat-new-label")], id="chat-new-btn", n_clicks=0,
                            style={"background": "rgba(37,99,235,0.15)", "border": f"1px solid {BLUE}", "color": "#93c5fd", "borderRadius": "10px", "padding": "8px 14px", "fontSize": "13px", "cursor": "pointer", "width": "100%", "marginBottom": "8px", "fontWeight": "500", "transition": "all 0.3s", "whiteSpace": "nowrap", "overflow": "hidden"}),
                html.Div(id="chat-session-list", style={"overflowY": "auto", "flex": "1"}),
            ]),
            html.Div(style={"flex": "1", "display": "flex", "flexDirection": "column", "background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "16px", "overflow": "hidden"}, children=[
                html.Div(style={"padding": "16px 20px", "borderBottom": f"1px solid {BORDER}", "display": "flex", "alignItems": "center", "gap": "12px"}, children=[
                    html.Div(style={"width": "36px", "height": "36px", "borderRadius": "50%", "background": "rgba(37,99,235,0.2)", "display": "flex", "alignItems": "center", "justifyContent": "center"}, children=[html.I(className="fas fa-robot", style={"color": BLUE, "fontSize": "16px"})]),
                    html.Div([
                        html.Div("Patent Intelligence Assistant", style={"fontWeight": "600", "fontSize": "14px", "color": TEXT}),
                        html.Div("Acesso em tempo real aos dados de patentes", style={"fontSize": "12px", "color": MUTED}),
                    ]),
                    html.Div(style={"marginLeft": "auto"}, children=[
                        html.Button([html.I(className="fas fa-trash-alt me-1"), "Apagar"], id="chat-delete-btn", n_clicks=0,
                                    style={"background": "rgba(231,76,60,0.1)", "border": "1px solid rgba(231,76,60,0.3)", "color": "#f87171", "borderRadius": "8px", "padding": "5px 12px", "fontSize": "12px", "cursor": "pointer"}),
                    ]),
                ]),
                dcc.Loading(
                    html.Div(id="chat-messages", style={"flex": "1", "overflowY": "auto", "padding": "20px", "display": "flex", "flexDirection": "column", "gap": "12px"}, children=[
                        html.Div(style={"display": "flex", "gap": "10px", "alignItems": "flex-start"}, children=[
                            html.Div(style={"width": "32px", "height": "32px", "borderRadius": "50%", "background": "rgba(37,99,235,0.2)", "flexShrink": "0", "display": "flex", "alignItems": "center", "justifyContent": "center"}, children=[html.I(className="fas fa-robot", style={"color": BLUE, "fontSize": "13px"})]),
                            html.Div(style={"background": "rgba(37,99,235,0.08)", "border": "1px solid rgba(37,99,235,0.2)", "borderRadius": "4px 16px 16px 16px", "padding": "12px 16px", "maxWidth": "75%", "fontSize": "13.5px", "lineHeight": "1.7", "color": "#d1d5db"}, children=[
                                html.P("Olá! Sou o assistente de inteligência em patentes do Patent Foresight Lab.", style={"margin": "0 0 6px"}),
                                html.P("Tenho acesso em tempo real aos dados do banco — posso analisar tendências, comparar termos, identificar oportunidades e responder perguntas estratégicas.", style={"margin": "0 0 6px"}),
                            ]),
                        ]),
                    ]),
                    type="dot", color=BLUE, style={"flex": "1", "display": "flex", "flexDirection": "column"},
                ),
                html.Div(style={"padding": "12px 20px 0", "display": "flex", "gap": "8px", "flexWrap": "wrap"}, children=[
                    html.Span("Sugestões:", style={"fontSize": "11px", "color": MUTED, "alignSelf": "center", "marginRight": "4px"}),
                    *[html.Button(q, id={"type": "chat-example", "index": i}, n_clicks=0, style={"background": "rgba(37,99,235,0.08)", "border": "1px solid rgba(37,99,235,0.25)", "color": "#93c5fd", "borderRadius": "20px", "padding": "4px 12px", "fontSize": "12px", "cursor": "pointer", "transition": "all 0.2s", "whiteSpace": "nowrap"})
                      for i, q in enumerate(["Qual termo tem maior crescimento?", "Quais são os top 5 por future score?", "Qual termo tem maior fusão semântica?", "Resumo geral do portfólio de patentes"])],
                ]),
                html.Div(style={"padding": "12px 20px 16px", "borderTop": f"1px solid {BORDER}", "marginTop": "10px", "display": "flex", "gap": "10px", "alignItems": "center"}, children=[
                    dcc.Input(id="chat-input", type="text", placeholder="Pergunte sobre tendências, termos, oportunidades ou estratégia de patentes...", n_submit=0, debounce=False,
                              style={"flex": "1", "background": "#111827", "border": f"1px solid {BORDER}", "borderRadius": "12px", "color": TEXT, "padding": "12px 16px", "fontSize": "13.5px", "height": "48px", "fontFamily": "Inter, sans-serif", "outline": "none"}),
                    html.Button(html.I(className="fas fa-paper-plane"), id="chat-send-btn", n_clicks=0, disabled=False,
                                style={"background": BLUE, "border": "none", "color": "white", "borderRadius": "12px", "width": "48px", "height": "48px", "cursor": "pointer", "fontSize": "16px", "flexShrink": "0"}),
                ]),
                html.Div(id="chat-typing-indicator", style={"padding": "0 20px 8px", "fontSize": "12px", "color": MUTED, "display": "none"}, children="● Analisando dados e gerando resposta..."),
            ]),
        ]),
    ], style={"padding": "0"})


# ─────────────────────────────────────────────
# ROTAS
# ─────────────────────────────────────────────
page_routes = {
    "/":              home_layout,
    "/dicionario":    dicionario_layout,
    "/similaridade":  similarity_layout,
    "/tendencias":    tendencias_layout,
    "/rede":          network_layout,
    "/indicadores":   indicadores_layout,
    "/correlacao":    correlacao_layout,
    "/ranking":       ranking_layout,
    "/oportunidades": oportunidades_layout,
    "/dashboard":     gallery_layout,
    "/dashboards":    dashboard_builder_layout,
    "/chat":          chat_layout,
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
sidebar = html.Div(
    id="sidebar", className="sidebar sidebar-open",
    children=[
        html.Div([
            html.Div([
                html.I(className="fas fa-brain", style={"fontSize": "28px", "color": BLUE}),
                html.Div([
                    html.Div("PATENT",        style={"fontSize": "17px", "fontWeight": "700", "letterSpacing": "1px"}),
                    html.Div("FORESIGHT LAB", style={"fontSize": "11px", "color": MUTED, "letterSpacing": "2px"}),
                ], className="nav-text"),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Button(html.I(className="fas fa-bars"), id="toggle-sidebar", n_clicks=0,
                        style={"background": "transparent", "border": "none", "color": TEXT, "fontSize": "20px", "cursor": "pointer"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "padding": "24px 20px"}),
        html.Div(style={"height": "1px", "background": "#111827", "marginBottom": "20px"}),
        dbc.Nav([
            dbc.NavLink([html.I(className="fas fa-house"),           html.Span(" Home",              className="nav-text")], href="/",              active="exact"),
            dbc.NavLink([html.I(className="fas fa-book"),            html.Span(" Dicionário",         className="nav-text")], href="/dicionario",    active="exact"),
            dbc.NavLink([html.I(className="fas fa-vector-square"),   html.Span(" Similaridade",       className="nav-text")], href="/similaridade",  active="exact"),
            dbc.NavLink([html.I(className="fas fa-chart-line"),      html.Span(" Tendências",         className="nav-text")], href="/tendencias",    active="exact"),
            dbc.NavLink([html.I(className="fas fa-project-diagram"), html.Span(" Rede Tecnológica",   className="nav-text")], href="/rede",          active="exact"),
            dbc.NavLink([html.I(className="fas fa-microchip"),       html.Span(" Indicadores",        className="nav-text")], href="/indicadores",   active="exact"),
            dbc.NavLink([html.I(className="fas fa-fire"),            html.Span(" Correlação",         className="nav-text")], href="/correlacao",    active="exact"),
            dbc.NavLink([html.I(className="fas fa-trophy"),          html.Span(" Ranking",            className="nav-text")], href="/ranking",       active="exact"),
            dbc.NavLink([html.I(className="fas fa-star"),            html.Span(" Oportunidades",      className="nav-text")], href="/oportunidades", active="exact"),
            html.Div(style={"height": "1px", "background": "#111827", "margin": "12px 0"}),
            dbc.NavLink([html.I(className="fas fa-table-columns"),   html.Span(" Dashboard",          className="nav-text")], href="/dashboard",     active="exact",
                        style={"color": "#34d399 !important"}),
            dbc.NavLink([html.I(className="fas fa-chart-area"),      html.Span(" Dashboard Builder",  className="nav-text")], href="/dashboards",    active="exact",
                        style={"color": "#a78bfa !important"}),
            dbc.NavLink([html.I(className="fas fa-comments"),        html.Span(" Chat Análises",      className="nav-text")], href="/chat",          active="exact",
                        style={"color": "#60a5fa !important"}),
        ], vertical=True, pills=True, style={"padding": "10px"}),

    ],
)

# ─────────────────────────────────────────────
# LAYOUT RAIZ
# ─────────────────────────────────────────────
content = html.Div(id="page-content", className="content content-open")

# botão de tema — overlay fixo canto superior direito
theme_btn = html.Button(
    html.I(className="fas fa-sun", id="theme-icon"),
    id="theme-toggle", n_clicks=0,
    title="Alternar tema",
    style={
        "position": "fixed", "top": "16px", "right": "20px", "zIndex": "1000",
        "background": "transparent", "border": "1px solid var(--border)",
        "color": "var(--muted)", "borderRadius": "8px",
        "width": "36px", "height": "36px",
        "cursor": "pointer", "fontSize": "15px",
        "display": "flex", "alignItems": "center", "justifyContent": "center",
        "transition": "all 0.2s",
    },
)

app.layout = html.Div(
    id="app-root",
    children=[
        dcc.Location(id="url"),
        dcc.Store(id="sidebar-state", data="open"),
        dcc.Store(id="theme-store",   data="dark"),
        sidebar,
        content,
        theme_btn,
        dcc.Interval(id="process-log-interval", interval=3000, n_intervals=0),
        html.Div(id="wake-up-status-div", style={"display": "none"}),
    ],
    className="theme-dark",
)

# ─────────────────────────────────────────────
# THEME CALLBACK
# ─────────────────────────────────────────────
@app.callback(
    Output("app-root",    "className"),
    Output("theme-store", "data"),
    Output("theme-icon",  "className"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store",  "data"),
    prevent_initial_call=True,
)
def toggle_theme(n, current):
    if current == "dark":
        return "theme-light", "light", "fas fa-moon"
    return "theme-dark", "dark", "fas fa-sun"

# ─────────────────────────────────────────────
# REGISTRAR CALLBACKS
# ─────────────────────────────────────────────
register_callbacks(app, page_routes)
register_dashboard_callbacks(app)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        /* ── VARIÁVEIS DE TEMA ───────────────────────────────────────── */
        .theme-dark {
            --bg:       #000000;
            --card:     #0a0a0a;
            --sidebar:  #050505;
            --text:     #ffffff;
            --muted:    #6b7280;
            --border:   #1f2937;
            --input-bg: #111827;
            --hover-row:rgba(37,99,235,0.1);
            --scroll-track:#050505;
            --scroll-thumb:#1f2937;
            --h-color:  #ffffff;
            --p-color:  #9ca3af;
            --table-text:#ffffff;
        }
        .theme-light {
            --bg:       #f1f5f9;
            --card:     #ffffff;
            --sidebar:  #ffffff;
            --text:     #0f172a;
            --muted:    #64748b;
            --border:   #e2e8f0;
            --input-bg: #f8fafc;
            --hover-row:rgba(37,99,235,0.06);
            --scroll-track:#f1f5f9;
            --scroll-thumb:#cbd5e1;
            --h-color:  #0f172a;
            --p-color:  #475569;
            --table-text:#0f172a;
        }

        /* ── BASE ───────────────────────────────────────────────────── */
        body { margin:0; padding:0; overflow-x:hidden; font-family:Inter,sans-serif; }
        #app-root { background: var(--bg); color: var(--text); min-height:100vh; transition: background 0.3s, color 0.3s; }

        /* ── SIDEBAR ────────────────────────────────────────────────── */
        .sidebar { position:fixed; top:0; left:0; bottom:0; background:var(--sidebar); border-right:1px solid var(--border); z-index:999; transition:all 0.35s ease; overflow:hidden; }
        .sidebar-open   { width:290px; }
        .sidebar-closed { width:78px;  }
        .theme-light .sidebar { box-shadow: 2px 0 12px rgba(0,0,0,0.08); }

        /* ── CONTENT ────────────────────────────────────────────────── */
        .content { min-height:100vh; background:var(--bg); transition:all 0.35s ease; padding:40px; }
        .content-open   { margin-left:290px; }
        .content-closed { margin-left:78px;  }

        /* ── NAV ────────────────────────────────────────────────────── */
        .nav-link { color:var(--muted) !important; border-radius:14px; margin-bottom:8px; padding:14px !important; transition:all 0.25s ease; font-size:15px; font-weight:500; white-space:nowrap; display:flex; align-items:center; }
        .nav-link:hover  { background:rgba(37,99,235,0.15); color:var(--text) !important; }
        .nav-link.active { background:rgba(37,99,235,0.20); color:var(--text) !important; border:1px solid rgba(37,99,235,0.35); }
        .nav-text { margin-left:14px; transition:opacity 0.2s ease; }
        .sidebar-closed .nav-text  { opacity:0; display:none; }
        .sidebar-closed .nav-link  { justify-content:center; }
        .sidebar-closed .nav-link i{ margin:0 !important; font-size:18px; }

        /* ── TIPOGRAFIA ─────────────────────────────────────────────── */
        h1,h2,h3,h4,h5 { color: var(--h-color) !important; }
        p { color: var(--p-color); }

        /* ── TABELAS ────────────────────────────────────────────────── */
        .table-dark th,.table-dark td { background-color:transparent !important; color:var(--table-text) !important; border-color:var(--border) !important; }
        .theme-light .table-dark { --bs-table-bg: transparent; }
        .table-hover tbody tr:hover { background:var(--hover-row) !important; }

        /* ── SCROLLBAR ──────────────────────────────────────────────── */
        ::-webkit-scrollbar { width:8px; }
        ::-webkit-scrollbar-track { background:var(--scroll-track); }
        ::-webkit-scrollbar-thumb { background:var(--scroll-thumb); border-radius:10px; }

        /* ── ANIMAÇÕES ──────────────────────────────────────────────── */
        button[id^="btn-ai-"]:hover { background:rgba(37,99,235,0.22) !important; border-color:rgba(37,99,235,0.7) !important; color:#bfdbfe !important; }
        @keyframes fadeInUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        [id^="ai-out-"] > div > div:not(:first-child) { animation:fadeInUp 0.35s ease both; }
        [id^="ai-out-"] > div > div:nth-child(2){animation-delay:0.04s}
        [id^="ai-out-"] > div > div:nth-child(3){animation-delay:0.10s}
        [id^="ai-out-"] > div > div:nth-child(4){animation-delay:0.16s}
        [id^="ai-out-"] > div > div:nth-child(5){animation-delay:0.22s}
        [id^="ai-out-"] > div > div:nth-child(6){animation-delay:0.28s}

        /* ── THEME TOGGLE BTN ───────────────────────────────────────── */
        #theme-toggle:hover { background:rgba(37,99,235,0.1) !important; border-color:rgba(37,99,235,0.4) !important; color:var(--text) !important; }

        /* ── CHAT ───────────────────────────────────────────────────── */
        #chat-input { transition:border-color 0.2s,box-shadow 0.2s; background:var(--input-bg) !important; color:var(--text) !important; border-color:var(--border) !important; }
        #chat-input:focus { border-color:rgba(37,99,235,0.6) !important; box-shadow:0 0 0 3px rgba(37,99,235,0.1); }
        #chat-send-btn:hover { background:#1d4ed8 !important; }
        #chat-new-btn:hover  { background:rgba(37,99,235,0.25) !important; }
        .chat-session-item { padding:10px 12px; border-radius:10px; cursor:pointer; font-size:13px; color:var(--muted); border:1px solid transparent; margin-bottom:4px; transition:all 0.2s; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .chat-session-item:hover  { background:rgba(37,99,235,0.1); color:var(--text); border-color:rgba(37,99,235,0.3); }
        .chat-session-item.active { background:rgba(37,99,235,0.18); color:var(--text); border-color:rgba(37,99,235,0.4); }
        #chat-sidebar-toggle:hover { background:rgba(255,255,255,0.05) !important; }
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
        #chat-typing-indicator { animation:pulse 1.5s ease-in-out infinite; }

        /* ── DASHBOARD BUILDER ──────────────────────────────────────── */
        #db-chat-input { background:var(--input-bg) !important; color:var(--text) !important; border-color:var(--border) !important; }
        #db-chat-input:focus { border-color:rgba(37,99,235,0.6) !important; box-shadow:0 0 0 3px rgba(37,99,235,0.1); outline:none; }
        #db-send-btn:hover   { background:#1d4ed8 !important; }
        #db-new-btn:hover    { background:rgba(37,99,235,0.28) !important; }
        #db-delete-btn:hover { background:rgba(231,76,60,0.18) !important; }
        @keyframes db-pulse{0%,100%{opacity:1}50%{opacity:0.3}}
        #db-typing-indicator { animation:db-pulse 1.4s ease-in-out infinite; }
        [id*="db-example"]:hover { background:rgba(37,99,235,0.18) !important; border-color:rgba(37,99,235,0.5) !important; }
        @keyframes slideInRight{from{transform:translateX(40px);opacity:0}to{transform:translateX(0);opacity:1}}
        #db-drill-panel::-webkit-scrollbar { width:5px; }
        #db-drill-panel::-webkit-scrollbar-thumb { background:var(--scroll-thumb); border-radius:6px; }
        [id*="db-ctrl"]:hover { background:rgba(37,99,235,0.28) !important; color:#bfdbfe !important; }

        /* ── MODO CLARO — ajustes específicos ───────────────────────── */
        .theme-light .sidebar { background:var(--sidebar); }
        .theme-light #db-chat-input { border-color:var(--border) !important; }
        .theme-light .nav-link { color:var(--muted) !important; }
        .theme-light .nav-link:hover { background:rgba(37,99,235,0.08); color:#1e3a8a !important; }
        .theme-light .nav-link.active { color:#1e3a8a !important; }

        /* ── MODO CLARO — sobrescreve estilos inline escuros ───────────
           Os layouts usam cores fixas inline (#0a0a0a etc). Estas regras
           forçam a aparência clara por cima desses inline styles. */
        .theme-light .content { background:#f1f5f9 !important; }
        /* cards (dbc.Card normalmente tem background inline escuro) */
        .theme-light .card,
        .theme-light .card-body { background:#ffffff !important; color:#0f172a !important; border-color:#e2e8f0 !important; }
        /* divs com fundo escuro inline mais comuns */
        .theme-light [style*="background: #0a0a0a"],
        .theme-light [style*="background:#0a0a0a"],
        .theme-light [style*="background: #050505"],
        .theme-light [style*="background:#050505"],
        .theme-light [style*="background: rgb(10, 10, 10)"],
        .theme-light [style*="background: rgb(5, 5, 5)"],
        .theme-light [style*="background: #000000"],
        .theme-light [style*="background:#000000"],
        .theme-light [style*="background: #000"],
        .theme-light [style*="background:#000"] {
            background:#ffffff !important;
            border-color:#e2e8f0 !important;
        }
        /* texto branco inline -> escuro */
        .theme-light [style*="color: #ffffff"],
        .theme-light [style*="color:#ffffff"],
        .theme-light [style*="color: white"],
        .theme-light [style*="color:white"],
        .theme-light [style*="color: #fff"],
        .theme-light [style*="color:#fff"] {
            color:#0f172a !important;
        }
        /* inputs e dropdowns no claro */
        .theme-light input, .theme-light textarea {
            background:#f8fafc !important; color:#0f172a !important; border-color:#e2e8f0 !important;
        }
        /* dropdowns do dash (react-select) já têm fundo branco; texto preto definido inline color:#000 — manter */
        /* botão de tema visível em ambos */
        .theme-light #theme-toggle { color:#475569 !important; border-color:#cbd5e1 !important; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""

server = app.server

if __name__ == "__main__":
    debug_mode = os.environ.get("DASH_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(debug=debug_mode, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))