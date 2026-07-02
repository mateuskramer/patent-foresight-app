import re
import json
import pandas as pd
from dash import html

from llm import generate

from data import (
    BLUE, MUTED,
    monthly_term_count,
    calc_growth, calc_density, calc_fusion, calc_shift, calc_future_score,
)

def build_term_context(term, hist, pred=None):
    ctx = {"term": term}
    if not hist.empty:
        h = hist.copy()
        h["year_month"] = pd.to_datetime(h["year_month"], errors="coerce")
        h = h.dropna(subset=["year_month"]).sort_values("year_month")
        counts = h["count"].tolist()
        months = h["year_month"].dt.strftime("%Y-%m").tolist()
        ctx.update({
            "total_patents":   int(h["count"].sum()),
            "first_month":     months[0],
            "last_month":      months[-1],
            "monthly_average": round(float(h["count"].mean()), 1),
            "monthly_std_dev": round(float(h["count"].std()), 1),
            "peak_value":      int(h["count"].max()),
            "peak_month":      h.loc[h["count"].idxmax(), "year_month"].strftime("%Y-%m"),
            "lowest_value":    int(h["count"].min()),
            "last_6_months":   [{"month": m, "count": int(c)}
                                 for m, c in zip(months[-6:], counts[-6:])],
            "annual_history":  (h.assign(year=h["year_month"].dt.year)
                                  .groupby("year")["count"].sum().to_dict()),
            "last_24_months":  [{"month": m, "count": int(c)}
                                 for m, c in zip(months[-24:], counts[-24:])],
            "top_3_peaks":     (h.nlargest(3, "count")
                                  .assign(year_month=lambda x: x["year_month"].dt.strftime("%Y-%m"))
                                  [["year_month", "count"]].to_dict("records")),
        })
        # anomalias
        limit = h["count"].mean() + 2 * h["count"].std()
        anomalies = (h[h["count"] > limit]
                       .assign(year_month=lambda x: x["year_month"].dt.strftime("%Y-%m"))
                       [["year_month", "count"]].to_dict("records"))
        if anomalies:
            ctx["anomalous_months"] = anomalies
        # momentum
        if len(h) >= 6:
            l3, p3 = h["count"].iloc[-3:].mean(), h["count"].iloc[-6:-3].mean()
            ctx["momentum_last3_vs_prev3_pct"] = round(((l3 - p3) / max(p3, 1)) * 100, 1)
        if len(h) >= 12:
            l6, p6 = h["count"].iloc[-6:].mean(), h["count"].iloc[-12:-6].mean()
            ctx["trend_last6_vs_prev6_pct"] = round(((l6 - p6) / max(p6, 1)) * 100, 1)
            ctx["avg_last_6_months"]        = round(float(l6), 2)
            ctx["avg_prev_6_months"]        = round(float(p6), 2)
        # crescimento geral
        mid = len(h) // 2
        if mid:
            ctx["overall_growth_pct"] = round(
                ((h["count"].iloc[mid:].mean() - h["count"].iloc[:mid].mean())
                 / max(h["count"].iloc[:mid].mean(), 1)) * 100, 1)
        # volatilidade
        cv = h["count"].std() / max(h["count"].mean(), 1)
        ctx["volatility_cv"]    = round(float(cv), 2)
        ctx["volatility_label"] = "high" if cv > 0.5 else "moderate" if cv > 0.25 else "low"
        # CAGR
        if len(h) >= 24 and h["count"].iloc[0] > 0:
            anos = len(h) / 12
            cagr = ((h["count"].iloc[-1] / h["count"].iloc[0]) ** (1 / anos) - 1) * 100
            ctx["cagr_pct"] = round(float(cagr), 1)

    if pred is not None and not pred.empty and "predicted_count" in pred.columns:
        ctx["forecast_horizon_months"] = len(pred)
        ctx["forecast_next_q50"]       = round(float(pred["predicted_count"].iloc[0]), 1)
        if len(pred) >= 3:
            ctx["forecast_3mo_q50"]    = round(float(pred["predicted_count"].iloc[2]), 1)
        if "pessimistic_count" in pred.columns:
            ctx["forecast_q10"]        = round(float(pred["pessimistic_count"].iloc[0]), 1)
        if "optimistic_count" in pred.columns:
            ctx["forecast_q90"]        = round(float(pred["optimistic_count"].iloc[0]), 1)
        if "target_year_month" in pred.columns:
            pred_aux = pred.copy()
            pred_aux["target_year_month"] = pd.to_datetime(
                pred_aux["target_year_month"], errors="coerce")
            ctx["forecast_series"] = (
                pred_aux[["target_year_month", "predicted_count"]]
                .assign(target_year_month=lambda x: x["target_year_month"].dt.strftime("%Y-%m"))
                .tail(12).to_dict("records"))
        if not hist.empty:
            last_real = hist["count"].iloc[-1]
            delta = ctx["forecast_next_q50"] - last_real
            ctx["forecast_delta"]     = round(float(delta), 1)
            ctx["forecast_delta_pct"] = round((delta / max(last_real, 1)) * 100, 1)
    return ctx


def build_indicators_context(term, df):
    return {
        "term":         term,
        "growth_pct":   round(calc_growth(term, df), 2),
        "density":      calc_density(term, df),
        "fusion":       calc_fusion(term, df),
        "shift_pct":    round(calc_shift(term, df), 2),
        "future_score": calc_future_score(term, df),
        "metric_definitions": {
            "growth":       "% change between last two months",
            "density":      "total unique patents with this term",
            "fusion":       "distinct co-occurring terms (breadth)",
            "shift":        "% semantic drift first vs last period (cosine)",
            "future_score": "Growth 35% + Fusion 25% + Shift 20% + Density 20%",
        },
    }


def build_comparison_context(terms, df):
    """
    FIX: recebe `df` explicitamente em vez de depender de import global.
    """
    return {
        "terms_compared": terms,
        "data": {t: build_term_context(t, monthly_term_count(t, df)) for t in terms},
    }


def build_correlation_context(term, corr_df, pearson_df):
    ctx = {"base_term": term}
    if not corr_df.empty:
        top = corr_df.head(10)
        ctx["top_cooccurrence"] = top.to_dict("records")
        ctx["max_lift"]    = round(float(top["lift"].max()), 4)    if "lift"    in top else None
        ctx["max_jaccard"] = round(float(top["jaccard"].max()), 4) if "jaccard" in top else None
    if not pearson_df.empty:
        top_p = pearson_df.head(10)
        ctx["top_temporal_correlations"] = top_p.to_dict("records")
        pos = top_p[top_p["pearson_r"] > 0]
        neg = top_p[top_p["pearson_r"] < 0]
        ctx["strongest_positive"] = pos.iloc[0].to_dict()  if not pos.empty else None
        ctx["strongest_negative"] = neg.iloc[-1].to_dict() if not neg.empty else None
    return ctx


def build_opportunities_context(term, sparse_df):
    ctx = {"anchor_term": term}
    if not sparse_df.empty:
        ctx.update({
            "top_opportunities": sparse_df.head(10).to_dict("records"),
            "total_found":       len(sparse_df),
            "max_bridge":        int(sparse_df["bridge_strength"].max()),
            "avg_bridge":        round(float(sparse_df["bridge_strength"].mean()), 1),
        })
    return ctx


def _briefing(ctx):
    lines = []
    if ctx.get("annual_history"):
        anos = sorted(ctx["annual_history"].keys())
        lines.append("Annual counts: " + ", ".join(
            f"{a}: {ctx['annual_history'][a]}" for a in anos))
    if ctx.get("trend_last6_vs_prev6_pct") is not None:
        d = "up" if ctx["trend_last6_vs_prev6_pct"] > 0 else "down"
        lines.append(
            f"6-month avg: {ctx.get('avg_last_6_months','?')} vs "
            f"{ctx.get('avg_prev_6_months','?')} ({ctx['trend_last6_vs_prev6_pct']:+.1f}%, {d})"
        )
    if ctx.get("peak_month"):
        lines.append(
            f"Peak: {ctx['peak_value']} in {ctx['peak_month']} "
            f"(avg {ctx.get('monthly_average','?')})"
        )
    if ctx.get("cagr_pct") is not None:
        lines.append(f"CAGR: {ctx['cagr_pct']:+.1f}%/yr")
    if ctx.get("volatility_cv") is not None:
        lines.append(
            f"Volatility (CV): {ctx['volatility_cv']:.2f} — {ctx.get('volatility_label','?')}"
        )
    if ctx.get("anomalous_months"):
        lines.append("Anomalous months: " + ", ".join(
            f"{m['year_month']} ({m['count']})" for m in ctx["anomalous_months"]))
    if ctx.get("last_6_months"):
        lines.append("Last 6 months: " + ", ".join(
            f"{m['month']}: {m['count']}" for m in ctx["last_6_months"]))
    if ctx.get("forecast_next_q50") is not None:
        lines.append(
            f"Forecast q50: {ctx['forecast_next_q50']}, "
            f"q10: {ctx.get('forecast_q10','N/A')}, q90: {ctx.get('forecast_q90','N/A')}, "
            f"delta: {ctx.get('forecast_delta','N/A')} ({ctx.get('forecast_delta_pct','N/A')}%)"
        )
    if ctx.get("forecast_series"):
        lines.append("Forecast series: " + ", ".join(
            f"{m['target_year_month']}: {m['predicted_count']:.1f}"
            for m in ctx["forecast_series"]))
    return "\n".join(f"• {l}" for l in lines)


_SYS = (
    "You are a senior technology intelligence analyst with 20+ years of experience in patent "
    "landscape analysis, R&D strategy, and emerging technology forecasting. You advise R&D Directors, "
    "VC firms, and Corporate Strategy teams."
)

_RULES = """
## Rules
- Output ONLY the 4 bullets — no intro, no conclusion, no headers
- Never open a bullet by describing the data — open with the strategic conclusion
- Never use "The data shows", "Analysis reveals", "The term registered"
- Every bullet must cite at least 2 specific numbers
- Each bullet: 3–5 sentences — conclusion → evidence → implication
- No markdown bold or headers inside bullets
- Do NOT number the bullets — just start each with a dash (-)
"""

_BAD_GOOD = """
BAD: "The term registered 518 patents over 24 months. The trajectory shows decline..."
GOOD: "This technology is entering post-peak consolidation — monthly activity collapsed 46% from its peak of 34 in 2024-03 to 15 by 2025-11, while the 6-month average fell from 28 to 17 (-39%). R&D teams should pivot toward narrow improvement claims before the filing window closes."
"""


def prompt_trend_single(ctx):
    return f"""{_SYS}

Analyze patent filing data for: "{ctx.get('term', '')}"

## BRIEFING
{_briefing(ctx)}

## DATA
{json.dumps(ctx, indent=2)}

## CRITICAL: Start each bullet with the conclusion, not the data.
{_BAD_GOOD}

## 4 BULLETS — one per angle (start each with -):
- Lifecycle phase & dominant trajectory (use annual trend + CAGR)
- Single most important anomaly, inflection, or structural shift (cite month + value)
- Recent momentum (last 3–6 months) + forecast interpretation if available
- Concrete strategic recommendation — where to file, invest, or avoid
{_RULES}"""


def prompt_trend_comparison(ctx):
    terms_str = ", ".join(f'"{t}"' for t in ctx["terms_compared"])
    briefings = "\n\n".join(
        f"### {t}\n{_briefing(d)}" for t, d in ctx.get("data", {}).items())
    return f"""{_SYS}

Compare patent trends for: {terms_str}

## BRIEFINGS
{briefings}

## DATA
{json.dumps(ctx, indent=2)}

## CRITICAL: Open each bullet with the comparative conclusion.
BAD: "EV had 320 patents vs hybrid 180, showing EV leads."
GOOD: "EV-specific IP is crowding out hybrid innovation — the 78% volume gap (320 vs 180) signals hybrid is a viable white space precisely because it's underfiled relative to market size."

## 4 BULLETS — each must compare ≥2 terms by name with specific numbers (start each with -):
- Volume leadership and what the gap signals about investment priorities
- Trajectory divergence — which is accelerating, which is stalling, and why
- Peak timing comparison — shared catalyst or independent cycles?
- Portfolio recommendation — increase, harvest, or white-space opportunity?
{_RULES}"""


def prompt_forecast(ctx):
    return f"""{_SYS}

Interpret patent history + TFT forecast for: "{ctx.get('term', '')}"

## BRIEFING
{_briefing(ctx)}

## DATA
{json.dumps(ctx, indent=2)}

## CRITICAL: Interpret what the forecast MEANS, don't restate the numbers.
BAD: "The q50 forecast is 18.3 vs last real value of 21, a delta of -2.7."
GOOD: "The model is calling a soft landing — the -2.7 delta (-12.9%) sits well within the q10–q90 band, suggesting the TFT reads the slowdown as temporary recalibration, not structural exit. Hold positions but delay major new filing programs until the 3-month series confirms direction."

## 4 BULLETS (start each with -):
- Near-term signal: continuation, reversal, or plateau? Act or wait?
- Confidence band (q10–q90): tight = predictable, wide = structural uncertainty — what does it mean for planning?
- Medium-term direction: is the near-term move sustained or transient?
- Optimal filing strategy given forecast direction and uncertainty level
{_RULES}"""


def prompt_indicators(ctx):
    t  = ctx.get("term", "");        s  = ctx.get("future_score", "?")
    g  = ctx.get("growth_pct", "?"); d  = ctx.get("density", "?")
    f  = ctx.get("fusion", "?");     sh = ctx.get("shift_pct", "?")
    return f"""{_SYS}

Technology assessment for: "{t}"

## DATA
{json.dumps(ctx, indent=2)}

Key values: Growth {g}% | Density {d} patents | Fusion {f} co-terms | Shift {sh}% | Score {s}

## CRITICAL: Interpret what each value MEANS, don't just state it.
BAD: "Growth is -12%, indicating decline."
GOOD: "The -12% growth places this at post-peak consolidation — foundational claims are crowded but improvement patents stay viable for 18–24 more months, making this the last clean window for differentiated filing."

## 4 BULLETS (start each with -):
- Growth {g}%: what lifecycle phase does this signal, and what action does that phase imply?
- Density {d} + Fusion {f} together: platform technology, niche, or commoditized space?
- Semantic shift {sh}%: stable domain or mutating into new contexts — strategic implication?
- Future Score {s} verdict: buy / hold / avoid — justify using the full indicator combination
{_RULES}"""


def prompt_correlation(ctx):
    term = ctx.get("base_term", "")
    return f"""{_SYS}

Analyze correlation patterns for: "{term}"

## DATA
{json.dumps(ctx, indent=2)}

## CRITICAL: Interpret what correlations reveal about strategy, not just who is correlated.
BAD: "'battery management' has lift 3.2, indicating strong co-occurrence."
GOOD: "'{term}' and 'battery management' are functionally inseparable in current IP — a lift of 3.2 means 3x more co-filing than chance predicts, revealing innovators treat them as a single integrated system. Filing on one without the other leaves exploitable claim gaps."

## 4 BULLETS — name specific terms with actual metric values (start each with -):
- Strongest co-occurrence partner (name + lift/jaccard): what does this pairing reveal about how "{term}" is applied?
- Top co-occurring cluster collectively: what integrated system or domain do they define?
- Temporal synchrony (cite Pearson r) + any anticorrelations signaling substitution dynamics
- Best 1–2 adjacency targets for IP portfolio expansion — and what claim type to pursue
{_RULES}"""


def prompt_opportunities(ctx):
    term  = ctx.get("anchor_term", "")
    total = ctx.get("total_found", "?")
    return f"""{_SYS}

Identify white-space opportunities for: "{term}" ({total} found)

These terms NEVER co-occurred with "{term}" but share strong mutual neighbors — structural gaps that precede breakthrough cross-domain patents.

## DATA
{json.dumps(ctx, indent=2)}

## CRITICAL: Provide a specific innovation hypothesis, not just a label.
BAD: "'solid-state electrolytes' has bridge strength 847, representing an opportunity."
GOOD: "Bridge 847 for 'solid-state electrolytes' means shared neighbors through battery architecture intermediaries — yet no patent has combined them with '{term}'. First-filer would occupy a position neither community currently defends."

## 4 BULLETS — name terms with bridge scores (start each with -):
- Top opportunity (name + score): specific innovation hypothesis — what would you file and why hasn't anyone yet?
- Top 5–7 collectively: what emerging application domain or capability gap do they point toward?
- Best first-mover candidate: low density + high bridge + plausible 2–3yr commercial path
- Specific claim scope (method/system/composition) most defensible if filed in next 6–12 months
{_RULES}"""


def call_gemini(prompt: str) -> str:
    """
    Chama o Gemini via llm.py (client centralizado, retry e tratamento
    de erro já embutidos ali). Mantém a assinatura antiga (recebe só o
    prompt e devolve uma string) para não exigir mudanças nos callbacks.
    """
    result = generate(prompt, temperature=0.4, max_output_tokens=8192)
    return str(result)


def render_analysis(text: str, label: str = "AI Analysis"):
    """
    Renderiza bullets do Gemini como cards visuais.
    FIX: remove numeração (ex: '1.', '2.') além de símbolos de lista.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    def _clean(line: str) -> str:
        # remove prefixos: "- ", "* ", "• ", "1. ", "2. ", etc.
        line = re.sub(r"^[\-\*•]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        return line.strip()

    bullets = [
        html.Div(_clean(line), style={
            "padding":      "12px 16px",
            "marginBottom": "8px",
            "background":   "rgba(37,99,235,0.06)",
            "borderLeft":   "2px solid rgba(37,99,235,0.45)",
            "borderRadius": "0 10px 10px 0",
            "fontSize":     "13.5px",
            "lineHeight":   "1.8",
            "color":        "#d1d5db",
        })
        for line in lines if line.strip()
    ]

    return html.Div([
        html.Div([
            html.I(className="fas fa-robot me-2",
                   style={"color": BLUE, "fontSize": "12px"}),
            html.Span(label.upper(), style={
                "fontSize":     "11px",
                "color":        "#4b5563",
                "fontWeight":   "600",
                "letterSpacing":"1px",
            }),
        ], style={
            "marginBottom": "10px",
            "marginTop":    "16px",
            "display":      "flex",
            "alignItems":   "center",
        }),
        *bullets,
    ])