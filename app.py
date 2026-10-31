from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.config import load_configurations
from core.model_loader import load_all_models
from core.prediction_engine import predict_comparison
from core.prediction_analytics import analyze_prediction_agreement
from core.result_comparison import compare_top_predictions


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}
MODEL_DISPLAY_NAMES = {value: key for key, value in MODEL_OPTIONS.items()}

st.set_page_config(
    page_title="Comparative Disease Prediction | Research Prototype",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading and validating the six saved models...")
def get_loaded_models() -> dict[str, Any]:
    """Load the six trusted, hash-verified models once per Streamlit server."""
    return load_all_models(PROJECT_ROOT, enforce_versions=True)


def inject_custom_css() -> None:
    """Apply a high-contrast black-and-white theme without changing app logic."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #000000;
            --body: #111111;
            --muted: #444444;
            --line: #777777;
            --panel: #ffffff;
            --canvas: #ffffff;
            --soft: #f2f2f2;
            --black: #000000;
            --white: #ffffff;
        }

        /* Overall page and typography */
        html, body, [data-testid="stApp"], .stApp,
        [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background: #ffffff !important;
            color: #000000 !important;
        }
        [data-testid="stHeader"] {
            background: #ffffff !important;
            border-bottom: 1px solid #dddddd;
        }
        [data-testid="stMainBlockContainer"] {
            padding-top: 1.25rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stText"],
        label, legend, small, summary {
            color: #111111 !important;
        }
        h1, h2, h3, h4, h5, h6,
        [data-testid="stHeader"] h1,
        [data-testid="stHeader"] h2 {
            color: #000000 !important;
            letter-spacing: -0.02em;
        }
        h1 { font-weight: 800 !important; }
        h2 { font-weight: 750 !important; }
        h3 { font-weight: 700 !important; }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        .stCaption {
            color: #444444 !important;
        }

        /* Header banner: black with white text */
        .hero {
            background: #000000 !important;
            color: #ffffff !important;
            padding: 1.75rem 1.9rem;
            border: 1px solid #000000;
            border-radius: 12px;
            margin-bottom: 1.1rem;
            box-shadow: none;
        }
        .hero-kicker, .hero-title, .hero-subtitle, .hero-badge {
            color: #ffffff !important;
        }
        .hero-kicker {
            font-size: .76rem;
            font-weight: 750;
            letter-spacing: .14em;
            text-transform: uppercase;
        }
        .hero-title {
            font-size: 2rem;
            line-height: 1.12;
            font-weight: 800;
            margin: .45rem 0 .55rem;
        }
        .hero-subtitle {
            font-size: 1rem;
            max-width: 850px;
            margin: 0;
        }
        .hero-badges { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.1rem; }
        .hero-badge {
            background: #222222 !important;
            border: 1px solid #ffffff !important;
            border-radius: 5px;
            padding: .35rem .7rem;
            font-size: .78rem;
            font-weight: 650;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #f2f2f2 !important;
            border-right: 1px solid #777777 !important;
        }
        [data-testid="stSidebar"] * { color: #000000 !important; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #222222 !important;
        }

        /* Section labels, cards, tags and prediction text */
        .section-kicker {
            color: #000000 !important;
            font-weight: 800;
            font-size: .72rem;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin-bottom: -.35rem;
        }
        .soft-card {
            background: #ffffff !important;
            border: 1px solid #555555 !important;
            border-radius: 8px;
            padding: 1rem 1.1rem;
        }
        .mini-label {
            color: #333333 !important;
            font-size: .76rem;
            font-weight: 750;
            letter-spacing: .07em;
            text-transform: uppercase;
        }
        .mini-value {
            color: #000000 !important;
            font-size: 1.48rem;
            font-weight: 800;
            margin-top: .18rem;
        }
        .mini-note { color: #444444 !important; font-size: .82rem; margin-top: .12rem; }
        .selected-wrap { display: flex; gap: .4rem; flex-wrap: wrap; margin: .55rem 0 .15rem; }
        .selected-chip {
            display: inline-block;
            background: #ffffff !important;
            border: 1px solid #333333 !important;
            color: #000000 !important;
            border-radius: 5px;
            padding: .28rem .65rem;
            font-size: .78rem;
            font-weight: 650;
        }
        .approach-tag {
            display: inline-block;
            color: #ffffff !important;
            background: #000000 !important;
            border: 1px solid #000000 !important;
            border-radius: 4px;
            padding: .22rem .55rem;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .04em;
        }
        .prediction-rank {
            color: #000000 !important;
            font-weight: 850;
            font-size: .78rem;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .top-disease {
            color: #000000 !important;
            font-size: 1.48rem;
            font-weight: 820;
            line-height: 1.2;
            margin: .25rem 0 .65rem;
        }
        .muted-note { color: #444444 !important; font-size: .82rem; }
        .comparison-callout {
            background: #f2f2f2 !important;
            border: 1px solid #555555 !important;
            border-radius: 8px;
            padding: .9rem 1rem;
            color: #000000 !important;
        }
        .analytics-rank-grid {
            display: grid;
            grid-template-columns: minmax(145px, 1.2fr) minmax(150px, 1fr) minmax(150px, 1fr);
            gap: .55rem .8rem;
            align-items: center;
            background: #ffffff;
            border: 1px solid #777777;
            border-radius: 8px;
            padding: .9rem;
            margin: .4rem 0 1rem;
        }
        .analytics-grid-head {
            color: #222222 !important;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .07em;
            text-transform: uppercase;
            padding-bottom: .4rem;
            border-bottom: 1px solid #999999;
        }
        .analytics-disease {
            color: #000000 !important;
            font-weight: 700;
            overflow-wrap: anywhere;
        }
        .analytics-rank-cell { display: flex; align-items: center; gap: .5rem; min-width: 0; }
        .analytics-rank-track {
            flex: 1;
            min-width: 38px;
            height: 13px;
            background: #e5e5e5;
            border: 1px solid #b0b0b0;
            border-radius: 3px;
            overflow: hidden;
        }
        .analytics-rank-fill { height: 100%; }
        .analytics-rank-fill.stable { background: #000000; }
        .analytics-rank-fill.star { background: #888888; }
        .analytics-rank-label {
            color: #000000 !important;
            font-size: .78rem;
            font-weight: 750;
            white-space: nowrap;
            min-width: 49px;
        }
        .analytics-not-listed {
            color: #555555 !important;
            font-size: .78rem;
            font-style: italic;
        }
        .analytics-legend { display: flex; flex-wrap: wrap; gap: 1rem; margin: .4rem 0 .8rem; }
        .analytics-legend-item { color: #222222 !important; font-size: .78rem; display: flex; align-items: center; gap: .4rem; }
        .analytics-legend-swatch { display: inline-block; width: 18px; height: 9px; border: 1px solid #555555; }
        .analytics-legend-swatch.stable { background: #000000; }
        .analytics-legend-swatch.star { background: #888888; }
        @media (max-width: 700px) {
            .analytics-rank-grid {
                grid-template-columns: minmax(95px, .9fr) minmax(110px, 1fr) minmax(110px, 1fr);
                gap: .45rem;
                padding: .55rem;
            }
            .analytics-grid-head { font-size: .62rem; }
            .analytics-rank-label { min-width: 42px; font-size: .7rem; }
        }
        .footer-note {
            color: #333333 !important;
            border-top: 1px solid #777777;
            padding-top: 1rem;
            font-size: .78rem;
        }

        /* Streamlit metric cards and bordered containers */
        div[data-testid="stMetric"],
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background: #ffffff !important;
            border-color: #777777 !important;
            color: #000000 !important;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #777777 !important;
            border-radius: 8px;
            padding: .75rem .85rem;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: #000000 !important;
        }

        /* Inputs, checkboxes and dropdowns */
        input, textarea,
        [data-baseweb="select"] > div,
        [data-testid="stTextInput"] input {
            background: #ffffff !important;
            color: #000000 !important;
            border-color: #555555 !important;
        }
        input::placeholder, textarea::placeholder {
            color: #555555 !important;
            opacity: 1 !important;
        }
        input[type="checkbox"], input[type="radio"] { accent-color: #000000 !important; }
        [data-baseweb="popover"], [data-baseweb="menu"],
        [role="listbox"], [role="option"] {
            background: #ffffff !important;
            color: #000000 !important;
        }
        [role="option"] * { color: #000000 !important; }

        /* Buttons: black primary, white secondary, both high contrast */
        div[data-testid="stButton"] button {
            background: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #000000 !important;
            border-radius: 6px !important;
            font-weight: 700 !important;
            min-height: 2.65rem;
        }
        div[data-testid="stButton"] button:hover {
            background: #e8e8e8 !important;
            color: #000000 !important;
            border-color: #000000 !important;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            background: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #000000 !important;
            border-radius: 6px !important;
            font-weight: 800 !important;
            min-height: 2.8rem;
        }
        div[data-testid="stButton"] button[kind="primary"] * { color: #ffffff !important; }
        div[data-testid="stButton"] button:disabled {
            background: #e5e5e5 !important;
            color: #555555 !important;
            border-color: #999999 !important;
        }

        /* Alerts, progress bars, expanders and tables */
        [data-testid="stAlert"] {
            background: #f2f2f2 !important;
            border: 1px solid #777777 !important;
            color: #000000 !important;
        }
        [data-testid="stAlert"] * { color: #000000 !important; }
        [data-testid="stProgress"] > div { background: #e5e5e5 !important; }
        [data-testid="stProgress"] > div > div { background: #000000 !important; }
        [data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid #777777 !important;
            border-radius: 7px;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary span { color: #000000 !important; }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background: #ffffff !important;
            color: #000000 !important;
        }
        [data-testid="stDataFrame"] * { color: #000000 !important; }
        hr { border-color: #999999 !important; }
        a, a:visited { color: #000000 !important; text-decoration: underline; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _display_symptom_items(master_config: dict[str, Any]) -> list[dict[str, str]]:
    """Create alphabetically ordered display items while preserving exact feature IDs."""
    items = master_config.get("symptoms", [])
    result: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for item in items:
        feature_name = item["feature_name"]
        display_name = item.get("display_name") or feature_name.replace("_", " ").title()
        # The internal identifier remains untouched; disambiguate only visible labels.
        label = display_name
        if label.casefold() in seen_names:
            label = f"{display_name} [{feature_name}]"
        seen_names.add(label.casefold())
        result.append({"feature_name": feature_name, "display_name": label})
    return sorted(result, key=lambda item: (item["display_name"].casefold(), item["feature_name"]))


def _render_metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="soft-card">
          <div class="mini-label">{escape(label)}</div>
          <div class="mini-value">{escape(value)}</div>
          <div class="mini-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_selected_chips(
    selected_symptoms: list[str], master_items: list[dict[str, str]], *, limit: int = 28
) -> None:
    selected_set = set(selected_symptoms)
    names = [
        item["display_name"]
        for item in master_items
        if item["feature_name"] in selected_set
    ]
    if not names:
        st.caption("No symptoms selected yet.")
        return
    chips = "".join(
        f'<span class="selected-chip">{escape(name)}</span>'
        for name in names[:limit]
    )
    remaining = len(names) - min(len(names), limit)
    if remaining:
        chips += f'<span class="selected-chip">+{remaining} more</span>'
    st.markdown(f'<div class="selected-wrap">{chips}</div>', unsafe_allow_html=True)


def _render_prediction_panel(title: str, result: dict[str, Any]) -> None:
    model_label = MODEL_DISPLAY_NAMES[result["model"]]
    top_prediction = result["predictions"][0]
    with st.container(border=True):
        st.markdown(f'<span class="approach-tag">{escape(title.upper())}</span>', unsafe_allow_html=True)
        st.markdown(f"### {escape(title)}")
        st.caption(
            f"Classifier: **{model_label}** · Feature set: **{result['model_feature_count']}** · "
            f"Selected features used: **{result['selected_model_feature_count']}**"
        )
        if result["ignored_selected_symptom_count"]:
            st.info(
                f"{result['ignored_selected_symptom_count']} checked symptom(s) are outside "
                "this approach's selected feature set and are not used by this model."
            )
        st.markdown('<div class="prediction-rank">Top-ranked prediction</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="top-disease">{escape(top_prediction["disease"])}</div>', unsafe_allow_html=True)
        st.metric("Model-estimated score", f"{top_prediction['model_score_percent']:.2f}%")
        st.markdown("**Top three predictions**")
        for prediction in result["predictions"]:
            rank = int(prediction["rank"])
            name = escape(str(prediction["disease"]))
            score = float(prediction["model_score"])
            percent = float(prediction["model_score_percent"])
            st.markdown(f"**{rank}. {name}** · {percent:.2f}%")
            st.progress(min(max(score, 0.0), 1.0))
        st.caption(
            f"{result['selected_model_feature_count']} of {result['selected_symptom_count']} "
            "checked symptom(s) belong to this model's selected feature set."
        )


def _rank_indicator(rank: int | None, approach: str) -> str:
    """Render an ordinal rank bar; this visual does not encode probability."""
    if rank is None:
        return '<span class="analytics-not-listed">Not in top three</span>'
    width = {1: 100, 2: 67, 3: 34}.get(int(rank), 15)
    css_class = "stable" if approach == "stable" else "star"
    return (
        '<div class="analytics-rank-cell">'
        f'<div class="analytics-rank-track"><div class="analytics-rank-fill {css_class}" '
        f'style="width:{width}%"></div></div>'
        f'<span class="analytics-rank-label">Rank {int(rank)}</span>'
        '</div>'
    )


def _render_comparison(stable_result: dict[str, Any], star_result: dict[str, Any]) -> None:
    """Render descriptive top-three overlap, rank alignment, and exportable outputs."""
    comparison = compare_top_predictions(
        stable_result["predictions"], star_result["predictions"]
    )
    analytics = analyze_prediction_agreement(
        stable_result["predictions"], star_result["predictions"]
    )

    st.markdown('<div class="section-kicker">Cross-approach analysis</div>', unsafe_allow_html=True)
    st.subheader("Prediction comparison and analytics")
    st.caption(
        "These measures describe the two selected classifiers for this one symptom input. "
        "They are not test-set performance metrics."
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        _render_metric_card(
            "Top-ranked agreement",
            "Yes" if analytics["top_prediction_agrees"] else "No",
            "Same disease ranked first by both approaches",
        )
    with metric_col2:
        _render_metric_card(
            "Top-three overlap",
            f"{analytics['shared_count']} / {analytics['union_count']}",
            f"Jaccard overlap: {analytics['jaccard_overlap_percent']:.1f}%",
        )
    with metric_col3:
        _render_metric_card(
            "Same-rank matches",
            f"{analytics['exact_rank_matches']} / {analytics['rank_positions_compared']}",
            "Same disease at the same list position",
        )
    with metric_col4:
        mean_shift = analytics["mean_absolute_rank_shift"]
        _render_metric_card(
            "Mean rank shift",
            f"{mean_shift:.2f}" if mean_shift is not None else "Not available",
            "Average absolute rank difference for shared diseases",
        )

    if analytics["top_prediction_agrees"]:
        st.success(
            f"Both approaches rank **{escape(str(analytics['stable_top_disease']))}** first. "
            "This is agreement between the selected classifiers for this input only."
        )
    else:
        st.info(
            "The first-ranked predictions differ: "
            f"Stable-FCI: **{escape(str(analytics['stable_top_disease'] or 'Unavailable'))}** · "
            f"STAR CI-selected: **{escape(str(analytics['star_top_disease'] or 'Unavailable'))}**."
        )

    overview_tab, rank_tab, export_tab = st.tabs(
        ["Overlap overview", "Rank alignment", "Detailed outputs & export"]
    )

    with overview_tab:
        shared_col, stable_only_col, star_only_col = st.columns(3)
        with shared_col:
            st.markdown("**Predicted by both**")
            st.write(analytics["shared_diseases"] or "No shared top-three diseases")
        with stable_only_col:
            st.markdown("**Stable-FCI only**")
            st.write(analytics["stable_only_diseases"] or "None")
        with star_only_col:
            st.markdown("**STAR CI-selected only**")
            st.write(analytics["star_only_diseases"] or "None")

        if analytics["union_count"]:
            st.markdown("**Top-three set overlap**")
            st.progress(analytics["jaccard_overlap"])
            st.caption(
                f"Jaccard overlap = shared diseases ÷ unique diseases across both lists "
                f"= {analytics['shared_count']} ÷ {analytics['union_count']}."
            )

    with rank_tab:
        st.markdown("**Where each disease appears in the two rankings**")
        st.markdown(
            '<div class="analytics-legend">'
            '<span class="analytics-legend-item"><span class="analytics-legend-swatch stable"></span>Stable-FCI</span>'
            '<span class="analytics-legend-item"><span class="analytics-legend-swatch star"></span>STAR CI-selected</span>'
            '<span class="analytics-legend-item">Longer bar = closer to rank 1; it is not a probability score.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        rows_by_disease = {row["disease"]: row for row in comparison["rows"]}
        html_rows = [
            '<div class="analytics-rank-grid">',
            '<div class="analytics-grid-head">Disease</div>',
            '<div class="analytics-grid-head">Stable-FCI position</div>',
            '<div class="analytics-grid-head">STAR CI position</div>',
        ]
        for disease, row in rows_by_disease.items():
            html_rows.extend(
                [
                    f'<div class="analytics-disease">{escape(str(disease))}</div>',
                    f'<div>{_rank_indicator(row["stable_fci_rank"], "stable")}</div>',
                    f'<div>{_rank_indicator(row["star_ci_rank"], "star")}</div>',
                ]
            )
        html_rows.append('</div>')
        st.markdown("".join(html_rows), unsafe_allow_html=True)

        rank_table = pd.DataFrame(analytics["rank_rows"])
        if not rank_table.empty:
            rank_table = rank_table.rename(
                columns={
                    "rank": "List position",
                    "stable_fci_disease": "Stable-FCI disease",
                    "star_ci_disease": "STAR CI disease",
                    "same_disease_at_rank": "Same disease at this position",
                }
            )
            st.markdown("**Rank-by-rank check**")
            st.dataframe(
                rank_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Same disease at this position": st.column_config.CheckboxColumn(),
                },
            )

        if analytics["shared_disease_rank_rows"]:
            shared_rank_table = pd.DataFrame(analytics["shared_disease_rank_rows"])
            shared_rank_table = shared_rank_table.rename(
                columns={
                    "disease": "Shared disease",
                    "stable_fci_rank": "Stable-FCI rank",
                    "star_ci_rank": "STAR CI rank",
                    "star_minus_stable_rank": "STAR minus Stable rank",
                    "absolute_rank_shift": "Absolute rank shift",
                }
            )
            st.markdown("**Rank movement for shared diseases**")
            st.dataframe(shared_rank_table, use_container_width=True, hide_index=True)
        else:
            st.info("There are no shared diseases, so rank movement cannot be calculated.")

    with export_tab:
        output_table = pd.DataFrame(comparison["rows"])
        if not output_table.empty:
            rank_shift_lookup = {
                row["disease"]: row["star_minus_stable_rank"]
                for row in analytics["shared_disease_rank_rows"]
            }
            output_table["star_minus_stable_rank"] = output_table["disease"].map(
                rank_shift_lookup
            )
            output_table = output_table.rename(
                columns={
                    "disease": "Disease",
                    "stable_fci_rank": "Stable-FCI rank",
                    "stable_fci_score_percent": "Stable-FCI model score (%)",
                    "star_ci_rank": "STAR CI rank",
                    "star_ci_score_percent": "STAR CI model score (%)",
                    "present_in_both_top_k": "Present in both top-three lists",
                    "star_minus_stable_rank": "STAR minus Stable rank",
                }
            )
            st.dataframe(
                output_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Stable-FCI model score (%)": st.column_config.NumberColumn(format="%.2f"),
                    "STAR CI model score (%)": st.column_config.NumberColumn(format="%.2f"),
                    "Present in both top-three lists": st.column_config.CheckboxColumn(),
                },
            )
            st.download_button(
                "Download comparison table (CSV)",
                data=output_table.to_csv(index=False).encode("utf-8"),
                file_name="stable_fci_star_ci_prediction_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.caption(
            f"Stable-FCI classifier: {MODEL_DISPLAY_NAMES[stable_result['model']]} · "
            f"STAR CI-selected classifier: {MODEL_DISPLAY_NAMES[star_result['model']]}"
        )

    st.caption(
        "Model scores are displayed separately and are not averaged. The overlap and rank "
        "statistics describe this single prediction only; they do not establish model accuracy, "
        "clinical reliability, causal validity, or calibrated probability."
    )


def _render_methodology(configurations: dict[str, Any]) -> None:
    stable_count = configurations["stable_fci_features"]["feature_count"]
    star_count = configurations["star_ci_features"]["feature_count"]
    total_symptoms = configurations["master_symptoms"]["feature_count"]
    disease_count = len(configurations["disease_labels"]["classes"])

    st.markdown('<div class="section-kicker">About this research prototype</div>', unsafe_allow_html=True)
    st.subheader("How the two prediction pipelines work")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("### Stable-FCI")
            st.markdown(f"**{stable_count} selected symptom features**")
            st.write(
                "The Stable-FCI experiment's saved feature set is applied to the common "
                "symptom input. The selected Logistic Regression, Random Forest, or "
                "XGBoost classifier then produces its own disease scores."
            )
    with right:
        with st.container(border=True):
            st.markdown("### STAR CI-selected")
            st.markdown(f"**{star_count} selected symptom features**")
            st.write(
                "The STAR CI-selected experiment's saved feature set is applied independently "
                "to the same common symptom input. Its selected classifier produces a separate "
                "set of disease scores."
            )

    st.markdown("**Shared application configuration**")
    col1, col2, col3 = st.columns(3)
    with col1:
        _render_metric_card("Input symptom vocabulary", str(total_symptoms), "Original symptom identifiers")
    with col2:
        _render_metric_card("Disease classes", str(disease_count), "Shared disease label set")
    with col3:
        _render_metric_card("Saved classifiers", "6", "Three classifiers per approach")

    st.info(
        "The two experiments retain their original training and testing protocols. "
        "This application deploys their saved models; it does not rerun Stable-FCI or STAR CI, "
        "retrain classifiers, merge feature sets, or ensemble predictions."
    )
    st.warning(
        "Research use only. This prototype is not a medical diagnostic system and must not "
        "be used to make healthcare or treatment decisions. The percentages are model-estimated "
        "predict_proba scores; they have not been established as calibrated clinical probabilities."
    )


def main() -> None:
    inject_custom_css()
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Causal feature selection · Machine learning</div>
          <div class="hero-title">Comparative Disease Prediction</div>
          <p class="hero-subtitle">One symptom input. Two independently trained research pipelines. Compare the predictions and ranking patterns produced by two independently trained research pipelines.</p>
          <div class="hero-badges">
            <span class="hero-badge">132 symptom inputs</span>
            <span class="hero-badge">2 research approaches</span>
            <span class="hero-badge">6 trained classifiers</span>
            <span class="hero-badge">41 disease classes</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        configurations = load_configurations(PROJECT_ROOT)
        models = get_loaded_models()
    except Exception as exc:
        st.error("The prediction models could not be loaded. No prediction was generated.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")
        st.info(
            "Activate the project's virtual environment and install the pinned requirements. "
            "The saved models require scikit-learn 1.9.0 and XGBoost 3.1.3."
        )
        return

    master_items = _display_symptom_items(configurations["master_symptoms"])
    stable_features = set(configurations["stable_fci_features"]["feature_names"])
    star_features = set(configurations["star_ci_features"]["feature_names"])

    with st.sidebar:
        st.markdown("## Prediction settings")
        st.caption("Choose each classifier independently.")
        stable_label = st.selectbox(
            "Stable-FCI classifier",
            options=list(MODEL_OPTIONS),
            index=1,
            help="Uses the saved Stable-FCI feature set and trained classifier.",
        )
        star_label = st.selectbox(
            "STAR CI-selected classifier",
            options=list(MODEL_OPTIONS),
            index=0,
            help="Uses the saved STAR CI-selected feature set and trained classifier.",
        )
        st.divider()
        st.markdown("**Experiment feature sets**")
        st.metric("Stable-FCI features", len(stable_features))
        st.metric("STAR CI-selected features", len(star_features))
        st.caption("Feature identifiers and model column ordering are preserved from the saved experiments.")
        st.divider()
        st.markdown("**Score interpretation**")
        st.caption(
            "Displayed percentages are classifier predict_proba outputs. They are not calibrated "
            "clinical probabilities or diagnostic certainty."
        )

    # ------------------------------- Input ---------------------------------
    st.markdown('<div class="section-kicker">Step 1 · Patient-reported input</div>', unsafe_allow_html=True)
    st.header("Select symptoms")
    st.write(
        "Tick the symptoms to include in the input. A checked symptom is encoded as 1; "
        "an unchecked symptom is encoded as 0, matching the original binary dataset format."
    )

    search_query = st.text_input(
        "Search the symptom list",
        placeholder="Search by symptom name (e.g. fever, fatigue, headache)...",
        help="Search uses the display name and exact internal feature identifier.",
    ).strip().casefold()
    visible_items = [
        item for item in master_items
        if not search_query
        or search_query in item["display_name"].casefold()
        or search_query in item["feature_name"].casefold()
    ]
    visible_features = [item["feature_name"] for item in visible_items]

    action_left, action_mid, action_right = st.columns([1.15, 1.15, 3.2])
    with action_left:
        select_visible = st.button(
            "Select search results", use_container_width=True, disabled=not visible_items
        )
    with action_mid:
        clear_all = st.button("Clear selection", use_container_width=True)
    with action_right:
        st.caption(f"Showing **{len(visible_items)}** of **{len(master_items)}** symptoms")

    selected_set = set(st.session_state.get("selected_symptoms", set()))
    if select_visible:
        selected_set.update(visible_features)
        st.session_state.selected_symptoms = selected_set
        for feature_name in visible_features:
            st.session_state[f"symptom__{feature_name}"] = True
    elif clear_all:
        selected_set.clear()
        st.session_state.selected_symptoms = set()
        for item in master_items:
            st.session_state[f"symptom__{item['feature_name']}"] = False

    # Let the user remove an item directly from the review area. This runs before
    # the symptom checkboxes are instantiated in this script run, so widget state
    # can be updated safely without changing an already-rendered checkbox.
    selected_feature_options = [
        item["feature_name"] for item in master_items
        if item["feature_name"] in selected_set
    ]
    if selected_feature_options:
        with st.expander(f"Manage selected symptoms ({len(selected_feature_options)})"):
            display_by_feature = {
                item["feature_name"]: item["display_name"] for item in master_items
            }
            remove_feature = st.selectbox(
                "Choose one selected symptom to remove",
                options=selected_feature_options,
                format_func=lambda feature: display_by_feature[feature],
                key="remove_symptom_choice",
            )
            if st.button("Remove selected symptom", key="remove_symptom_button"):
                selected_set.discard(remove_feature)
                st.session_state.selected_symptoms = selected_set
                st.session_state[f"symptom__{remove_feature}"] = False

    with st.container(border=True):
        if visible_items:
            checkbox_columns = st.columns(3)
            for index, item in enumerate(visible_items):
                feature_name = item["feature_name"]
                widget_key = f"symptom__{feature_name}"
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = feature_name in selected_set
                with checkbox_columns[index % len(checkbox_columns)]:
                    checked = st.checkbox(
                        item["display_name"],
                        key=widget_key,
                        help=f"Exact model feature identifier: {feature_name}",
                    )
                if checked:
                    selected_set.add(feature_name)
                else:
                    selected_set.discard(feature_name)
                st.session_state.selected_symptoms = selected_set
        else:
            st.info("No symptoms match that search. Try a different term.")

    selected_symptoms = [
        item["feature_name"] for item in master_items
        if item["feature_name"] in set(st.session_state.get("selected_symptoms", set()))
    ]
    selected_set = set(selected_symptoms)

    count_col, stable_cov_col, star_cov_col = st.columns(3)
    with count_col:
        _render_metric_card("Symptoms selected", f"{len(selected_symptoms)} / {len(master_items)}", "Shared input for both approaches")
    with stable_cov_col:
        used = len(selected_set & stable_features)
        ignored = len(selected_set - stable_features)
        _render_metric_card("Stable-FCI input coverage", f"{used} used", f"{ignored} checked symptom(s) outside its feature set")
    with star_cov_col:
        used = len(selected_set & star_features)
        ignored = len(selected_set - star_features)
        _render_metric_card("STAR CI input coverage", f"{used} used", f"{ignored} checked symptom(s) outside its feature set")

    with st.expander("Review selected symptoms", expanded=bool(selected_symptoms)):
        _render_selected_chips(selected_symptoms, master_items)
        if selected_symptoms:
            stable_ignored = [name for name in selected_symptoms if name not in stable_features]
            star_ignored = [name for name in selected_symptoms if name not in star_features]
            detail_left, detail_right = st.columns(2)
            with detail_left:
                st.markdown("**Checked symptoms not used by Stable-FCI**")
                st.write(stable_ignored or "None")
            with detail_right:
                st.markdown("**Checked symptoms not used by STAR CI-selected**")
                st.write(star_ignored or "None")

    if not selected_symptoms:
        st.info("Select at least one symptom to enable prediction.")

    # ----------------------------- Prediction ------------------------------
    st.markdown('<div class="section-kicker">Step 2 · Independent inference</div>', unsafe_allow_html=True)
    st.header("Generate predictions")
    st.write(
        "The same selected symptom list is passed to both pipelines. Each pipeline independently "
        "projects it into its own saved feature set and column order."
    )
    predict_clicked = st.button(
        "Run both prediction pipelines",
        type="primary",
        use_container_width=True,
        disabled=not selected_symptoms,
    )
    if predict_clicked:
        with st.spinner("Running the selected saved classifiers..."):
            try:
                result = predict_comparison(
                    models,
                    selected_symptoms,
                    stable_model=MODEL_OPTIONS[stable_label],
                    star_model=MODEL_OPTIONS[star_label],
                    top_k=3,
                )
                st.session_state.last_prediction = {
                    "result": result,
                    "stable_model_label": stable_label,
                    "star_model_label": star_label,
                    "selected_symptoms": selected_symptoms.copy(),
                }
            except Exception as exc:
                st.error("Prediction failed. No new result was saved.")
                with st.expander("Technical details"):
                    st.code(f"{type(exc).__name__}: {exc}")

    last_run = st.session_state.get("last_prediction")
    if last_run:
        result = last_run["result"]
        st.markdown('<div class="section-kicker">Step 3 · Results</div>', unsafe_allow_html=True)
        st.header("Prediction results")
        st.caption(
            f"Last run · Stable-FCI: {last_run['stable_model_label']} · "
            f"STAR CI-selected: {last_run['star_model_label']} · "
            f"{len(last_run['selected_symptoms'])} selected symptoms"
        )
        with st.expander("Review the exact input used for this prediction"):
            _render_selected_chips(last_run["selected_symptoms"], master_items)
            st.caption(
                "Both approaches received this same symptom selection. Each used only the "
                "features retained by its own experiment."
            )

        stable_column, star_column = st.columns(2, gap="large")
        with stable_column:
            _render_prediction_panel("Stable-FCI", result["stable_fci"])
        with star_column:
            _render_prediction_panel("STAR CI-selected", result["star_ci"])
        _render_comparison(result["stable_fci"], result["star_ci"])
        st.caption(
            "Changing the symptom selection or classifier settings does not silently rerun the "
            "models. Click the prediction button again to generate a new result."
        )

    st.divider()
    _render_methodology(configurations)

    st.markdown(
        '<div class="footer-note">Comparative Disease Prediction Research Prototype · '
        'Saved research models only · No model retraining, causal discovery rerun, prediction '
        'averaging, or clinical decision support.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
