from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from core.config import load_configurations
from core.model_loader import load_all_models
from core.prediction_engine import predict_one

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}
MODEL_DISPLAY_NAMES = {value: key for key, value in MODEL_OPTIONS.items()}

st.set_page_config(
    page_title="STAR CI-Selected Disease Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading and validating the saved research models...")
def get_loaded_models() -> dict[str, Any]:
    # Reuse the repository's audited loader. It verifies artifact hashes,
    # feature ordering, disease labels, and pinned sklearn/XGBoost versions.
    return load_all_models(PROJECT_ROOT, enforce_versions=True)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
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
            max-width: 1450px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        label, legend, small, summary, h1, h2, h3, h4 {
            color: #111111 !important;
        }
        h1, h2, h3 { color: #000000 !important; letter-spacing: -.02em; }
        [data-testid="stSidebar"] {
            background: #f2f2f2 !important;
            border-right: 1px solid #777777 !important;
        }
        [data-testid="stSidebar"] * { color: #000000 !important; }
        .hero {
            background: #000000;
            color: #ffffff;
            padding: 1.7rem 1.9rem;
            border: 1px solid #000000;
            border-radius: 12px;
            margin-bottom: 1.2rem;
        }
        .hero * { color: #ffffff !important; }
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
        .hero-subtitle { font-size: 1rem; max-width: 900px; margin: 0; }
        .hero-badges { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1rem; }
        .hero-badge {
            background: #222222;
            border: 1px solid #ffffff;
            border-radius: 5px;
            padding: .35rem .7rem;
            font-size: .78rem;
            font-weight: 650;
        }
        .section-kicker {
            color: #000000;
            font-weight: 800;
            font-size: .72rem;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin-bottom: -.35rem;
        }
        .soft-card {
            background: #ffffff;
            border: 1px solid #555555;
            border-radius: 8px;
            padding: 1rem 1.1rem;
        }
        .mini-label {
            color: #333333;
            font-size: .76rem;
            font-weight: 750;
            letter-spacing: .07em;
            text-transform: uppercase;
        }
        .mini-value { color: #000000; font-size: 1.48rem; font-weight: 800; margin-top: .18rem; }
        .mini-note { color: #444444; font-size: .82rem; margin-top: .12rem; }
        .selected-wrap { display: flex; gap: .4rem; flex-wrap: wrap; margin: .55rem 0 .15rem; }
        .selected-chip {
            display: inline-block;
            background: #ffffff;
            border: 1px solid #333333;
            color: #000000;
            border-radius: 5px;
            padding: .28rem .65rem;
            font-size: .78rem;
            font-weight: 650;
        }
        .prediction-rank {
            color: #333333;
            font-size: .8rem;
            font-weight: 750;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .top-disease { color: #000000; font-size: 1.65rem; font-weight: 850; margin: .2rem 0 .4rem; }
        div[data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] > div {
            background: #ffffff !important;
            border-color: #777777 !important;
            color: #000000 !important;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #777777;
            border-radius: 8px;
            padding: .75rem .85rem;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #000000 !important; }
        input, textarea, [data-baseweb="select"] > div,
        [data-testid="stTextInput"] input {
            background: #ffffff !important;
            color: #000000 !important;
            border-color: #555555 !important;
        }
        [data-baseweb="popover"], [data-baseweb="menu"],
        [role="listbox"], [role="option"] {
            background: #ffffff !important;
            color: #000000 !important;
        }
        [role="option"] * { color: #000000 !important; }
        input[type="checkbox"], input[type="radio"] { accent-color: #000000 !important; }
        div[data-testid="stButton"] button {
            background: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #000000 !important;
            border-radius: 6px !important;
            font-weight: 700 !important;
            min-height: 2.65rem;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            background: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #000000 !important;
            font-weight: 800 !important;
        }
        div[data-testid="stButton"] button[kind="primary"] * { color: #ffffff !important; }
        [data-testid="stAlert"] {
            background: #f2f2f2 !important;
            border: 1px solid #777777 !important;
            color: #000000 !important;
        }
        [data-testid="stAlert"] * { color: #000000 !important; }
        [data-testid="stExpander"] {
            border: 1px solid #777777 !important;
            border-radius: 7px;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary span { color: #000000 !important; }
        .footer-note {
            color: #333333;
            border-top: 1px solid #777777;
            padding-top: 1rem;
            font-size: .78rem;
        }
        a, a:visited { color: #000000 !important; }
        hr { border-color: #999999 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_star_items(configurations: dict[str, Any]) -> list[dict[str, str]]:
    """Return exactly the configured STAR feature vocabulary, with display labels."""
    star_config = configurations["star_ci_features"]
    master_config = configurations["master_symptoms"]
    feature_names = star_config["feature_names"]
    declared_count = star_config["feature_count"]

    if len(feature_names) != declared_count or len(feature_names) != len(set(feature_names)):
        raise ValueError("STAR CI feature configuration has an invalid count or duplicate identifiers.")

    master_by_feature = {
        item["feature_name"]: item
        for item in master_config.get("symptoms", [])
    }
    missing = [name for name in feature_names if name not in master_by_feature]
    if missing:
        raise ValueError(f"STAR CI features missing from master symptom configuration: {missing}")

    items: list[dict[str, str]] = []
    seen_labels: set[str] = set()
    for feature_name in feature_names:
        source_item = master_by_feature[feature_name]
        display_name = source_item.get("display_name") or feature_name.replace("_", " ").title()
        label = display_name
        if label.casefold() in seen_labels:
            label = f"{display_name} [{feature_name}]"
        seen_labels.add(label.casefold())
        items.append({"feature_name": feature_name, "display_name": label})
    return sorted(items, key=lambda item: (item["display_name"].casefold(), item["feature_name"]))


def render_metric_card(label: str, value: str, note: str) -> None:
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


def render_selected_chips(selected: list[str], items: list[dict[str, str]], limit: int = 30) -> None:
    selected_set = set(selected)
    labels = [item["display_name"] for item in items if item["feature_name"] in selected_set]
    if not labels:
        st.caption("No symptoms selected yet.")
        return
    chips = "".join(f'<span class="selected-chip">{escape(label)}</span>' for label in labels[:limit])
    remaining = len(labels) - min(len(labels), limit)
    if remaining:
        chips += f'<span class="selected-chip">+{remaining} more</span>'
    st.markdown(f'<div class="selected-wrap">{chips}</div>', unsafe_allow_html=True)


def render_prediction(result: dict[str, Any], model_label: str) -> None:
    predictions = result["predictions"]
    top = predictions[0]
    with st.container(border=True):
        st.markdown('<div class="section-kicker">STAR CI-selected · Prediction output</div>', unsafe_allow_html=True)
        st.subheader("Top-three disease predictions")
        st.caption(
            f"Classifier: **{model_label}** · STAR CI features: **{result['model_feature_count']}** · "
            f"Selected features used: **{result['selected_model_feature_count']}**"
        )
        st.markdown('<div class="prediction-rank">Top-ranked prediction</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="top-disease">{escape(str(top["disease"]))}</div>', unsafe_allow_html=True)
        st.metric("Model-estimated score", f"{float(top['model_score_percent']):.2f}%")
        st.markdown("**Top three predictions**")
        for prediction in predictions:
            rank = int(prediction["rank"])
            disease = escape(str(prediction["disease"]))
            score = float(prediction["model_score"])
            percent = float(prediction["model_score_percent"])
            st.markdown(f"**{rank}. {disease}** · {percent:.2f}%")
            st.progress(min(max(score, 0.0), 1.0))

        st.caption(result["score_note"])
        export_rows = [
            {
                "Rank": int(row["rank"]),
                "Disease": row["disease"],
                "Model score (%)": float(row["model_score_percent"]),
            }
            for row in predictions
        ]
        import pandas as pd

        export = pd.DataFrame(export_rows)
        st.download_button(
            "Download top-three predictions (CSV)",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="star_ci_top_three_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    inject_custom_css()
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Causal feature selection · Machine learning</div>
          <div class="hero-title">STAR CI-Selected Disease Prediction</div>
          <p class="hero-subtitle">
            A standalone prediction interface using only the 106 symptoms retained by the
            STAR CI-selected feature-screening experiment and its saved machine-learning models.
          </p>
          <div class="hero-badges">
            <span class="hero-badge">106 CI-selected symptoms</span>
            <span class="hero-badge">3 trained classifiers</span>
            <span class="hero-badge">41 disease classes</span>
            <span class="hero-badge">Top-three predictions</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "Research-use prototype only. It is not a medical diagnostic system and must not be "
        "used to make healthcare or treatment decisions. Displayed percentages are model-estimated "
        "predict_proba scores, not calibrated clinical probabilities."
    )

    try:
        configurations = load_configurations(PROJECT_ROOT)
        models = get_loaded_models()
        symptom_items = display_star_items(configurations)
    except Exception as exc:
        st.error("The STAR CI prediction resources could not be loaded. No prediction was generated.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")
        st.info("The app uses the repository's pinned requirements and saved model artifacts.")
        return

    star_feature_names = configurations["star_ci_features"]["feature_names"]
    star_feature_set = set(star_feature_names)
    if len(symptom_items) != 106:
        st.error(f"Expected 106 STAR CI-selected symptoms, but loaded {len(symptom_items)}.")
        return

    with st.sidebar:
        st.markdown("## Prediction settings")
        model_label = st.selectbox(
            "STAR CI classifier",
            options=list(MODEL_OPTIONS),
            index=0,
            help="Select one of the three saved STAR CI-selected classifiers.",
        )
        st.divider()
        st.markdown("**STAR CI feature set**")
        st.metric("CI-selected symptoms", len(star_feature_names))
        st.metric("Disease classes", len(configurations["disease_labels"]["classes"]))
        st.caption("The selector contains only the symptoms retained by the STAR CI experiment.")
        st.divider()
        st.markdown("**Score interpretation**")
        st.caption(
            "Scores come from the selected model's predict_proba output. They are not "
            "calibrated clinical probabilities or diagnostic certainty."
        )

    st.markdown('<div class="section-kicker">Step 1 · Patient-reported input</div>', unsafe_allow_html=True)
    st.header("Select STAR CI-selected symptoms")
    st.write(
        "Choose symptoms from the STAR CI-selected feature set only. Checked symptoms are encoded "
        "as 1 and unchecked symptoms as 0, matching the original binary input representation."
    )

    search_query = st.text_input(
        "Search the STAR CI symptom list",
        placeholder="Search by symptom name (e.g. fever, fatigue, headache)...",
        help="Only the 106 STAR CI-selected symptoms are searchable and selectable.",
        key="star_only_search_query",
    ).strip().casefold()
    visible_items = [
        item for item in symptom_items
        if not search_query
        or search_query in item["display_name"].casefold()
        or search_query in item["feature_name"].casefold()
    ]
    visible_features = [item["feature_name"] for item in visible_items]

    action_left, action_mid, action_right = st.columns([1.2, 1.2, 3.0])
    with action_left:
        select_visible = st.button(
            "Select search results",
            use_container_width=True,
            disabled=not visible_items,
            key="star_only_select_visible",
        )
    with action_mid:
        clear_all = st.button(
            "Clear selection",
            use_container_width=True,
            key="star_only_clear_all",
        )
    with action_right:
        st.caption(f"Showing **{len(visible_items)}** of **{len(symptom_items)}** STAR CI symptoms")

    selected_set = set(st.session_state.get("star_only_selected_symptoms", set())) & star_feature_set
    if select_visible:
        selected_set.update(visible_features)
        for feature_name in visible_features:
            st.session_state[f"star_only_symptom__{feature_name}"] = True
    elif clear_all:
        selected_set.clear()
        for item in symptom_items:
            st.session_state[f"star_only_symptom__{item['feature_name']}"] = False
    st.session_state["star_only_selected_symptoms"] = selected_set

    selected_feature_options = [
        item["feature_name"] for item in symptom_items
        if item["feature_name"] in selected_set
    ]
    if selected_feature_options:
        with st.expander(f"Manage selected symptoms ({len(selected_feature_options)})"):
            display_by_feature = {item["feature_name"]: item["display_name"] for item in symptom_items}
            remove_feature = st.selectbox(
                "Choose one selected symptom to remove",
                options=selected_feature_options,
                format_func=lambda feature: display_by_feature[feature],
                key="star_only_remove_choice",
            )
            if st.button("Remove selected symptom", key="star_only_remove_button"):
                selected_set.discard(remove_feature)
                st.session_state["star_only_selected_symptoms"] = selected_set
                st.session_state[f"star_only_symptom__{remove_feature}"] = False

    with st.container(border=True):
        if visible_items:
            checkbox_columns = st.columns(3)
            for index, item in enumerate(visible_items):
                feature_name = item["feature_name"]
                widget_key = f"star_only_symptom__{feature_name}"
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = feature_name in selected_set
                with checkbox_columns[index % len(checkbox_columns)]:
                    checked = st.checkbox(
                        item["display_name"],
                        key=widget_key,
                        help=f"Exact STAR CI model feature identifier: {feature_name}",
                    )
                if checked:
                    selected_set.add(feature_name)
                else:
                    selected_set.discard(feature_name)
        else:
            st.info("No STAR CI-selected symptoms match that search. Try another term.")

    selected_symptoms = [
        item["feature_name"] for item in symptom_items
        if item["feature_name"] in selected_set
    ]
    st.session_state["star_only_selected_symptoms"] = set(selected_symptoms)

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        render_metric_card(
            "Symptoms selected",
            f"{len(selected_symptoms)} / {len(symptom_items)}",
            "All selectable symptoms belong to the STAR CI feature set",
        )
    with metric_col2:
        render_metric_card(
            "Model input coverage",
            f"{len(set(selected_symptoms) & star_feature_set)} used",
            "Selected STAR CI features included in the prediction input",
        )

    with st.expander("Review selected symptoms", expanded=bool(selected_symptoms)):
        render_selected_chips(selected_symptoms, symptom_items)

    if not selected_symptoms:
        st.info("Select at least one STAR CI symptom to enable prediction.")

    st.markdown('<div class="section-kicker">Step 2 · Model inference</div>', unsafe_allow_html=True)
    st.header("Generate STAR CI prediction")
    st.write(
        "The selected symptoms are passed only to the STAR CI-selected model. "
        "No Stable-FCI prediction, model averaging, or retraining is performed."
    )
    predict_clicked = st.button(
        "Run STAR CI prediction",
        type="primary",
        use_container_width=True,
        disabled=not selected_symptoms,
        key="star_only_run_prediction",
    )
    if predict_clicked:
        with st.spinner("Running the selected saved STAR CI classifier..."):
            try:
                result = predict_one(
                    models,
                    selected_symptoms,
                    approach="star_ci",
                    model_name=MODEL_OPTIONS[model_label],
                    top_k=3,
                )
                st.session_state["star_only_last_prediction"] = {
                    "result": result,
                    "model_label": model_label,
                    "selected_symptoms": selected_symptoms.copy(),
                }
            except Exception as exc:
                st.error("Prediction failed. No new result was saved.")
                with st.expander("Technical details"):
                    st.code(f"{type(exc).__name__}: {exc}")

    last_run = st.session_state.get("star_only_last_prediction")
    if last_run:
        st.markdown('<div class="section-kicker">Step 3 · Results</div>', unsafe_allow_html=True)
        st.header("STAR CI prediction results")
        st.caption(
            f"Last run · Classifier: {last_run['model_label']} · "
            f"{len(last_run['selected_symptoms'])} selected STAR CI symptoms"
        )
        with st.expander("Review the exact input used for this prediction"):
            render_selected_chips(last_run["selected_symptoms"], symptom_items)
            st.caption("Every selected input belongs to the STAR CI-selected feature set.")
        render_prediction(last_run["result"], last_run["model_label"])
        st.caption(
            "Changing symptoms or the classifier does not silently rerun the model. "
            "Click the prediction button again to generate a new result."
        )

    st.divider()
    st.markdown("### About this standalone application")
    st.write(
        "This application uses the saved STAR CI-selected feature list (106 symptoms) and one of "
        "the three saved STAR CI classifiers: Logistic Regression, Random Forest, or XGBoost. "
        "It does not rerun CI screening, retrain a model, or combine predictions from different approaches."
    )
    st.markdown(
        '<div class="footer-note">STAR CI-Selected Disease Prediction · Research prototype only · '
        'Saved STAR CI models · No model retraining or clinical decision support.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
