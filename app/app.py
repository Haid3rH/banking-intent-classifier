import html
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Banking Intent Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    :root{--bg:#07090f;--surface:#0e1320;--surface-2:#131a2b;--border:#1d2840;
          --text:#e8edf5;--muted:#8392a8;--accent:#3b82f6;--accent-hover:#2f6fe0;}
    html,body,.stApp,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
    .stApp{background:var(--bg);color:var(--text);}
    [data-testid="stHeader"]{background:transparent;}
    #MainMenu,footer{visibility:hidden;}
    .block-container{padding-top:3rem;padding-bottom:4rem;max-width:1200px;}
    h1,h2,h3,h4{color:var(--text);letter-spacing:-.01em;}
    hr{border-color:var(--border);}

    [data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--border);}
    [data-testid="stSidebar"] *{color:var(--text);}
    .side-title{font-size:1.15rem;font-weight:600;margin-bottom:.25rem;}
    .side-muted{color:var(--muted)!important;font-size:.9rem;line-height:1.5;}
    .side-row{display:flex;justify-content:space-between;gap:1rem;padding:.55rem 0;
              border-bottom:1px solid var(--border);font-size:.9rem;}
    .side-row span:first-child{color:var(--muted)!important;}

    .page-title{font-size:2.6rem;font-weight:700;letter-spacing:-.02em;margin:0 0 .4rem;line-height:1.15;}
    .page-sub{font-size:1.05rem;color:var(--muted);max-width:650px;margin-bottom:1.8rem;line-height:1.55;}
    .section-title{font-size:1rem;font-weight:600;margin:0 0 .75rem;}

    .stTabs [data-baseweb="tab-list"]{gap:.4rem;border-bottom:1px solid var(--border);}
    .stTabs [data-baseweb="tab"]{color:var(--muted);font-weight:500;padding:.6rem 1rem;}
    .stTabs [aria-selected="true"]{color:var(--text)!important;}
    .stTabs [data-baseweb="tab-highlight"]{background:var(--accent);}
    .stTabs [data-baseweb="tab-panel"]{padding-top:1.8rem;}

    [data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);
        border-radius:10px;padding:1rem 1.2rem;}
    [data-testid="stMetricLabel"] *{color:var(--muted)!important;}
    [data-testid="stMetricValue"]{color:var(--text);}

    .stTextArea label{display:none;}
    .stTextArea textarea{background:var(--surface)!important;color:var(--text)!important;
        border:1px solid var(--border)!important;border-radius:10px!important;
        font-size:1.02rem!important;line-height:1.55!important;padding:1rem!important;}
    .stTextArea textarea:focus{border-color:var(--accent)!important;box-shadow:0 0 0 1px var(--accent)!important;}
    .stTextArea textarea::placeholder{color:#5b6b82!important;}

    div.stButton>button,div.stDownloadButton>button{width:100%;background:var(--surface);color:var(--text);
        border:1px solid var(--border);border-radius:8px;padding:.55rem 1rem;font-weight:500;
        transition:border-color .15s,background-color .15s;}
    div.stButton>button:hover,div.stDownloadButton>button:hover{border-color:var(--accent);
        background:var(--surface-2);color:var(--text);}
    div.stButton>button:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
    div.stButton>button[kind="primary"]{background:var(--accent);border:1px solid var(--accent);color:#fff;font-weight:600;}
    div.stButton>button[kind="primary"]:hover{background:var(--accent-hover);border-color:var(--accent-hover);}

    .result{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);
            border-radius:10px;padding:1.5rem;}
    .result-label{color:var(--muted);font-size:.9rem;margin-bottom:.35rem;}
    .result-value{font-size:2rem;font-weight:700;letter-spacing:-.01em;line-height:1.2;}
    .result-raw{color:var(--muted);font-size:.85rem;margin-top:.3rem;}
    .result-quote{color:var(--muted);font-size:.95rem;margin-top:1.1rem;padding-top:1rem;
                  border-top:1px solid var(--border);line-height:1.5;}
    .related-title{font-weight:600;margin:1.3rem 0 .7rem;}
    .related-item{display:flex;justify-content:space-between;align-items:center;
                  padding:.55rem 0;border-bottom:1px solid var(--border);}
    .related-name{font-size:.92rem;}
    .related-score{color:var(--muted);font-size:.82rem;}

    .empty{background:var(--surface);border:1px dashed var(--border);border-radius:10px;
           padding:2rem 1.5rem;color:var(--muted);line-height:1.55;}
    .empty strong{color:var(--text);font-weight:600;}
    .hist{padding:.8rem 0;border-bottom:1px solid var(--border);}
    .hist-text{font-size:.95rem;line-height:1.45;margin-bottom:.2rem;}
    .hist-intent{color:var(--accent);font-size:.85rem;font-weight:500;}

    [data-testid="stAlert"]{background:var(--surface);border:1px solid var(--border);
                            color:var(--text);border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
MODEL_NAME = "banking_intent_classifier.joblib"
HERE = Path(__file__).resolve().parent
CANDIDATES = [
    HERE.parent / "model" / MODEL_NAME,
    HERE / "model" / MODEL_NAME,
    HERE / MODEL_NAME,
    Path.cwd() / "model" / MODEL_NAME,
    Path.cwd() / MODEL_NAME,
]
EXAMPLES = [
    "My card has not arrived yet",
    "I was charged twice for one payment",
    "How do I change my PIN?",
]


@st.cache_resource
def load_model():
    for path in CANDIDATES:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(
        "Could not find " + MODEL_NAME + ". Looked in:\n" + "\n".join(map(str, CANDIDATES))
    )


try:
    model = load_model()
except Exception as exc:
    st.error(f"The model could not be loaded.\n\n{exc}")
    st.stop()

# ------------------------------------------------------------
# State and helpers
# ------------------------------------------------------------
for key, default in [("history", []), ("latest", None), ("user_text", ""), ("warn", False)]:
    st.session_state.setdefault(key, default)


def pretty(label) -> str:
    text = str(label).replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def top_matches(text, k=3):
    """Top k intents by Linear SVM decision score (not probabilities)."""
    try:
        scores = model.decision_function([text])[0]
        order = scores.argsort()[::-1][:k]
        return [(model.classes_[i], float(scores[i])) for i in order]
    except Exception:
        return []


def save_prediction(text):
    text = text.strip()
    prediction = model.predict([text])[0]
    st.session_state.warn = False
    st.session_state.latest = {"text": text, "prediction": prediction, "matches": top_matches(text)}
    st.session_state.history.insert(
        0,
        {
            "text": text,
            "prediction": prediction,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def classify():
    if st.session_state.user_text.strip():
        save_prediction(st.session_state.user_text)
    else:
        st.session_state.warn = True


def run_example(text):
    st.session_state.user_text = text
    save_prediction(text)


def clear_history():
    st.session_state.update(history=[], latest=None, user_text="", warn=False)


def metrics(items):
    for col, (label, value) in zip(st.columns(len(items)), items):
        col.metric(label, value)


def show_table(df, pct=(), dec=()):
    df = df.copy()
    for c in pct:
        df[c] = df[c].map("{:.2%}".format)
    for c in dec:
        df[c] = df[c].map("{:.3f}".format)
    st.dataframe(df, hide_index=True)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    n_classes = len(getattr(model, "classes_", []))
    st.markdown(
        f"""
        <div class="side-title">Banking NLP</div>
        <div class="side-muted">Classifies a customer message into the banking request it relates to.</div>
        <br>
        <div class="side-row"><span>Approach</span><span>Traditional ML</span></div>
        <div class="side-row"><span>Features</span><span>Word + Character TF-IDF</span></div>
        <div class="side-row"><span>Classifier</span><span>Linear SVM</span></div>
        <div class="side-row"><span>Intents</span><span>{n_classes or "n/a"}</span></div>
        <br>
        """,
        unsafe_allow_html=True,
    )
    st.button("Clear history", on_click=clear_history, disabled=not st.session_state.history)

# ------------------------------------------------------------
# Header + tabs
# ------------------------------------------------------------
st.markdown(
    '<div class="page-title">Banking Intent Classifier</div>'
    '<div class="page-sub">Paste a customer message and see what the customer is trying to do.</div>',
    unsafe_allow_html=True,
)
tab_cls, tab_hist, tab_perf, tab_err, tab_about = st.tabs(
    ["Classifier", "History", "Model performance", "Error analysis", "About"]
)

# ============================================================
# Classifier
# ============================================================
with tab_cls:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">Customer message</div>', unsafe_allow_html=True)
        st.text_area(
            "Customer message",
            key="user_text",
            placeholder="Example: My card has not arrived yet",
            height=170,
        )
        st.button("Classify message", type="primary", on_click=classify)
        if st.session_state.warn:
            st.warning("Enter a message above, then select Classify message.")

        st.markdown(
            '<div class="section-title" style="margin-top:1.6rem;">Try an example</div>',
            unsafe_allow_html=True,
        )
        for i, example in enumerate(EXAMPLES):
            st.button(example, key=f"example_{i}", on_click=run_example, args=(example,))

    with right:
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
        latest = st.session_state.latest

        if latest is None:
            st.markdown(
                '<div class="empty"><strong>No result yet.</strong><br>'
                "Enter a message or choose an example to classify it.</div>",
                unsafe_allow_html=True,
            )
        else:
            related = ""
            if latest["matches"]:
                rows = "".join(
                    f'<div class="related-item"><span class="related-name">{html.escape(pretty(lbl))}</span>'
                    f'<span class="related-score">Decision score: {score:.3f}</span></div>'
                    for lbl, score in latest["matches"]
                )
                related = f'<div class="related-title">Top related intents</div>{rows}'
            st.markdown(
                f"""
                <div class="result">
                    <div class="result-label">Predicted intent</div>
                    <div class="result-value">{html.escape(pretty(latest["prediction"]))}</div>
                    <div class="result-raw">Raw label: {html.escape(str(latest["prediction"]))}</div>
                    {related}
                    <div class="result-quote">&ldquo;{html.escape(latest["text"])}&rdquo;</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.session_state.history:
            st.markdown(
                '<div class="section-title" style="margin-top:2rem;">Recent messages</div>',
                unsafe_allow_html=True,
            )
            for item in st.session_state.history[:5]:
                st.markdown(
                    f'<div class="hist"><div class="hist-text">{html.escape(item["text"])}</div>'
                    f'<div class="hist-intent">{html.escape(pretty(item["prediction"]))}</div></div>',
                    unsafe_allow_html=True,
                )

# ============================================================
# History
# ============================================================
with tab_hist:
    history = st.session_state.history
    metrics([
        ("Predictions this session", len(history)),
        ("Unique intents", len({h["prediction"] for h in history})),
        ("Unique messages", len({h["text"] for h in history})),
    ])
    st.write("")
    if not history:
        st.markdown(
            '<div class="empty"><strong>No predictions yet.</strong><br>'
            "Classify a message and it will appear here for this session.</div>",
            unsafe_allow_html=True,
        )
    else:
        hist_df = pd.DataFrame(
            [
                {
                    "#": i,
                    "Time": h["timestamp"],
                    "Customer message": h["text"],
                    "Predicted intent": pretty(h["prediction"]),
                    "Raw label": h["prediction"],
                }
                for i, h in enumerate(history, 1)
            ]
        )
        c1, c2 = st.columns(2)
        c1.download_button(
            "Download history as CSV",
            data=hist_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="banking_intent_prediction_history.csv",
            mime="text/csv",
        )
        c2.button("Clear prediction history", key="clear_tab", on_click=clear_history)
        st.dataframe(hist_df, hide_index=True)
        st.caption("History is temporary session data and is not saved to a database.")

# ============================================================
# Model performance
# ============================================================
with tab_perf:
    VAL = {"Accuracy": 0.9171, "Macro F1": 0.9181, "Weighted F1": 0.9171}
    TEST = {"Accuracy": 0.9081, "Macro F1": 0.9083, "Weighted F1": 0.9083}

    metrics([
        ("Test accuracy", f'{TEST["Accuracy"]:.2%}'),
        ("Test Macro F1", f'{TEST["Macro F1"]:.2%}'),
        ("Validation Macro F1", f'{VAL["Macro F1"]:.2%}'),
    ])
    st.caption("Final model: Word + Character TF-IDF with Linear SVM. Test metrics were calculated after model selection.")

    st.subheader("Seven-model baseline comparison")
    st.write("Baselines used word TF-IDF and the same train/validation split. Macro F1 was the primary metric.")
    base = pd.DataFrame(
        [
            ("Linear SVM", .8901, .8912, .8896, 2.146, .008),
            ("SGD Classifier", .8811, .8824, .8802, 5.667, .019),
            ("Logistic Regression", .8591, .8563, .8580, 21.926, .013),
            ("Ridge Classifier", .8551, .8573, .8542, 14.787, .013),
            ("Multinomial Naive Bayes", .8252, .7970, .8195, .652, .015),
            ("Complement Naive Bayes", .8042, .7965, .7996, .503, .019),
            ("k-Nearest Neighbors", .7423, .7386, .7423, .107, 1.996),
        ],
        columns=["Model", "Accuracy", "Macro F1", "Weighted F1", "Train time (s)", "Predict time (s)"],
    ).sort_values("Macro F1", ascending=False)
    st.bar_chart(base.set_index("Model")[["Macro F1"]], horizontal=True, color="#3b82f6")
    show_table(base, pct=["Accuracy", "Macro F1", "Weighted F1"], dec=["Train time (s)", "Predict time (s)"])

    st.subheader("Feature representation experiments")
    feat = pd.DataFrame(
        [
            ("Word TF-IDF + SVM", .8901, .8912),
            ("Character TF-IDF + SVM", .9131, .9147),
            ("Word + Character TF-IDF + SVM", .9171, .9181),
            ("Tuned combined pipeline", .9171, .9181),
        ],
        columns=["Experiment", "Accuracy", "Macro F1"],
    )
    st.bar_chart(feat.set_index("Experiment")[["Macro F1"]], horizontal=True, color="#3b82f6")
    show_table(feat, pct=["Accuracy", "Macro F1"])

    st.subheader("Validation versus final test")
    comp = pd.DataFrame({"Metric": list(VAL), "Validation": list(VAL.values()), "Test": list(TEST.values())})
    show_table(comp, pct=["Validation", "Test"])
    st.info("The test set was kept out of model selection and used for the final evaluation only.")

# ============================================================
# Error analysis
# ============================================================
with tab_err:
    metrics([("Validation samples", "1,001"), ("Validation accuracy", "91.71%"), ("Validation Macro F1", "91.81%")])

    st.subheader("Ten lowest per-intent F1 scores")
    st.write("Macro F1 weights every intent equally. Support is shown to put each score in context.")
    weak = pd.DataFrame(
        [
            ("topping_up_by_card", .778, .700, .737, 10),
            ("card_acceptance", .714, .833, .769, 6),
            ("top_up_reverted", .750, .800, .774, 15),
            ("top_up_failed", .846, .733, .786, 15),
            ("compromised_card", .692, 1.000, .818, 9),
            ("receiving_money", .875, .778, .824, 9),
            ("balance_not_updated_after_bank_transfer", .824, .824, .824, 17),
            ("supported_cards_and_currencies", .750, .923, .828, 13),
            ("pending_card_payment", .923, .750, .828, 16),
            ("transfer_into_account", .769, .909, .833, 11),
        ],
        columns=["Intent", "Precision", "Recall", "F1 score", "Support"],
    ).sort_values("F1 score")
    st.bar_chart(weak.set_index("Intent")[["F1 score"]], horizontal=True, color="#3b82f6")
    show_table(weak, dec=["Precision", "Recall", "F1 score"])
    st.caption("With small support, one or two predictions can substantially move a class score.")

    st.subheader("Most frequent confusion pairs")
    st.write("Counts from the combined model's validation confusion matrix, not test-set results.")
    conf = pd.DataFrame(
        [
            ("balance_not_updated_after_bank_transfer", "pending_transfer", 2),
            ("direct_debit_payment_not_recognised", "card_payment_not_recognised", 2),
            ("topping_up_by_card", "top_up_reverted", 2),
            ("why_verify_identity", "unable_to_verify_identity", 2),
            ("top_up_failed", "pending_top_up", 2),
            ("country_support", "card_acceptance", 2),
            ("beneficiary_not_allowed", "pending_card_payment", 1),
            ("beneficiary_not_allowed", "pin_blocked", 1),
            ("receiving_money", "topping_up_by_card", 1),
            ("card_not_working", "order_physical_card", 1),
            ("card_acceptance", "card_not_working", 1),
            ("pending_cash_withdrawal", "cash_withdrawal_charge", 1),
            ("pending_cash_withdrawal", "cash_withdrawal_not_recognised", 1),
            ("beneficiary_not_allowed", "transfer_into_account", 1),
            ("pending_cash_withdrawal", "declined_cash_withdrawal", 1),
        ],
        columns=["True intent", "Predicted intent", "Count"],
    )
    st.dataframe(conf, hide_index=True)

    st.subheader("Why these intents overlap")
    st.markdown(
        """
- **`balance_not_updated_after_bank_transfer` vs `pending_transfer`:** both can mention a transfer that has not appeared yet.
- **`direct_debit_payment_not_recognised` vs `card_payment_not_recognised`:** both describe unfamiliar transactions; only the payment method differs.
- **`top_up_failed` vs `pending_top_up`:** a failed top-up and one still processing can share wording.
- **`country_support` vs `card_acceptance`:** questions about using a card in a country resemble acceptance questions.
- **`why_verify_identity` vs `unable_to_verify_identity`:** asking why verification is needed differs from being unable to complete it, but the wording overlaps.
"""
    )
    st.info("Most remaining mistakes involve semantically similar intents.")

# ============================================================
# About
# ============================================================
with tab_about:
    st.subheader("The project")
    st.write(
        "A traditional machine-learning NLP system that maps a short banking message to one of 77 intent "
        "classes. For example, a message about a card that has not arrived maps to `card_arrival`."
    )
    metrics([("Intent classes", "77"), ("Training examples", "10,003"), ("Held-out test examples", "3,080")])

    st.subheader("How the pipeline works")
    st.markdown(
        """
1. **Customer message:** the user enters a natural-language banking request.
2. **Word TF-IDF:** words and two-word phrases, `ngram_range=(1, 2)`.
3. **Character TF-IDF:** character sequences of length 3 to 5, which capture subword patterns and spelling variation.
4. **Combined features:** `FeatureUnion` joins the word and character representations.
5. **Linear SVM:** `LinearSVC(C=1)` predicts one of the 77 intent labels.
6. **Result:** the app shows the predicted intent and ranks related intents by SVM decision score. These scores are not probabilities.
"""
    )

    st.subheader("Development process")
    st.markdown(
        """
- Inspected missing values, duplicates, intent distribution, and message length.
- Created a stratified 90/10 train/validation split.
- Compared seven baselines, using Macro F1 as the main metric alongside accuracy and weighted F1.
- Checked train-versus-validation gaps for overfitting.
- Tested word-only, character-only, and combined TF-IDF features.
- Tuned inside a pipeline so TF-IDF was fitted separately within each CV fold.
- Retrained the selected pipeline on all 10,003 labeled examples before the final test evaluation.
"""
    )
    st.write(
        "Final configuration: word n-grams (1, 2), character n-grams (3, 5), `min_df=2`, "
        "`sublinear_tf=True`, Linear SVM with `C=1`."
    )

    st.subheader("Limitations")
    st.markdown(
        """
- Semantically similar intents can still be confused.
- Classes with few validation examples have less stable metrics.
- Related-intent scores are rankings, not calibrated probabilities.
- The classifier predicts a label only; it does not answer the customer or resolve the issue.
- Prediction history is temporary session data.
"""
    )

    st.divider()
    st.subheader("About me")
    st.write(
        "Haid3rH. This portfolio project covers data exploration, model comparison, word and character "
        "TF-IDF feature engineering, validation-based model selection, held-out testing, and a Streamlit interface."
    )
    st.link_button("Visit GitHub profile", "https://github.com/Haid3rH")
    st.caption("A portfolio project for intent classification. It is not a banking service; do not enter real account data.")
