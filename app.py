from pathlib import Path
import json
import urllib.parse

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
IMAGE_DIR = BASE_DIR / "images"

# UIT-inspired palette from UIT brand page snippet
UIT_BLUE = "#2F6BFF"      # RGB(47,107,255)
UIT_DEEP_BLUE = "#0000FD" # RGB(0,0,253)
UIT_NAVY = "#102A56"
UIT_TEXT = "#16315F"
UIT_TEXT_SOFT = "#56719E"
UIT_BG = "#F5F9FF"
UIT_CARD = "#FFFFFF"
UIT_BORDER = "#DCE7FF"
UIT_SURFACE = "#EEF4FF"
UIT_SUCCESS = "#11875D"
UIT_DANGER = "#D64545"

st.set_page_config(
    page_title="IE105 Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@st.cache_resource
def load_artifacts():
    lr_model = joblib.load(ARTIFACT_DIR / "logistic_regression.pkl")
    iso_model = joblib.load(ARTIFACT_DIR / "isolation_forest.pkl")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")

    thresholds_path_json = ARTIFACT_DIR / "thresholds.json"
    thresholds_path_pkl = ARTIFACT_DIR / "thresholds.pkl"
    if thresholds_path_json.exists():
        with open(thresholds_path_json, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
    else:
        thresholds = joblib.load(thresholds_path_pkl)

    eval_details = joblib.load(ARTIFACT_DIR / "evaluation_details.pkl")
    summary = pd.read_csv(ARTIFACT_DIR / "summary_metrics.csv")

    ae_model = Autoencoder(int(thresholds["input_dim"]))
    state_dict = torch.load(ARTIFACT_DIR / "autoencoder_state_dict.pth", map_location="cpu")
    ae_model.load_state_dict(state_dict)
    ae_model.eval()

    return {
        "lr_model": lr_model,
        "iso_model": iso_model,
        "ae_model": ae_model,
        "scaler": scaler,
        "thresholds": thresholds,
        "eval_details": eval_details,
        "summary": summary,
    }


def build_logo_uri() -> str:
    logo_path = IMAGE_DIR / "Logo.svg"
    if logo_path.exists():
        svg = logo_path.read_text(encoding="utf-8")
        return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)
    return ""


def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
              radial-gradient(circle at top right, rgba(47,107,255,0.08), transparent 28%),
              linear-gradient(180deg, {UIT_BG} 0%, #F8FBFF 100%);
            color: {UIT_TEXT};
        }}

        .block-container {{
            max-width: 1380px;
            padding-top: 1.15rem;
            padding-bottom: 2rem;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #F8FBFF 0%, #EEF4FF 100%);
            border-right: 1px solid {UIT_BORDER};
        }}

        html, body, [class*="css"], p, li, label, span, div {{
            color: {UIT_TEXT};
        }}

        h1, h2, h3, h4 {{
            color: {UIT_DEEP_BLUE};
            letter-spacing: -0.02em;
        }}

        .hero {{
            background: linear-gradient(135deg, rgba(47,107,255,0.13), rgba(0,0,253,0.05));
            border: 1px solid rgba(47,107,255,0.14);
            border-radius: 24px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 14px 36px rgba(47,107,255,0.10);
            margin-bottom: 1rem;
        }}

        .hero-grid {{
            display: grid;
            grid-template-columns: 160px 1fr;
            gap: 1rem;
            align-items: center;
        }}

        .hero-title {{
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 800;
            color: {UIT_DEEP_BLUE};
            margin-bottom: 0.25rem;
        }}

        .hero-sub {{
            font-size: 1rem;
            line-height: 1.6;
            color: {UIT_TEXT_SOFT};
            margin-bottom: 0.7rem;
        }}

        .logo-shell {{
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(47,107,255,0.16);
            border-radius: 18px;
            padding: 0.45rem;
        }}

        .pill-wrap {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }}

        .pill {{
            display: inline-block;
            padding: 0.38rem 0.78rem;
            border-radius: 999px;
            font-size: 0.83rem;
            font-weight: 700;
            color: {UIT_DEEP_BLUE};
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(47,107,255,0.18);
        }}

        .section-title {{
            font-size: 1.18rem;
            font-weight: 800;
            color: {UIT_DEEP_BLUE};
            margin: 0.2rem 0 0.75rem 0;
        }}

        .card {{
            background: {UIT_CARD};
            border: 1px solid {UIT_BORDER};
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 10px 28px rgba(32, 70, 158, 0.08);
            margin-bottom: 0.85rem;
        }}

        .kpi-card {{
            background: {UIT_CARD};
            border: 1px solid {UIT_BORDER};
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 10px 28px rgba(32, 70, 158, 0.08);
            height: 100%;
        }}

        .kpi-label {{
            font-size: 0.90rem;
            font-weight: 700;
            color: {UIT_TEXT_SOFT};
            margin-bottom: 0.25rem;
        }}

        .kpi-value {{
            font-size: 1.48rem;
            font-weight: 800;
            color: {UIT_DEEP_BLUE};
            margin-bottom: 0.2rem;
        }}

        .kpi-note {{
            font-size: 0.88rem;
            color: {UIT_TEXT_SOFT};
            line-height: 1.5;
        }}

        .caption-strong {{
            font-size: 1.04rem;
            font-weight: 800;
            color: {UIT_DEEP_BLUE};
            margin-bottom: 0.2rem;
            line-height: 1.35;
        }}

        .caption-soft {{
            font-size: 0.92rem;
            color: {UIT_TEXT_SOFT};
            margin-bottom: 0.72rem;
            line-height: 1.6;
        }}

        .pred-card {{
            background: {UIT_CARD};
            border: 1px solid {UIT_BORDER};
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 10px 28px rgba(32, 70, 158, 0.08);
        }}

        .pred-title {{
            font-size: 1.03rem;
            font-weight: 800;
            color: {UIT_DEEP_BLUE};
            margin-bottom: 0.55rem;
        }}

        .badge-normal, .badge-fraud {{
            display: inline-block;
            padding: 0.28rem 0.68rem;
            border-radius: 999px;
            font-size: 0.81rem;
            font-weight: 800;
            margin-bottom: 0.55rem;
        }}

        .badge-normal {{
            background: rgba(17,135,93,0.12);
            color: {UIT_SUCCESS};
            border: 1px solid rgba(17,135,93,0.22);
        }}

        .badge-fraud {{
            background: rgba(214,69,69,0.12);
            color: {UIT_DANGER};
            border: 1px solid rgba(214,69,69,0.22);
        }}

        .pred-metric {{
            font-size: 0.93rem;
            color: {UIT_TEXT_SOFT};
            margin-bottom: 0.22rem;
        }}

        .pred-highlight {{
            font-size: 1.2rem;
            font-weight: 800;
            color: {UIT_DEEP_BLUE};
        }}

        .small-note {{
            color: {UIT_TEXT_SOFT};
            font-size: 0.87rem;
            line-height: 1.55;
        }}

        .sidebar-note {{
            border: 1px solid {UIT_BORDER};
            border-radius: 16px;
            padding: 0.95rem;
            background: rgba(255,255,255,0.86);
            box-shadow: 0 8px 24px rgba(32, 70, 158, 0.07);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.35rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 46px;
            background: rgba(255,255,255,0.9);
            border: 1px solid {UIT_BORDER};
            border-radius: 14px 14px 0 0;
            padding-left: 1rem;
            padding-right: 1rem;
            font-weight: 700;
            color: {UIT_TEXT};
            font-size: 0.96rem;
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(180deg, #FFFFFF 0%, #F4F8FF 100%);
            color: {UIT_DEEP_BLUE};
            border-bottom-color: transparent;
        }}

        [data-testid="stMetric"], [data-testid="stDataFrame"] {{
            border-radius: 16px;
        }}

        .stButton > button, .stDownloadButton > button {{
            border-radius: 12px;
            border: 1px solid rgba(47,107,255,0.20);
            background: linear-gradient(180deg, {UIT_BLUE} 0%, #2358D9 100%);
            color: white !important;
            font-weight: 700;
            font-size: 0.96rem;
        }}

        .stButton > button:hover, .stDownloadButton > button:hover {{
            background: linear-gradient(180deg, #2A62EE 0%, #1F51CC 100%);
        }}

        .stSelectbox label, .stFileUploader label, .stNumberInput label {{
            font-weight: 700 !important;
            color: {UIT_DEEP_BLUE} !important;
            font-size: 0.95rem !important;
        }}

        .stCaption {{
            color: {UIT_TEXT_SOFT} !important;
            font-size: 0.91rem !important;
        }}

        code {{
            color: {UIT_DEEP_BLUE};
            background: {UIT_SURFACE};
            border-radius: 6px;
            padding: 0.1rem 0.35rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


assets = load_artifacts()
inject_css()

lr_model = assets["lr_model"]
iso_model = assets["iso_model"]
ae_model = assets["ae_model"]
scaler = assets["scaler"]
thresholds = assets["thresholds"]
eval_details = assets["eval_details"]
summary_df = assets["summary"]
feature_names = eval_details["feature_names"]
LOGO_URI = build_logo_uri()


def render_image_with_caption(image_path: Path, title: str, subtitle: str = ""):
    if image_path.exists():
        st.markdown(f"<div class='caption-strong'>{title}</div>", unsafe_allow_html=True)
        if subtitle:
            st.markdown(f"<div class='caption-soft'>{subtitle}</div>", unsafe_allow_html=True)
        st.image(str(image_path), use_container_width=True)


def render_kpi(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-note'>{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(model_name: str, prediction: str, score: float, threshold: float, note: str = ""):
    badge_class = "badge-fraud" if prediction == "Fraud" else "badge-normal"
    st.markdown(
        f"""
        <div class='pred-card'>
            <div class='pred-title'>{model_name}</div>
            <div class='{badge_class}'>{prediction}</div>
            <div class='pred-metric'>Score</div>
            <div class='pred-highlight'>{score:.6f}</div>
            <div class='pred-metric' style='margin-top:0.55rem;'>Threshold: <b>{threshold:.6f}</b></div>
            <div class='small-note' style='margin-top:0.55rem;'>{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_download_button(df, label, filename):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")


def make_template_csv():
    return pd.DataFrame([{col: 0.0 for col in feature_names}]).to_csv(index=False).encode("utf-8")



def prepare_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare the 30 required input features in the exact training order."""
    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError("Thiếu các cột: " + ", ".join(missing))

    X = df[feature_names].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError("Có giá trị không hợp lệ ở các cột: " + ", ".join(bad_cols))

    return X


def get_threshold(model_name: str) -> float:
    """Return model threshold with a safe fallback."""
    value = thresholds.get(model_name, 0.5)
    try:
        return float(value)
    except Exception:
        return 0.5


def score_models(df: pd.DataFrame) -> pd.DataFrame:
    """Run all 3 models on a dataframe and return raw scores plus binary predictions."""
    X = prepare_input_features(df)
    X_scaled = scaler.transform(X)

    # Logistic Regression: higher probability means higher fraud risk.
    lr_score = lr_model.predict_proba(X_scaled)[:, 1]
    lr_thr = get_threshold("Logistic Regression")
    lr_pred = (lr_score >= lr_thr).astype(int)

    # Isolation Forest: score_samples/decision_function usually gives lower values for anomalies,
    # so we multiply by -1 to make higher score mean higher fraud/anomaly risk.
    if_score = -iso_model.score_samples(X_scaled)
    if_thr = get_threshold("Isolation Forest")
    if_pred = (if_score >= if_thr).astype(int)

    # Autoencoder: higher reconstruction error means higher anomaly risk.
    ae_model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        recon = ae_model(x_tensor).detach().cpu().numpy()
    ae_score = np.mean((X_scaled - recon) ** 2, axis=1)
    ae_thr = get_threshold("Autoencoder")
    ae_pred = (ae_score >= ae_thr).astype(int)

    return pd.DataFrame({
        "LR_Score": lr_score,
        "LR_Threshold": lr_thr,
        "LR_Pred": lr_pred,
        "IF_Score": if_score,
        "IF_Threshold": if_thr,
        "IF_Pred": if_pred,
        "AE_Score": ae_score,
        "AE_Threshold": ae_thr,
        "AE_Pred": ae_pred,
    })


def score_one_row(input_df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form prediction table for the manual one-transaction tab."""
    scores = score_models(input_df).iloc[0]

    rows = [
        {
            "Model": "Logistic Regression",
            "Score": float(scores["LR_Score"]),
            "Threshold": float(scores["LR_Threshold"]),
            "Prediction": "Fraud" if int(scores["LR_Pred"]) == 1 else "Normal",
        },
        {
            "Model": "Isolation Forest",
            "Score": float(scores["IF_Score"]),
            "Threshold": float(scores["IF_Threshold"]),
            "Prediction": "Fraud" if int(scores["IF_Pred"]) == 1 else "Normal",
        },
        {
            "Model": "Autoencoder",
            "Score": float(scores["AE_Score"]),
            "Threshold": float(scores["AE_Threshold"]),
            "Prediction": "Fraud" if int(scores["AE_Pred"]) == 1 else "Normal",
        },
    ]
    return pd.DataFrame(rows)


def run_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Run batch prediction for uploaded CSV and append outputs from all 3 models."""
    scores = score_models(df)
    pred_df = df.copy()

    for col in scores.columns:
        pred_df[col] = scores[col].values

    pred_df["LR_Label"] = np.where(pred_df["LR_Pred"] == 1, "Fraud", "Normal")
    pred_df["IF_Label"] = np.where(pred_df["IF_Pred"] == 1, "Fraud", "Normal")
    pred_df["AE_Label"] = np.where(pred_df["AE_Pred"] == 1, "Fraud", "Normal")

    # Convenience summary label based on the best model used in the report.
    pred_df["Best_Model_Label"] = pred_df["LR_Label"]
    pred_df["Best_Model_Score"] = pred_df["LR_Score"]

    score_cols = ["LR_Score", "IF_Score", "AE_Score", "Best_Model_Score"]
    for col in score_cols:
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce").round(6)

    return pred_df


summary_df = summary_df.sort_values(["F1-score", "PR-AUC", "Recall"], ascending=False).reset_index(drop=True)
best_model_row = summary_df.iloc[0]

with st.sidebar:
    if LOGO_URI:
        st.image(LOGO_URI, use_container_width=True)
    st.markdown("**University of Information Technology - UIT**")
    st.markdown("**IE105 Fraud Detection**")
    st.markdown("## Điều hướng")
    if st.button("Tổng quan", use_container_width=True):
        st.session_state["target_tab"] = "Tổng quan"
    if st.button("So sánh mô hình", use_container_width=True):
        st.session_state["target_tab"] = "So sánh mô hình"
    if st.button("Dự đoán 1 giao dịch", use_container_width=True):
        st.session_state["target_tab"] = "Dự đoán 1 giao dịch"
    if st.button("Upload CSV", use_container_width=True):
        st.session_state["target_tab"] = "Upload CSV"
    st.markdown("### Mô hình tốt nhất")
    st.markdown(
        f"""
        <div class='sidebar-note'>
            <b>{best_model_row['Model']}</b><br>
            F1-score: <b>{best_model_row['F1-score']:.4f}</b><br>
            Recall: <b>{best_model_row['Recall']:.4f}</b><br>
            PR-AUC: <b>{best_model_row['PR-AUC']:.4f}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Tệp mẫu")
    st.download_button(
        "Tải CSV mẫu",
        data=make_template_csv(),
        file_name="sample_input.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown(
    f"""
    <div class='hero'>
        <div class='hero-grid'>
            <div class='logo-shell'>
                {f"<img src='{LOGO_URI}' style='width:100%; height:auto; display:block;'>" if LOGO_URI else ""}
            </div>
            <div>
                <div class='hero-title'>IE105 Fraud Detection Dashboard</div>
                <div class='hero-sub'>Bảng điều khiển trực quan cho đồ án ứng dụng Machine Learning và Deep Learning vào phát hiện bất thường trong giao dịch tài chính. Giao diện ưu tiên tính hiện đại, độ tương phản cao và nhấn mạnh các caption để đọc nhanh khi demo.</div>
                <div class='pill-wrap'>
                    <span class='pill'>UIT • VNUHCM</span>
                    <span class='pill'>Credit Card Fraud Detection</span>
                    <span class='pill'>3 mô hình đánh giá</span>
                    <span class='pill'>Best model: Logistic Regression</span>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi("Mô hình tốt nhất", best_model_row["Model"], "Xếp hạng theo F1-score, PR-AUC và Recall")
with k2:
    render_kpi("Recall cao nhất", f"{summary_df['Recall'].max():.4f}", "Độ nhạy trong phát hiện fraud")
with k3:
    render_kpi("PR-AUC cao nhất", f"{summary_df['PR-AUC'].max():.4f}", "Quan trọng với dữ liệu mất cân bằng")
with k4:
    render_kpi("Số mô hình", "3", "Isolation Forest · Autoencoder · Logistic Regression")


tab_labels = ["Tổng quan", "So sánh mô hình", "Dự đoán 1 giao dịch", "Upload CSV"]
tab1, tab2, tab3, tab4 = st.tabs(tab_labels)

target_tab = st.session_state.pop("target_tab", None)
if target_tab in tab_labels:
    escaped_tab = json.dumps(target_tab)
    components.html(
        f"""
        <script>
        const targetName = {escaped_tab};
        const tabButtons = Array.from(window.parent.document.querySelectorAll('button[role="tab"]'));
        const targetButton = tabButtons.find((btn) => btn.innerText.trim() === targetName);
        if (targetButton) targetButton.click();
        </script>
        """,
        height=0,
    )

with tab1:
    left, right = st.columns([1.03, 1.37], gap="large")
    with left:
        st.markdown("<div class='section-title'>Thông tin đề tài</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='card'>
                <ul>
                    <li><b>Môn học:</b> Nhập môn bảo đảm và an ninh thông tin (IE105)</li>
                    <li><b>Chủ đề:</b> Ứng dụng ML, DL vào phát hiện bất thường trong giao dịch tài chính</li>
                    <li><b>Tập dữ liệu:</b> Credit Card Fraud Detection</li>
                    <li><b>Đầu vào:</b> 30 đặc trưng gồm <code>Time</code>, <code>V1-V28</code>, <code>Amount</code></li>
                    <li><b>3 mô hình:</b> Isolation Forest, Autoencoder, Logistic Regression</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-title'>Bảng metrics tổng hợp</div>", unsafe_allow_html=True)
        show_df = summary_df.copy()
        num_cols = show_df.select_dtypes(include=["number"]).columns
        show_df[num_cols] = show_df[num_cols].apply(pd.to_numeric, errors="coerce").round(6)
        st.dataframe(show_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown("<div class='section-title'>Khám phá dữ liệu</div>", unsafe_allow_html=True)
        eda_pairs = [
            ("eda_class_distribution.png", "Phân bố lớp", "Cho thấy dữ liệu mất cân bằng rất mạnh giữa normal và fraud."),
            ("eda_amount_distribution.png", "Phân bố Amount", "Giá trị giao dịch có phân phối lệch phải, tập trung ở vùng nhỏ."),
            ("eda_log_amount_distribution.png", "Phân bố log(Amount+1)", "Biến đổi log giúp quan sát dữ liệu ổn định và trực quan hơn."),
            ("eda_time_distribution.png", "Phân bố Time", "Mô tả sự phân bố các giao dịch theo thời gian thu thập."),
            ("eda_top_corr_heatmap.png", "Heatmap tương quan", "Các đặc trưng PCA có tương quan khác nhau với nhãn gian lận."),
        ]
        c1, c2 = st.columns(2, gap="large")
        for idx, (fname, title, subtitle) in enumerate(eda_pairs):
            col = c1 if idx % 2 == 0 else c2
            with col:
                render_image_with_caption(IMAGE_DIR / fname, title, subtitle)

with tab2:
    st.markdown("<div class='section-title'>Bảng xếp hạng mô hình</div>", unsafe_allow_html=True)
    rank_df = summary_df[["Model", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC", "MCC"]].copy()
    num_cols = rank_df.select_dtypes(include=["number"]).columns
    rank_df[num_cols] = rank_df[num_cols].apply(pd.to_numeric, errors="coerce").round(6)
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    mcol1, mcol2 = st.columns([0.95, 1.05], gap="large")
    with mcol1:
        st.markdown("<div class='section-title'>Biểu đồ so sánh chỉ số</div>", unsafe_allow_html=True)
        metric_choice = st.selectbox(
            "Chọn chỉ số cần trực quan hóa",
            ["Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC", "Accuracy", "Balanced Accuracy", "MCC"],
            index=1,
        )
        chart_df = summary_df[["Model", metric_choice]].set_index("Model")
        st.bar_chart(chart_df, height=350)
        st.caption("Biểu đồ cột giúp so sánh nhanh hiệu quả của ba mô hình theo từng chỉ số đánh giá.")

    with mcol2:
        st.markdown("<div class='section-title'>Ngưỡng phân loại</div>", unsafe_allow_html=True)
        threshold_df = pd.DataFrame({
            "Model": ["Isolation Forest", "Autoencoder", "Logistic Regression"],
            "Threshold": [thresholds["Isolation Forest"], thresholds["Autoencoder"], thresholds["Logistic Regression"]],
        })
        threshold_df["Threshold"] = pd.to_numeric(threshold_df["Threshold"], errors="coerce").round(6)
        st.dataframe(threshold_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class='card'>
                <b>Nhận xét nhanh</b><br>
                Logistic Regression cho kết quả nổi bật nhất theo thực nghiệm hiện tại, đặc biệt mạnh ở Precision, Recall, F1-score và PR-AUC.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
    cm1, cm2, cm3 = st.columns(3, gap="large")
    with cm1:
        render_image_with_caption(IMAGE_DIR / "cm_if.png", "Isolation Forest", "Hiển thị số lượng TN, FP, FN, TP của mô hình Isolation Forest.")
    with cm2:
        render_image_with_caption(IMAGE_DIR / "cm_ae.png", "Autoencoder", "Phát hiện fraud tốt hơn IF nhưng vẫn còn nhiều false positives.")
    with cm3:
        render_image_with_caption(IMAGE_DIR / "cm_lr.png", "Logistic Regression", "Confusion matrix tốt nhất, ít bỏ sót fraud và rất ít báo động giả.")

    st.markdown("<div class='section-title'>ROC và Precision–Recall Curve</div>", unsafe_allow_html=True)
    roc_col, pr_col = st.columns(2, gap="large")
    with roc_col:
        render_image_with_caption(IMAGE_DIR / "roc_compare.png", "So sánh ROC Curve", "ROC-AUC cao phản ánh khả năng phân tách lớp trên nhiều ngưỡng.")
    with pr_col:
        render_image_with_caption(IMAGE_DIR / "pr_compare.png", "So sánh Precision–Recall Curve", "PR-AUC phù hợp hơn trong bối cảnh dữ liệu gian lận mất cân bằng mạnh.")

with tab3:
    st.markdown("<div class='section-title'>Dự đoán cho một giao dịch</div>", unsafe_allow_html=True)
    st.caption("Các đặc trưng V1-V28 là kết quả PCA trong bộ dữ liệu gốc, nên phần nhập tay chủ yếu dùng để demo kỹ thuật mô hình.")

    with st.expander("Nhập 30 đặc trưng", expanded=True):
        inputs = {}
        cols = st.columns(3)
        for i, feat in enumerate(feature_names):
            with cols[i % 3]:
                default = 100.0 if feat == "Amount" else 0.0
                inputs[feat] = st.number_input(feat, value=float(default), format="%.6f", key=f"manual_{feat}")

    if st.button("Chạy dự đoán", type="primary", use_container_width=True):
        input_df = pd.DataFrame([inputs])
        result_df = score_one_row(input_df)

        st.markdown("<div class='section-title'>Kết quả dự đoán</div>", unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3, gap="large")
        notes = {
            "Logistic Regression": "Mô hình tốt nhất theo kết quả thực nghiệm hiện tại.",
            "Isolation Forest": "Mô hình anomaly detection không giám sát.",
            "Autoencoder": "Mô hình deep learning dựa trên reconstruction error.",
        }
        with p1:
            row = result_df[result_df["Model"] == "Logistic Regression"].iloc[0]
            render_prediction_card(row["Model"], row["Prediction"], row["Score"], row["Threshold"], notes[row["Model"]])
        with p2:
            row = result_df[result_df["Model"] == "Isolation Forest"].iloc[0]
            render_prediction_card(row["Model"], row["Prediction"], row["Score"], row["Threshold"], notes[row["Model"]])
        with p3:
            row = result_df[result_df["Model"] == "Autoencoder"].iloc[0]
            render_prediction_card(row["Model"], row["Prediction"], row["Score"], row["Threshold"], notes[row["Model"]])

        st.markdown("<div class='section-title'>Bảng chi tiết</div>", unsafe_allow_html=True)
        show_pred = result_df.copy()
        show_pred[["Score", "Threshold"]] = show_pred[["Score", "Threshold"]].apply(pd.to_numeric, errors="coerce").round(6)
        st.dataframe(show_pred, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("<div class='section-title'>Dự đoán hàng loạt từ file CSV</div>", unsafe_allow_html=True)
    st.caption("File CSV cần chứa đủ 30 cột: Time, V1-V28, Amount. App sẽ chạy đồng thời cả 3 mô hình.")

    with st.expander("Xem danh sách cột yêu cầu"):
        st.code(", ".join(feature_names), language="text")

    uploaded = st.file_uploader("Chọn file CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.markdown("<div class='section-title'>Xem trước dữ liệu</div>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            st.error("Thiếu các cột: " + ", ".join(missing))
        else:
            pred_df = run_batch(df)
            st.success("Đã chạy xong cho cả 3 mô hình.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud theo LR", int(pred_df["LR_Pred"].sum()))
            c2.metric("Fraud theo IF", int(pred_df["IF_Pred"].sum()))
            c3.metric("Fraud theo AE", int(pred_df["AE_Pred"].sum()))

            st.markdown("<div class='section-title'>Kết quả đầu ra</div>", unsafe_allow_html=True)
            st.dataframe(pred_df.head(20), use_container_width=True)
            prediction_download_button(pred_df, "Tải kết quả CSV", "fraud_predictions.csv")
