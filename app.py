"""
HCC 디지털 트윈 PoC - 의사용 인터랙티브 대시보드 (Streamlit)
=================================================================
!! 주의: 본 대시보드는 patients_timeseries.csv의 SYNTHETIC(가상) 데이터와
   그것으로 학습된 모델(model.pkl)만을 사용하는 개념 검증(PoC)이다.
   실제 환자 데이터가 아니며, 예측 확률/시나리오 비교 결과는 실제 임상적
   유효성이나 인과적 효과를 의미하지 않는다. 어떠한 임상적 의사결정에도
   사용해서는 안 된다.

실행: streamlit run app.py
"""

import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from plotly.subplots import make_subplots

from train_response_model import BIOMARKERS, build_feature_table

DATA_PATH = "patients_timeseries.csv"
MODEL_PATH = "model.pkl"
SHAP_VALUES_PATH = "shap_values.pkl"

BIOMARKER_LABELS = {
    "vaf": "ctDNA VAF",
    "methylation_score": "Methylation Score",
    "radiomics_score": "Radiomics Score",
    "afp": "AFP (ng/mL)",
}

st.set_page_config(page_title="HCC Digital Twin PoC", layout="wide")


# ---------------------------------------------------------------------------
# 캐싱된 로더
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_feature_table(df):
    return build_feature_table(df)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_shap_bundle():
    with open(SHAP_VALUES_PATH, "rb") as f:
        return pickle.load(f)


def predict_proba(model_bundle, feature_row):
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    X = feature_row.reindex(feature_names).to_frame().T
    return float(model.predict_proba(X)[:, 1][0])


def apply_early_switch_scenario(feature_row, improvement=0.20):
    """
    "조기 치료 전환" 가정: T2 시점 최근 추세(T1->T2 변화율)가
    개선된다고 단순 가정하여 재계산한다.
    - 이미 감소(호전) 중이던 지표는 감소 속도를 improvement 만큼 더 가속.
    - 증가(악화) 중이던 지표는 증가 폭을 improvement 만큼 완화.
    T0, T1 값과 T0->T1 변화율은 그대로 유지 (개입은 T2 시점 전후로 가정).
    """
    scenario_row = feature_row.copy()
    for b in BIOMARKERS:
        v_t1 = feature_row[f"{b}_T1"]
        change = feature_row[f"{b}_change_T1_T2"]
        new_change = change * (1 + improvement) if change < 0 else change * (1 - improvement)
        new_v_t2 = v_t1 * (1 + new_change)

        scenario_row[f"{b}_T2"] = new_v_t2
        scenario_row[f"{b}_change_T1_T2"] = new_change
    return scenario_row


def render_waterfall(shap_bundle, patient_id):
    idx = shap_bundle["patient_ids"].index(patient_id)
    explanation = shap.Explanation(
        values=shap_bundle["shap_values"][idx],
        base_values=shap_bundle["base_values"][idx],
        data=shap_bundle["data"][idx],
        feature_names=shap_bundle["feature_names"],
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


# ---------------------------------------------------------------------------
# 상단 경고 배지
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="background-color:#7a1f1f; color:white; padding:14px 18px;
                border-radius:8px; font-size:16px; margin-bottom:18px;">
        ⚠️ <b>Synthetic Data — For Demonstration Purposes Only.</b><br/>
        이 대시보드는 실제 환자 데이터가 아닌 가상 시뮬레이션 데이터를 사용한
        개념 검증(PoC)입니다.
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("HCC Digital Twin PoC — Atezolizumab + Bevacizumab 반응 예측")

# ---------------------------------------------------------------------------
# 데이터 / 모델 로딩
# ---------------------------------------------------------------------------
df = load_data()
X_all, y_all = load_feature_table(df)
model_bundle = load_model()
shap_bundle = load_shap_bundle()

# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
patient_ids = sorted(df["patient_id"].unique())
with st.sidebar:
    st.header("환자 선택")
    selected_patient = st.selectbox("Patient ID", patient_ids)
    st.caption(f"LOOCV AUROC (학습 시): {model_bundle['loocv_auroc']:.3f}")

patient_df = df[df["patient_id"] == selected_patient].sort_values("week")
patient_meta = patient_df.iloc[0]
feature_row = X_all.loc[selected_patient]

# ---------------------------------------------------------------------------
# a) 기본 정보 카드
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Patient ID", selected_patient)
col2.metric("CTNNB1 mutation", "Yes" if patient_meta["ctnnb1_mutation"] else "No")
col3.metric("실제 response_label (T3)", patient_meta["response_label"])

st.divider()

# ---------------------------------------------------------------------------
# b) T0~T3 시계열 그래프 (탭)
# ---------------------------------------------------------------------------
st.subheader("바이오마커 시계열 (T0 -> T1 -> T2 -> T3)")
tabs = st.tabs([BIOMARKER_LABELS[b] for b in BIOMARKERS])
for tab, b in zip(tabs, BIOMARKERS):
    with tab:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=patient_df["timepoint"], y=patient_df[b],
            mode="lines+markers", line=dict(width=3), marker=dict(size=10),
            name=BIOMARKER_LABELS[b],
        ))
        fig.add_vline(x=2.5, line_dash="dot", line_color="gray")  # T2/T3 경계(예측 대상 표시)
        fig.update_layout(
            xaxis_title="Timepoint", yaxis_title=BIOMARKER_LABELS[b],
            height=380, margin=dict(t=30),
        )
        st.plotly_chart(fig, use_container_width=True)
st.caption("T3(12주)는 모델 입력에서 제외되고 예측 대상으로만 사용됩니다.")

st.divider()

# ---------------------------------------------------------------------------
# c) 12주 후 반응 확률 예측
# ---------------------------------------------------------------------------
st.subheader("12주 후 반응 예측")
current_proba = predict_proba(model_bundle, feature_row)
pred_col, shap_col = st.columns([1, 2])
with pred_col:
    st.metric("12주 후 반응(Responder) 확률", f"{current_proba:.1%}")
    st.caption("모델 입력: T0, T1, T2 시점 데이터 + T0→T1 / T1→T2 변화율")

with shap_col:
    st.markdown("**SHAP Waterfall — 이 환자 예측에 대한 변수별 기여도**")
    render_waterfall(shap_bundle, selected_patient)

st.divider()

# ---------------------------------------------------------------------------
# 시나리오 비교 섹션
# ---------------------------------------------------------------------------
st.subheader("시나리오 비교")
scenario = st.radio(
    "치료 시나리오 선택",
    ["현재 치료 유지", "조기 치료 전환 가정"],
    horizontal=True,
)

scenario_row = apply_early_switch_scenario(feature_row, improvement=0.20)
scenario_proba = predict_proba(model_bundle, scenario_row)

bar_fig = go.Figure(data=[
    go.Bar(
        x=["현재 치료 유지", "조기 치료 전환 가정"],
        y=[current_proba, scenario_proba],
        marker_color=["#4C78A8", "#F58518"],
        text=[f"{current_proba:.1%}", f"{scenario_proba:.1%}"],
        textposition="outside",
    )
])
bar_fig.update_layout(
    yaxis_title="12주 후 반응(Responder) 확률",
    yaxis_range=[0, 1],
    height=400,
    margin=dict(t=30),
)
st.plotly_chart(bar_fig, use_container_width=True)

if scenario == "현재 치료 유지":
    st.info(f"현재 치료 유지 시 예측 반응 확률: **{current_proba:.1%}**")
else:
    st.info(
        f"조기 치료 전환 가정 시 예측 반응 확률: **{scenario_proba:.1%}** "
        f"(현재 유지 대비 {scenario_proba - current_proba:+.1%}p)"
    )

st.caption(
    "※ 조기 전환 가정은 T1→T2 구간의 악화/호전 추세를 임의로 20% 완화·가속하는 "
    "단순 규칙 기반 시뮬레이션입니다. 실제 인과적 효과 추정이 아닌, "
    "모델 기반 탐색적 비교입니다."
)
