"""
HCC 디지털 트윈 PoC - T0/T1/T2 데이터 기반 T3(12주) 치료 반응 예측 모델
=================================================================
!! 주의: 아래에서 계산되는 AUROC 등 모든 성능 지표는 patients_timeseries.csv의
   SYNTHETIC(가상) 데이터를 기반으로 산출된 것이며, 실제 임상적 유효성이나
   실제 환자 집단에서의 예측 성능을 의미하지 않는다. 어떠한 임상적
   의사결정에도 사용해서는 안 되며, 파이프라인/코드 검증 목적의 PoC이다.

설정:
   - 입력(feature): T0(baseline), T1(2주), T2(6주) 시점의 ctDNA VAF,
     methylation score, radiomics score, AFP + T0->T1, T1->T2 변화율
     + CTNNB1_mutation(baseline에 이미 알려진 값)
   - 목표(target): T3(12주) 시점의 response_label (Responder=1 / Non-responder=0)
   - 모델: XGBoost 분류기 (XGBClassifier)
   - 평가: 환자 수(20명)가 적어 train/test 분할 대신 Leave-One-Out
     Cross-Validation(LOOCV)으로 AUROC 산출
   - 해석: 전체 데이터로 학습한 최종 모델에 SHAP(TreeExplainer) 적용
     -> summary plot, 가장 예측이 애매했던(LOOCV 예측확률이 0.5에 가장 가까운)
        환자 1명의 waterfall plot 저장
   - 산출물: model.pkl, shap_values.pkl (대시보드에서 재사용)
"""

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBClassifier

RANDOM_SEED = 42
INPUT_CSV = "patients_timeseries.csv"
FEATURE_TIMEPOINTS = ["T0", "T1", "T2"]
BIOMARKERS = ["vaf", "methylation_score", "radiomics_score", "afp"]

MODEL_PATH = "model.pkl"
SHAP_VALUES_PATH = "shap_values.pkl"
SUMMARY_PLOT_PATH = "shap_summary_plot.png"
WATERFALL_PLOT_PATH = "shap_waterfall_ambiguous_patient.png"

XGB_PARAMS = dict(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=RANDOM_SEED,
)


def safe_pct_change(v_from, v_to):
    """0으로 나누기를 방지한 변화율 (v_to - v_from) / |v_from|."""
    denom = np.where(np.abs(v_from) < 1e-6, 1e-6, np.abs(v_from))
    return (v_to - v_from) / denom


def build_feature_table(df):
    """long format(T0~T3) -> 환자 1행 = 1레코드의 wide feature 테이블(T0~T2만 사용)."""
    wide = df.pivot(index="patient_id", columns="timepoint", values=BIOMARKERS)
    wide.columns = [f"{biomarker}_{tp}" for biomarker, tp in wide.columns]

    meta = (
        df.drop_duplicates("patient_id")
        .set_index("patient_id")[["response_label", "ctnnb1_mutation"]]
    )

    feat = wide[[f"{b}_{tp}" for b in BIOMARKERS for tp in FEATURE_TIMEPOINTS]].copy()

    # T0->T1, T1->T2 변화율 파생 변수
    for b in BIOMARKERS:
        feat[f"{b}_change_T0_T1"] = safe_pct_change(
            wide[f"{b}_T0"].values, wide[f"{b}_T1"].values
        )
        feat[f"{b}_change_T1_T2"] = safe_pct_change(
            wide[f"{b}_T1"].values, wide[f"{b}_T2"].values
        )

    feat["ctnnb1_mutation"] = meta["ctnnb1_mutation"].astype(int)
    y = (meta["response_label"] == "Responder").astype(int)

    feat = feat.sort_index()
    y = y.loc[feat.index]
    return feat, y


def run_loocv(X, y):
    """LOOCV로 out-of-fold 예측확률을 구하고 AUROC를 계산."""
    loo = LeaveOneOut()
    oof_proba = pd.Series(index=X.index, dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)
        oof_proba.iloc[test_idx] = model.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y, oof_proba)
    return oof_proba, auroc


def main():
    df = pd.read_csv(INPUT_CSV)
    X, y = build_feature_table(df)
    print(f"Feature matrix: {X.shape[0]} patients x {X.shape[1]} features")
    print(f"Responders: {y.sum()} / {len(y)}\n")

    # --- LOOCV 성능 평가 ---
    oof_proba, auroc = run_loocv(X, y)
    print(f"LOOCV AUROC: {auroc:.3f}\n")

    results = pd.DataFrame({
        "response_label": y.map({1: "Responder", 0: "Non-responder"}),
        "loocv_predicted_proba": oof_proba.round(4),
    })
    print(results)

    # --- 최종 모델: 전체 데이터로 재학습 (해석용) ---
    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X, y)

    # --- SHAP (TreeExplainer) ---
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(X)  # shap.Explanation object

    # 1) 전체 변수 중요도 summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved SHAP summary plot -> {SUMMARY_PLOT_PATH}")

    # 2) 가장 예측이 애매한 환자(LOOCV 예측확률이 0.5에 가장 가까움) waterfall plot
    ambiguous_patient = (oof_proba - 0.5).abs().idxmin()
    ambiguous_pos = X.index.get_loc(ambiguous_patient)
    print(
        f"Most ambiguous patient: {ambiguous_patient} "
        f"(LOOCV predicted proba = {oof_proba.loc[ambiguous_patient]:.3f}, "
        f"true label = {results.loc[ambiguous_patient, 'response_label']})"
    )

    plt.figure()
    shap.plots.waterfall(shap_values[ambiguous_pos], show=False)
    plt.tight_layout()
    plt.savefig(WATERFALL_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP waterfall plot for {ambiguous_patient} -> {WATERFALL_PLOT_PATH}")

    # --- 대시보드용 산출물 저장 ---
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": final_model,
            "feature_names": list(X.columns),
            "xgb_params": XGB_PARAMS,
            "loocv_auroc": auroc,
        }, f)
    print(f"\nSaved model -> {MODEL_PATH}")

    with open(SHAP_VALUES_PATH, "wb") as f:
        pickle.dump({
            "shap_values": shap_values.values,
            "base_values": shap_values.base_values,
            "data": shap_values.data,
            "feature_names": list(X.columns),
            "patient_ids": list(X.index),
            "ambiguous_patient_id": ambiguous_patient,
            "loocv_predicted_proba": oof_proba,
        }, f)
    print(f"Saved SHAP values -> {SHAP_VALUES_PATH}")

    return X, y, oof_proba, auroc, final_model, shap_values


if __name__ == "__main__":
    main()
