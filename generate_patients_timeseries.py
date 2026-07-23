"""
HCC 디지털 트윈 PoC - 가상 환자 시계열 데이터 생성 스크립트
=================================================================
!! 주의: 본 스크립트가 생성하는 모든 데이터는 100% SYNTHETIC(가상) 데이터입니다.
   실제 환자 데이터, 임상시험 결과, 실측값이 아니며, 알고리즘/파이프라인
   프로토타이핑 목적의 통계적 시뮬레이션입니다. 어떠한 임상적 의사결정에도
   사용해서는 안 됩니다.

배경:
   Atezolizumab + Bevacizumab 병용요법을 받는 진행성 간세포암종(HCC) 환자를
   가정하여, T0(baseline), T1(2주), T2(6주), T3(12주) 4개 시점에서
   ctDNA VAF, DNA methylation score, radiomics score, AFP를 시뮬레이션한다.

CTNNB1-cold tumor 가정에 대한 근거(단순화):
   CTNNB1(베타카테닌 경로) 활성화 돌연변이를 가진 HCC는 면역학적으로
   "cold tumor"(T세포 침윤이 적은 종양미세환경)로 보고되며, 면역관문억제제
   기반 치료(atezolizumab+bevacizumab 등)에 대한 반응률이 낮은 경향이
   문헌에서 시사된 바 있다 (Ogawa et al., 2022, 관련 문헌들을 단순화하여
   참고함). 본 스크립트는 이 가설을 다음의 단순 규칙으로만 반영한다:
     1) CTNNB1 변이 양성 환자는 반응군(responder)으로 배정될 확률을 낮춘다.
     2) CTNNB1 변이 양성 반응군은 ctDNA VAF 감소 속도를 더 느리게 만든다.
   이는 실제 문헌의 정량적 수치를 그대로 사용한 것이 아니라, PoC 파이프라인
   검증을 위해 방향성만 단순화하여 규칙화한 것이다.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_PATIENTS = 20
TIMEPOINTS = ["T0", "T1", "T2", "T3"]
WEEKS = {"T0": 0, "T1": 2, "T2": 6, "T3": 12}

CTNNB1_MUTATION_PROB = 0.20
TARGET_RESPONSE_RATE = 0.30
# CTNNB1 양성 환자는 반응군으로 뽑힐 가중치를 낮춘다 (야생형 대비 상대 가중치).
CTNNB1_RESPONSE_WEIGHT_RATIO = 0.10 / 0.35

OUTPUT_PATH = "patients_timeseries.csv"

rng = np.random.default_rng(RANDOM_SEED)


def clip(x, lo, hi):
    return np.clip(x, lo, hi)


def exponential_trend(baseline, total_change_fraction, week, noise_sigma):
    """
    baseline(T0) 값에서 T3(12주) 시점까지 total_change_fraction 만큼
    변화하는 지수적 추세 곡선을 주 단위로 보간하고, 시점별 곱셈 노이즈를 더한다.

    total_change_fraction: (value_T3 - value_T0) / value_T0
    """
    ratio_total = max(1.0 + total_change_fraction, 1e-6)
    trend_value = baseline * (ratio_total ** (week / WEEKS["T3"]))
    noise = rng.normal(1.0, noise_sigma)
    noise = max(noise, 0.0)  # 음수 배율 방지
    return trend_value * noise


def simulate_patient(patient_id, rng, is_ctnnb1, is_responder):
    response_label = "Responder" if is_responder else "Non-responder"

    # --- ctDNA VAF ---
    vaf0 = rng.uniform(0.05, 0.4)
    if is_responder:
        # 평균 60~80% 감소. CTNNB1 양성 반응군은 감소 속도를 더 느리게(폭을 줄임).
        if is_ctnnb1:
            vaf_decline = rng.uniform(0.30, 0.50)
        else:
            vaf_decline = rng.uniform(0.60, 0.80)
        vaf_change_fraction = -vaf_decline
        vaf_noise_sigma = 0.08
    else:
        vaf_change_fraction = rng.uniform(-0.10, 0.30)
        vaf_noise_sigma = 0.12

    # --- Methylation score (0~1, VAF와 유사한 방향, 노이즈는 더 크게) ---
    meth0 = rng.uniform(0.10, 0.60)
    if is_responder:
        meth_decline = rng.uniform(0.50, 0.75) * (0.6 if is_ctnnb1 else 1.0)
        meth_change_fraction = -meth_decline
    else:
        meth_change_fraction = rng.uniform(-0.10, 0.30)
    meth_noise_sigma = 0.18

    # --- Radiomics score (0~100, 종양 부담 지표) ---
    radiomics0 = rng.uniform(20, 80)
    if is_responder:
        radiomics_decline = rng.uniform(0.40, 0.70) * (0.6 if is_ctnnb1 else 1.0)
        radiomics_change_fraction = -radiomics_decline
    else:
        radiomics_change_fraction = rng.uniform(-0.05, 0.25)
    radiomics_noise_sigma = 0.15

    # --- AFP (ng/mL, log-normal 유사, 10~5000) ---
    afp0 = clip(rng.lognormal(mean=np.log(200), sigma=1.2), 10, 5000)
    if is_responder:
        afp_decline = rng.uniform(0.50, 0.80) * (0.6 if is_ctnnb1 else 1.0)
        afp_change_fraction = -afp_decline
        afp_noise_sigma = 0.20
    else:
        # 비반응군은 뚜렷한 추세 없이 불규칙하게 변동
        afp_change_fraction = rng.uniform(-0.20, 0.40)
        afp_noise_sigma = 0.45

    rows = []
    for tp in TIMEPOINTS:
        week = WEEKS[tp]
        vaf = clip(
            exponential_trend(vaf0, vaf_change_fraction, week, vaf_noise_sigma),
            0.0, 1.0,
        )
        meth = clip(
            exponential_trend(meth0, meth_change_fraction, week, meth_noise_sigma),
            0.0, 1.0,
        )
        radiomics = clip(
            exponential_trend(radiomics0, radiomics_change_fraction, week, radiomics_noise_sigma),
            0.0, 100.0,
        )
        afp = clip(
            exponential_trend(afp0, afp_change_fraction, week, afp_noise_sigma),
            10.0, 5000.0,
        )

        rows.append({
            "patient_id": patient_id,
            "timepoint": tp,
            "week": week,
            "ctnnb1_mutation": is_ctnnb1,
            "response_label": response_label,
            "vaf": round(float(vaf), 4),
            "methylation_score": round(float(meth), 4),
            "radiomics_score": round(float(radiomics), 2),
            "afp": round(float(afp), 2),
        })
    return rows


def assign_ctnnb1_and_response(rng):
    """
    CTNNB1 변이는 개별 20% 확률로 배정.
    반응군 인원수는 목표 반응률(30%)에 맞춰 round(N*0.3)명으로 고정하되,
    CTNNB1 양성 환자는 반응군으로 뽑힐 가중치를 낮춰(CTNNB1-cold tumor 가정)
    가중 비복원추출로 선택한다.
    """
    ctnnb1_flags = rng.random(N_PATIENTS) < CTNNB1_MUTATION_PROB
    n_responders = round(N_PATIENTS * TARGET_RESPONSE_RATE)

    weights = np.where(ctnnb1_flags, CTNNB1_RESPONSE_WEIGHT_RATIO, 1.0)
    weights = weights / weights.sum()

    responder_idx = rng.choice(
        N_PATIENTS, size=n_responders, replace=False, p=weights
    )
    responder_flags = np.zeros(N_PATIENTS, dtype=bool)
    responder_flags[responder_idx] = True
    return ctnnb1_flags, responder_flags


def main():
    ctnnb1_flags, responder_flags = assign_ctnnb1_and_response(rng)

    all_rows = []
    for i in range(1, N_PATIENTS + 1):
        patient_id = f"P{i:03d}"
        all_rows.extend(
            simulate_patient(
                patient_id, rng,
                is_ctnnb1=bool(ctnnb1_flags[i - 1]),
                is_responder=bool(responder_flags[i - 1]),
            )
        )

    df = pd.DataFrame(all_rows)
    df = df[[
        "patient_id", "timepoint", "week", "response_label", "ctnnb1_mutation",
        "vaf", "methylation_score", "radiomics_score", "afp",
    ]]
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows ({df['patient_id'].nunique()} patients) to {OUTPUT_PATH}")

    n_responders = df.loc[df["timepoint"] == "T0", "response_label"].eq("Responder").sum()
    n_total = df["patient_id"].nunique()
    print(f"Responder count: {n_responders}/{n_total} ({n_responders / n_total:.0%})")

    print("\n--- head(10) ---")
    print(df.head(10).to_string(index=False))

    print("\n--- describe by response_label (numeric columns) ---")
    for label, group in df.groupby("response_label"):
        print(f"\n[{label}] n_patients={group['patient_id'].nunique()}")
        print(group[["vaf", "methylation_score", "radiomics_score", "afp"]].describe())

    return df


if __name__ == "__main__":
    main()
