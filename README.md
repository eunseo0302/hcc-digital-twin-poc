# HCC Digital Twin PoC

2026년 고려대학교 의과대학 의사과학자 양성사업 실험실습수업에서 작성하여 시상 대상으로 선정된 연구계획서의 핵심 산출물(세부목표 3-3: 환자 맞춤형 치료 경로 제시 시스템 프로토타입)을, 개인적으로 확장하여 실제 동작하는 PoC로 구현한 프로젝트입니다.

---

## ⚠️ 중요 고지

**이 프로젝트는 synthetic(가상) 데이터를 사용한 소프트웨어 엔지니어링 PoC이며, 실제 임상적 예측 성능이나 유효성을 주장하지 않습니다.**

**이 프로젝트가 증명하는 것**
- 시계열 바이오마커(ctDNA VAF, methylation score, radiomics score, AFP) 데이터 파이프라인이 설계대로 동작함
- T0~T2 데이터로 T3 반응을 예측하는 XGBoost 모델링 + LOOCV 평가 파이프라인이 정상 동작함
- SHAP 기반 설명가능성(변수 중요도, 환자별 waterfall) 통합이 정상 동작함
- 의사가 사용할 수 있는 형태의 인터랙티브 대시보드(Streamlit) UI가 계획서 설계대로 통합되어 동작함

**이 프로젝트가 증명하지 않는 것**
- 실제 임상적 예측 정확도 (여기 제시된 AUROC 등은 전부 synthetic data 기준)
- 실제 환자 데이터 기반의 결론이나 임상적 유효성
- CTNNB1-cold tumor 가정 등 문헌 기반 규칙의 정량적 정확성 (방향성만 단순화하여 반영한 것)
- 시나리오 비교 섹션의 인과적 효과 (모델 기반 탐색적 비교일 뿐, 실제 치료 전환의 인과 효과 추정이 아님)

---

## 배경

이 프로젝트의 출발점이 된 연구계획서는 2026년 고려대학교 의과대학 의사과학자 양성사업 실험실습수업의 결과물로 작성되었으며, 해당 수업에서 시상 대상으로 선정되었습니다.

실제로 연구비가 집행되거나 임상 코호트가 운영된 적은 없으며, 수업 과제로서 연구계획서 작성 역량을 평가받은 것입니다. 이 GitHub 프로젝트는 그 계획서의 세부목표 3-3에 해당하는 아이디어를 개인적으로 확장하여, synthetic data 기반 소프트웨어 PoC로 구현한 것입니다. **수업의 공식 후속 프로젝트가 아니라, 개인 학습 및 포트폴리오 목적의 독립적인 사이드 프로젝트임을 명시합니다.**

**원본 문제의식 요약**: 진행성 간세포암종(HCC)에서 Atezolizumab+Bevacizumab 병용요법의 반응률은 약 30%에 불과하고, 치료 전 단일 시점 조직검사만으로는 치료 중 종양의 동적 변화(반응/저항 획득 등)를 포착할 수 없다는 한계가 있습니다. 이 연구계획서는 이를 시계열 ctDNA 기반 "디지털 트윈" 개념으로 해결하고자 했습니다.

원본 연구계획서: [연구계획서 원문](./연구계획서.pdf)

---

## 이 프로젝트가 구현한 범위

원 연구계획서는 3단계 구조로 설계되었습니다. 이 PoC가 실제로 구현한 범위는 아래와 같습니다.

| 세부목표 | 계획서 내용(요약) | 구현 여부 | 비고 |
|---|---|---|---|
| 1-1 (데이터 수집 및 전처리 파이프라인 구축) | 환자 코호트 등록, ctDNA(baseline WES + 시계열 targeted sequencing), 영상 데이터(CT/MRI radiomics), 임상·병리학적 지표를 수집하고 QC 프로토콜을 수립. 환자 등록 30명 이상, 시계열 데이터 완성률 80% 이상이 목표.[^1] | 미구현 | 실제 환자 검체 수집/전처리는 수행하지 않음. synthetic 데이터로 스키마만 모사 |
| 1-2 (환자별 종양 상태 동적 표현 모델 개발) | 멀티오믹스 데이터 통합 구조를 설계하고, VAF 변화율·메틸화 추이·종양 성장 속도 등 시계열 특징을 추출하여 환자별 종양 상태 벡터를 정의.[^1] | 미구현 | 실제 어세이(ctDNA/methylation/radiomics) 파이프라인 없음. 다만 시계열 특징(변화율) 추출 로직은 `train_response_model.py`의 feature 엔지니어링으로 개념만 재현 |
| 2-1 (시계열 예측 모델 구현) | 규제 기반 머신러닝(XGBoost, Elastic Net with L1/L2 regularization)을 주 분석으로, LSTM 기반 시계열 딥러닝을 보조 분석으로 하여 12주 치료 반응을 예측하는 모델을 학습하고 baseline 모델과 성능 비교. 목표 AUROC ≥ 0.72 (baseline 대비 +0.10 이상 개선).[^1] | 일부 구현 | XGBoost만 구현했고 Elastic Net·LSTM 보조 분석 및 baseline 대비 비교는 미구현. synthetic 데이터 기반 |
| 2-3 (표준 치료 하 반응/비반응 패턴 모델링) | SHAP을 이용한 변수 기여도 분석과, 시계열 모델 vs 단일시점 모델의 비교를 통해 상위 10개 변수 중요도를 시각화.[^1] | 일부 구현 | SHAP summary/waterfall 플롯 구현. 시계열 모델 vs 단일시점 모델 비교는 미구현. 실제 데이터 기반 검증은 없음 |
| 3-1 (치료 시나리오별 예측 결과 비교 분석) | "현재 치료 지속" 시나리오와 "모델 기반 고위험군 조기 식별 시 치료 변경" 시나리오를 비교하고, 예측 불확실성을 정량화. 단, 실제 인과적 효과 추정이 아닌 모델 예측 비교 수준임을 명시.[^1] | 일부 구현 | 두 시나리오 비교는 대시보드에 구현. 예측 불확실성 정량화는 미구현 |
| 3-3 | 환자 맞춤형 치료 경로 제시 시스템 프로토타입 | 구현함 | Streamlit 대시보드로 구현 (`app.py`) |

[^1]: 계획서 원문에서는 실제 환자 30명, 1.5억원 예산의 전향적 코호트 연구로 설계되었으나, 본 PoC는 이를 20명 규모의 synthetic 데이터로 축소 재현한 것입니다.

---

## 아키텍처

```
generate_patients_timeseries.py
        │   20명 x 4시점(T0/T1/T2/T3) synthetic 시계열 생성
        │   (ctDNA VAF, methylation score, radiomics score, AFP, CTNNB1_mutation)
        ▼
patients_timeseries.csv
        │
        ▼
train_response_model.py
        │   - T0~T2 raw 값 + T0→T1 / T1→T2 변화율 feature 엔지니어링
        │   - XGBoost 분류기 학습 + Leave-One-Out CV(LOOCV) 평가
        │   - 전체 데이터로 재학습한 최종 모델에 SHAP(TreeExplainer) 적용
        ▼
model.pkl / shap_values.pkl / shap_summary_plot.png / shap_waterfall_*.png
        │
        ▼
app.py (Streamlit)
        │   - 환자별 시계열 시각화, 12주 반응 확률 예측, SHAP waterfall
        │   - "현재 치료 유지" vs "조기 치료 전환 가정" 시나리오 비교
        ▼
   의사용 인터랙티브 대시보드
```

---

## 실행 방법

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib streamlit plotly
```

```bash
python3 generate_patients_timeseries.py   # 1) synthetic 데이터 생성 -> patients_timeseries.csv
python3 train_response_model.py           # 2) 모델 학습 + LOOCV + SHAP -> model.pkl, shap_values.pkl
streamlit run app.py                      # 3) 대시보드 실행
```

> 참고(macOS): xgboost는 OpenMP 런타임(libomp)이 필요합니다. `pip install xgboost`만으로 로드 오류가 나는 경우 `conda install -c conda-forge xgboost`로 설치하면 libomp 의존성이 함께 해결됩니다.

---

## 주요 결과 (synthetic data 기준)

- **LOOCV AUROC: 0.833** (synthetic data 기준 — 실제 임상적 예측 성능을 의미하지 않음)
- 20명 중 반응군 6명(30%), 비반응군 14명(70%) (synthetic data 생성 규칙에 의해 목표 반응률 30%로 고정 배정)
- SHAP summary plot 기준 중요 변수 상위 항목 (synthetic data 기준):
  1. `vaf_change_T1_T2` (T1→T2 구간 ctDNA VAF 변화율) — 가장 큰 기여도
  2. `methylation_score_T2` (T2 시점 methylation score)
  3. `methylation_score_change_T1_T2`
  4. `radiomics_score_change_T1_T2`
  - baseline(T0) 원값 단독 변수들과 `ctnnb1_mutation`은 상대적으로 기여도가 작게 나타남 (synthetic data 기준)
- 가장 예측이 애매했던 환자(LOOCV 예측확률이 0.5에 가장 근접, synthetic data 기준): P019 (predicted proba ≈ 0.504)

---

## 데이터 생성 로직의 근거

`generate_patients_timeseries.py`의 synthetic 데이터는 다음 가정을 단순 규칙으로 반영합니다.

- **CTNNB1-cold tumor 가정**: CTNNB1(베타카테닌 경로) 활성화 돌연변이를 가진 HCC는 면역학적으로 "cold tumor"(T세포 침윤이 적은 종양미세환경) 경향을 보이며, 면역관문억제제 기반 치료에 대한 반응률이 낮다고 보고된 문헌(Ogawa et al., 2022 등)을 단순화하여 참고했습니다. 이 PoC에서는 이를 (1) CTNNB1 변이 양성 환자의 반응군 배정 확률을 낮추고, (2) CTNNB1 양성 반응군의 VAF 감소 속도를 느리게 만드는 두 가지 규칙으로만 반영했습니다. 문헌의 정량적 수치를 그대로 사용한 것이 아니라 방향성만 단순화한 것입니다.
- 그 외 VAF/methylation/radiomics/AFP의 반응군-비반응군 간 궤적 차이(지수적 감소 vs 유지/증가)는 임상적으로 통용되는 일반적 패턴을 참고해 임의로 설계한 것이며, 특정 문헌의 수치를 인용한 것은 아닙니다.

---

## 한계점

**원 연구계획서 항목 6(연구의 한계)과 일관된 한계**
- 표본크기: 계획서 자체가 소규모 코호트를 전제로 설계됨
- 단일기관 설계
- 단기 추적 관찰 기간

**이 PoC 자체의 추가 한계**
- 모든 데이터가 synthetic이므로 실제 임상적 검증이 원천적으로 불가능함
- 실제 연구비 집행이나 IRB 승인, 임상 코호트 운영 없이 개인이 사이드 프로젝트로 확장한 것으로, 공식 연구의 후속 산출물이 아님
- 학습/평가에 사용된 환자 수가 20명(그마저도 synthetic)으로, LOOCV AUROC 등 지표는 통계적으로 매우 불안정함
- CTNNB1-cold tumor 등 문헌 기반 가정은 방향성만 단순화하여 반영했을 뿐, 정량적으로 검증되지 않음
- 외부 검증 데이터셋이 없고, 데이터 생성 규칙 자체가 모델 성능에 순환적으로 유리하게 작용할 수 있음(synthetic 데이터를 생성한 규칙과 유사한 패턴을 같은 사람이 모델링했기 때문)
- "조기 치료 전환" 시나리오 비교는 실제 인과추론이 아닌, 학습된 모델에 대한 단순 입력값 조작 기반 탐색적 비교임

---

## 향후 계획

- 실제(비식별화된) HCC 환자 코호트 확보 시, 동일한 파이프라인 구조(데이터 → 모델 → SHAP → 대시보드)에 실제 데이터를 연결하여 재검증
- 실제 어세이 기반 ctDNA VAF / methylation / radiomics 데이터 확보 및 전처리 파이프라인 구축 (계획서 세부목표 1-1, 1-2에 해당)
- 실제 데이터 기반 모델 재학습 및 외부 기관 데이터를 활용한 외부 검증(external validation) 추가
- 시나리오 비교 섹션을 단순 규칙 기반이 아닌, 인과추론 방법론(예: target trial emulation, causal forest 등)으로 고도화
- IRB 승인 및 공식 연구 절차를 통한 전향적(prospective) 코호트 검증 (실행 시 별도 공식 연구로 진행되어야 함)
