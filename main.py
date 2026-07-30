
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Hippo 경로 · MAPK 우회 모델링", layout="wide")

# ----------------------------------------------------------------------------
# 1. 모델 정의
# ----------------------------------------------------------------------------
#
# [기본 모델] MST -> LATS* -> YAP_n 억제, 단일 캐스케이드
#   d[LATS*]/dt = k1 * MST * (LATS_tot - LATS*) - d1 * LATS*
#   d[YAP_n]/dt = k_syn - k2 * LATS* * YAP_n - d2 * YAP_n
#
# [MAPK 우회 모델] 위 두 식에 MAPK 활성화 변수를 추가.
#   MST 자극과 독립적으로 활성화되는 MAPK(예: ERK)가 LATS를 거치지 않고
#   YAP_n을 직접 인산화/격리한다고 가정 (우회 경로).
#   d[MAPK*]/dt = k3 * S_mapk * (MAPK_tot - MAPK*) - d3 * MAPK*
#   d[YAP_n]/dt = k_syn - k2*LATS**YAP_n - k4*MAPK**YAP_n - d2*YAP_n
#
# k4 = 0 이면 MAPK 우회 모델은 기본 모델과 완전히 일치한다 (검증 포인트).

def base_model(y, t, k1, d1, k2, d2, k_syn, MST, LATS_tot):
    LATS_star, YAP_n = y
    dLATS = k1 * MST * (LATS_tot - LATS_star) - d1 * LATS_star
    dYAP = k_syn - k2 * LATS_star * YAP_n - d2 * YAP_n
    return [dLATS, dYAP]


def mapk_bypass_model(y, t, k1, d1, k2, d2, k_syn, MST, LATS_tot,
                       k3, d3, S_mapk, MAPK_tot, k4):
    LATS_star, YAP_n, MAPK_star = y
    dLATS = k1 * MST * (LATS_tot - LATS_star) - d1 * LATS_star
    dMAPK = k3 * S_mapk * (MAPK_tot - MAPK_star) - d3 * MAPK_star
    dYAP = k_syn - k2 * LATS_star * YAP_n - k4 * MAPK_star * YAP_n - d2 * YAP_n
    return [dLATS, dYAP, dMAPK]


def run_base(params, t):
    y0 = [0.0, params["YAP_n0"]]
    sol = odeint(base_model, y0, t, args=(
        params["k1"], params["d1"], params["k2"], params["d2"],
        params["k_syn"], params["MST"], params["LATS_tot"]
    ))
    return sol  # columns: LATS*, YAP_n


def run_mapk(params, t):
    y0 = [0.0, params["YAP_n0"], 0.0]
    sol = odeint(mapk_bypass_model, y0, t, args=(
        params["k1"], params["d1"], params["k2"], params["d2"],
        params["k_syn"], params["MST"], params["LATS_tot"],
        params["k3"], params["d3"], params["S_mapk"], params["MAPK_tot"], params["k4"]
    ))
    return sol  # columns: LATS*, YAP_n, MAPK*


# ----------------------------------------------------------------------------
# 2. 사이드바 - 파라미터 입력
# ----------------------------------------------------------------------------
st.sidebar.header("공통 파라미터")
MST = st.sidebar.slider("MST 자극 강도 (MST)", 0.0, 5.0, 1.0, 0.1)
LATS_tot = st.sidebar.slider("LATS 총량 (LATS_tot)", 0.1, 5.0, 1.0, 0.1)
k1 = st.sidebar.slider("k1 (MST → LATS* 활성화 속도)", 0.0, 5.0, 1.0, 0.1)
d1 = st.sidebar.slider("d1 (LATS* 비활성화 속도)", 0.01, 3.0, 0.5, 0.01)
k2 = st.sidebar.slider("k2 (LATS* → YAP_n 억제 속도)", 0.0, 5.0, 1.0, 0.1)
d2 = st.sidebar.slider("d2 (YAP_n 자연 분해 속도)", 0.01, 3.0, 0.3, 0.01)
k_syn = st.sidebar.slider("k_syn (YAP_n 합성/유입 속도)", 0.0, 5.0, 1.0, 0.1)
YAP_n0 = st.sidebar.slider("YAP_n 초기값", 0.0, 3.0, 0.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("MAPK 우회항 파라미터")
k4 = st.sidebar.slider(
    "k4 (MAPK* → YAP_n 우회 억제 속도) ⭐ 핵심",
    0.0, 5.0, 1.0, 0.1,
    help="k4 = 0이면 우회 모델이 기본 모델과 완전히 동일해집니다."
)
S_mapk = st.sidebar.slider("S_mapk (MAPK 활성 자극 강도)", 0.0, 5.0, 1.0, 0.1)
MAPK_tot = st.sidebar.slider("MAPK 총량 (MAPK_tot)", 0.1, 5.0, 1.0, 0.1)
k3 = st.sidebar.slider("k3 (MAPK 활성화 속도)", 0.0, 5.0, 1.0, 0.1)
d3 = st.sidebar.slider("d3 (MAPK* 비활성화 속도)", 0.01, 3.0, 0.5, 0.01)

st.sidebar.markdown("---")
t_max = st.sidebar.slider("시뮬레이션 시간 범위", 10, 200, 50, 10)
n_points = 1000

params = dict(
    k1=k1, d1=d1, k2=k2, d2=d2, k_syn=k_syn, MST=MST, LATS_tot=LATS_tot,
    YAP_n0=YAP_n0, k3=k3, d3=d3, S_mapk=S_mapk, MAPK_tot=MAPK_tot, k4=k4
)

t = np.linspace(0, t_max, n_points)
sol_base = run_base(params, t)
sol_mapk = run_mapk(params, t)

# ----------------------------------------------------------------------------
# 3. 본문
# ----------------------------------------------------------------------------
st.title("Hippo 경로 동역학: 기본 모델 vs MAPK 우회 모델")

st.markdown(
    """
MST가 억제되어 정상적인 Hippo 경로가 차단되더라도, 세포는 MAPK(ERK) 경로를 통해
LATS를 거치지 않고 YAP을 직접 조절하는 우회로를 가질 수 있다는 가설을 수식으로 표현했습니다.
왼쪽 슬라이더로 각 반응 속도 상수를 조절하면서, 우회항(k4)이 YAP_n의 정상상태(steady state)에
미치는 영향을 관찰할 수 있습니다.
"""
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("① 기본 모델 (MAPK 제외)")
    st.latex(r"\frac{d[LATS^*]}{dt} = k_1 [MST]([LATS]_{tot}-[LATS^*]) - d_1 [LATS^*]")
    st.latex(r"\frac{d[YAP_n]}{dt} = k_{syn} - k_2 [LATS^*][YAP_n] - d_2 [YAP_n]")

with col2:
    st.subheader("② MAPK 우회항 추가 모델")
    st.latex(r"\frac{d[MAPK^*]}{dt} = k_3 S_{mapk}([MAPK]_{tot}-[MAPK^*]) - d_3 [MAPK^*]")
    st.latex(r"\frac{d[YAP_n]}{dt} = k_{syn} - k_2 [LATS^*][YAP_n] - k_4 [MAPK^*][YAP_n] - d_2 [YAP_n]")

st.markdown("---")

# --- 그래프 1: YAP_n 시간에 따른 변화 비교 ---
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=t, y=sol_base[:, 1], name="YAP_n (기본 모델)",
                           line=dict(color="#4C78A8", width=3)))
fig1.add_trace(go.Scatter(x=t, y=sol_mapk[:, 1], name="YAP_n (MAPK 우회 모델)",
                           line=dict(color="#E45756", width=3, dash="dash")))
fig1.update_layout(
    title="핵 내 활성 YAP 농도([YAP_n]) 시간 변화 비교",
    xaxis_title="시간 (t)",
    yaxis_title="[YAP_n] 농도",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=450,
)
st.plotly_chart(fig1, use_container_width=True)

# --- 그래프 2: LATS*, MAPK* 비교 ---
fig2 = make_subplots(specs=[[{"secondary_y": False}]])
fig2.add_trace(go.Scatter(x=t, y=sol_base[:, 0], name="LATS* (기본 모델)",
                           line=dict(color="#54A24B", width=2)))
fig2.add_trace(go.Scatter(x=t, y=sol_mapk[:, 0], name="LATS* (우회 모델)",
                           line=dict(color="#54A24B", width=2, dash="dot")))
fig2.add_trace(go.Scatter(x=t, y=sol_mapk[:, 2], name="MAPK* (우회 모델)",
                           line=dict(color="#F58518", width=2)))
fig2.update_layout(
    title="상류 인산화 효소 활성 비교 (LATS*, MAPK*)",
    xaxis_title="시간 (t)",
    yaxis_title="활성형 농도",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=400,
)
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------------------
# 4. 정상상태 비교 및 정량 지표
# ----------------------------------------------------------------------------
st.markdown("---")
st.subheader("정상상태(steady-state) 정량 비교")

yap_base_ss = sol_base[-1, 1]
yap_mapk_ss = sol_mapk[-1, 1]
diff = yap_mapk_ss - yap_base_ss
diff_pct = (diff / yap_base_ss * 100) if yap_base_ss != 0 else float("nan")

m1, m2, m3 = st.columns(3)
m1.metric("기본 모델 YAP_n 정상상태", f"{yap_base_ss:.4f}")
m2.metric("MAPK 우회 모델 YAP_n 정상상태", f"{yap_mapk_ss:.4f}", f"{diff:+.4f}")
m3.metric("우회항으로 인한 변화율", f"{diff_pct:+.2f}%")

st.markdown(
    """
**해석 가이드**
- `k4 = 0`으로 설정하면 두 모델의 YAP_n 곡선이 완전히 겹칩니다 (모델 축소 검증).
- `k4`를 증가시키면 MAPK가 LATS와 무관하게 YAP_n을 추가로 억제하므로, MST가 낮아
  Hippo 경로 자체는 느슨해진 상황에서도 YAP_n 정상상태가 더 낮게 유지될 수 있습니다.
- 즉 MAPK 우회항은 "Hippo 경로가 꺼져도 YAP이 완전히 자유로워지지 않는" 상황,
  즉 두 경로가 YAP 억제에 대해 이중 안전장치(redundant control)로 작동하는 상황을 표현합니다.
"""
)

# ----------------------------------------------------------------------------
# 5. k4 민감도 스캔
# ----------------------------------------------------------------------------
st.markdown("---")
st.subheader("k4 (MAPK 우회 강도) 민감도 스캔")
st.caption("다른 모든 파라미터는 왼쪽 슬라이더 값으로 고정한 채, k4만 변화시켜 YAP_n 정상상태를 스캔합니다.")

k4_range = np.linspace(0, 5, 40)
yap_ss_scan = []
for k4_val in k4_range:
    p = dict(params)
    p["k4"] = k4_val
    sol = run_mapk(p, t)
    yap_ss_scan.append(sol[-1, 1])

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=k4_range, y=yap_ss_scan, mode="lines+markers",
                           line=dict(color="#B279A2", width=3)))
fig3.add_hline(y=yap_base_ss, line_dash="dash", line_color="#4C78A8",
               annotation_text="기본 모델 YAP_n 정상상태", annotation_position="top left")
fig3.add_vline(x=k4, line_dash="dot", line_color="gray",
               annotation_text="현재 k4", annotation_position="top right")
fig3.update_layout(
    title="k4 값에 따른 YAP_n 정상상태 변화",
    xaxis_title="k4 (MAPK 우회 억제 속도)",
    yaxis_title="YAP_n 정상상태 값",
    height=420,
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown(
    """
---
### 모델링 노트
- 두 모델 모두 `scipy.integrate.odeint`로 수치 적분했습니다.
- 기본 모델은 2변수([LATS*], [YAP_n]), 우회 모델은 3변수([LATS*], [YAP_n], [MAPK*]) 시스템입니다.
- MAPK 활성화 항의 형태를 LATS 활성화 항과 동일한 구조(효소-기질 포화형, 1차 분해)로 둔 것은
  두 경로가 같은 인산화 반응 문법을 공유한다고 가정했기 때문이며, 이후 탐구에서 MAPK 특유의
  피드백(예: 음성 피드백에 의한 진동)을 추가로 반영해 구조를 분화시킬 수 있습니다.
"""
)
