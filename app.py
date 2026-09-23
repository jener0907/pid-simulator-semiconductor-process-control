import csv
import io
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

plt.rcParams["font.family"] = "Malgun Gothic" if sys.platform == "win32" else "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="온도 제어 시뮬레이터", layout="wide")
st.title("온도 제어 시뮬레이터")
st.caption("FOPDT 공정에서 온오프, P, PID 제어를 비교합니다. 왼쪽 값을 바꾸면 그래프와 결과가 즉시 갱신됩니다.")

with st.sidebar:
    st.header("공정 설정")
    setpoint = st.number_input("설정 온도 (℃)", value=1000.0, step=10.0)
    ambient = st.number_input("초기·주변 온도 (℃)", value=25.0, step=1.0)
    tau = st.number_input("시정수 τ (s)", min_value=1.0, value=300.0, step=10.0)
    gain = st.number_input("공정 이득 K", value=16.0, step=1.0)
    delay = st.number_input("시간 지연 (s)", min_value=0, value=30, step=1)
    total_time = st.number_input("총 시뮬레이션 시간 (s)", min_value=60, value=5400, step=60)
    dt = st.number_input("시간 간격 (s)", min_value=1, value=1, step=1)

    st.header("제어기 설정")
    band = st.number_input("온오프 히스테리시스 ±(℃)", min_value=0.0, value=5.0, step=1.0)
    kp = st.number_input("P/PID 비례 이득 Kc", min_value=0.0, value=0.3, step=0.05, format="%.2f")
    ti = st.number_input("PID 적분 시간 τI (s)", min_value=1.0, value=200.0, step=10.0)
    td = st.number_input("PID 미분 시간 τD (s)", min_value=0.0, value=25.0, step=1.0)

    st.header("그래프 표시")
    show_onoff = st.checkbox("온오프 온도", value=True)
    show_p = st.checkbox("P 온도", value=True)
    show_pid = st.checkbox("PID 온도", value=True)
    show_setpoint = st.checkbox("설정값", value=True)
    show_heater = st.checkbox("온오프 히터 출력", value=True)

times = np.arange(int(total_time // dt) + 1, dtype=float) * dt
count = len(times)
delay_steps = round(delay / dt)


def simulate(mode):
    temperature = np.zeros(count)
    output = np.zeros(count)
    temperature[0] = ambient
    previous_output = 0.0
    integral = 0.0
    previous_error = setpoint - ambient

    for k in range(count - 1):
        error = setpoint - temperature[k]
        if mode == "온오프":
            if temperature[k] < setpoint - band:
                output[k] = 100.0
            elif temperature[k] > setpoint + band:
                output[k] = 0.0
            else:
                output[k] = previous_output
            previous_output = output[k]
        elif mode == "P":
            output[k] = np.clip(kp * error, 0.0, 100.0)
        else:
            derivative = (error - previous_error) / dt
            candidate_integral = integral + error * dt
            candidate = kp * (error + candidate_integral / ti + td * derivative)
            if 0.0 <= candidate <= 100.0:
                integral = candidate_integral
                output[k] = candidate
            else:
                output[k] = np.clip(kp * (error + integral / ti + td * derivative), 0.0, 100.0)
            previous_error = error

        delayed_output = output[k - delay_steps] if k >= delay_steps else 0.0
        temperature[k + 1] = temperature[k] + dt / tau * (-(temperature[k] - ambient) + gain * delayed_output)

    output[-1] = output[-2] if count > 1 else output[0]
    return temperature, output


onoff_t, onoff_u = simulate("온오프")
p_t, p_u = simulate("P")
pid_t, pid_u = simulate("PID")
minutes = times / 60.0

graph_tab, table_tab = st.tabs(["그래프 화면", "수치표"])

with graph_tab:
    chart1, chart2 = st.columns(2)
    with chart1:
        st.subheader("1. 온오프 전체")
        fig, ax = plt.subplots(figsize=(10, 4))
        if show_onoff:
            ax.plot(minutes, onoff_t, color="#e76f51", label="온도")
        if show_setpoint:
            ax.axhline(setpoint, color="#264653", linestyle="--", label="설정값")
        ax.set(xlabel="시간 (분)", ylabel="온도 (℃)")
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with chart2:
        st.subheader("2. 온오프 확대 (40~70분)")
        fig, ax = plt.subplots(figsize=(10, 4))
        window = (minutes >= 40) & (minutes <= 70)
        if show_onoff:
            ax.plot(minutes[window], onoff_t[window], color="#e76f51", label="온도")
            ax.axhline(setpoint - band, color="#888", linestyle="--", linewidth=1, label="히스테리시스")
            ax.axhline(setpoint + band, color="#888", linestyle="--", linewidth=1)
        ax.set(xlabel="시간 (분)", ylabel="온도 (℃)")
        ax.grid(alpha=0.25)
        if show_setpoint:
            ax.axhline(setpoint, color="#264653", linestyle="--", label="설정값")
        if show_heater:
            ax2 = ax.twinx()
            ax2.step(minutes[window], onoff_u[window], where="post", color="#2a9d8f", label="히터 출력")
            ax2.set(ylabel="히터 출력 (%)", ylim=(-5, 105))
        lines = ax.get_lines() + (ax2.get_lines() if show_heater else [])
        if lines:
            ax.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("3. 세 제어기 온도 비교")
    fig, ax = plt.subplots(figsize=(14, 5))
    if show_onoff:
        ax.plot(minutes, onoff_t, label="온오프", linewidth=1)
    if show_p:
        ax.plot(minutes, p_t, label="P", linewidth=1.5)
    if show_pid:
        ax.plot(minutes, pid_t, label="PID", linewidth=1.5)
    if show_setpoint:
        ax.axhline(setpoint, color="#555", linestyle="--", label="설정값")
    ax.set(xlabel="시간 (분)", ylabel="온도 (℃)", ylim=(750, 1100))
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    last = times >= total_time - 1800
    st.subheader("시뮬레이션 결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("온오프 마지막 30분 진동 폭", f"{np.ptp(onoff_t[last]):.2f} ℃")
    c2.metric("P 잔류편차", f"{setpoint - p_t[-1]:.2f} ℃", f"최종 온도 {p_t[-1]:.2f} ℃")
    c3.metric("PID 최종 오차", f"{setpoint - pid_t[-1]:.6f} ℃", f"최종 온도 {pid_t[-1]:.2f} ℃")

with table_tab:
    st.subheader("시간별 시뮬레이션 수치")
    st.dataframe({
        "시간 (s)": times,
        "시간 (분)": minutes,
        "온오프 온도 (℃)": onoff_t,
        "온오프 히터 출력 (%)": onoff_u,
        "P 온도 (℃)": p_t,
        "P 히터 출력 (%)": p_u,
        "PID 온도 (℃)": pid_t,
        "PID 히터 출력 (%)": pid_u,
    }, hide_index=True, height=600, use_container_width=True)

    csv_data = io.StringIO(newline="")
    writer = csv.writer(csv_data)
    settings = [setpoint, ambient, tau, gain, delay, total_time, dt, band, kp, ti, td]
    writer.writerow(["time_s", "time_min", "onoff_temp_C", "onoff_heater_pct", "p_temp_C", "p_heater_pct", "pid_temp_C", "pid_heater_pct", "setpoint_C", "ambient_C", "tau_s", "gain", "delay_s", "total_time_s", "dt_s", "hysteresis_C", "Kc", "tauI_s", "tauD_s"])
    writer.writerows([t, t / 60, ot, ou, pt, pu, it, iu, *settings] for t, ot, ou, pt, pu, it, iu in zip(times, onoff_t, onoff_u, p_t, p_u, pid_t, pid_u))
    st.download_button("CSV로 결과 저장", csv_data.getvalue().encode("utf-8-sig"), file_name="temperature_control_simulation.csv", mime="text/csv")
