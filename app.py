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

PROCESS_PRESETS = {
    "반도체 열처리로 (기본)": {"setpoint": 1000.0, "ambient": 25.0, "tau": 300.0, "gain": 16.0, "delay": 30},
    "웨이퍼 핫플레이트 (가정)": {"setpoint": 250.0, "ambient": 25.0, "tau": 120.0, "gain": 8.0, "delay": 5},
    "CVD 챔버 가열 (가정)": {"setpoint": 400.0, "ambient": 25.0, "tau": 240.0, "gain": 10.0, "delay": 20},
    "열교환기 출구 온도 (가정)": {"setpoint": 120.0, "ambient": 25.0, "tau": 180.0, "gain": 6.0, "delay": 15},
    "보일러 출구 온도 (가정)": {"setpoint": 180.0, "ambient": 25.0, "tau": 600.0, "gain": 12.0, "delay": 45},
}
DEFAULTS = {
    "setpoint": 1000.0, "ambient": 25.0, "tau": 300.0, "gain": 16.0, "delay": 30,
    "total_time": 5400, "dt": 1, "band": 5.0, "kp": 0.3, "ti": 200.0, "td": 25.0,
    "disturbance": 0.0, "disturbance_start": 2700, "noise_std": 0.0, "heater_slew": 0.0,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("process_preset", "반도체 열처리로 (기본)")


def apply_process_preset():
    for key, value in PROCESS_PRESETS[st.session_state.process_preset].items():
        st.session_state[key] = value


with st.sidebar:
    with st.expander("공정 설정", expanded=False):
        st.selectbox("공정 사례 (가정값)", list(PROCESS_PRESETS), key="process_preset", on_change=apply_process_preset)
        st.caption("사례별 수치는 실측값이 아닌 교육용 FOPDT 가정값입니다. 불러온 값은 직접 수정할 수 있습니다.")
        setpoint = st.number_input("설정 온도 (℃)", step=10.0, key="setpoint")
        ambient = st.number_input("초기·주변 온도 (℃)", step=1.0, key="ambient")
        tau = st.number_input("시정수 τ (s)", min_value=1.0, step=10.0, key="tau")
        gain = st.number_input("공정 이득 K", step=1.0, key="gain")
        delay = st.number_input("시간 지연 (s)", min_value=0, step=1, key="delay")
        total_time = st.number_input("총 시뮬레이션 시간 (s)", min_value=60, step=60, key="total_time")
        dt = st.number_input("시간 간격 (s)", min_value=1, step=1, key="dt")

    with st.expander("제어기 설정", expanded=False):
        band = st.number_input("온오프 히스테리시스 ±(℃)", min_value=0.0, step=1.0, key="band")
        kp = st.number_input("P/PID 비례 이득 Kc", min_value=0.0, step=0.05, format="%.2f", key="kp")
        ti = st.number_input("PID 적분 시간 τI (s)", min_value=1.0, step=10.0, key="ti")
        td = st.number_input("PID 미분 시간 τD (s)", min_value=0.0, step=1.0, key="td")

    with st.expander("시험 조건", expanded=False):
        disturbance = st.number_input("열부하 외란 (등가 ℃)", step=10.0, key="disturbance")
        disturbance_start = st.number_input("외란 시작 시간 (s)", min_value=0, step=60, key="disturbance_start")
        noise_std = st.number_input("센서 잡음 표준편차 (℃)", min_value=0.0, step=0.1, key="noise_std")
        heater_slew = st.number_input("히터 변화속도 제한 (%/s, 0=제한 없음)", min_value=0.0, step=1.0, key="heater_slew")

    with st.expander("그래프 표시", expanded=False):
        show_onoff = st.checkbox("온오프 온도", value=True, key="show_onoff")
        show_p = st.checkbox("P 온도", value=True, key="show_p")
        show_pid = st.checkbox("PID 온도", value=True, key="show_pid")
        show_setpoint = st.checkbox("설정값", value=True, key="show_setpoint")
        show_heater = st.checkbox("온오프 히터 출력", value=True, key="show_heater")

process_preset = st.session_state.process_preset
times = np.arange(int(total_time // dt) + 1, dtype=float) * dt
count = len(times)
delay_steps = round(delay / dt)
sensor_noise = np.random.default_rng(0).normal(0.0, noise_std, count) if noise_std else np.zeros(count)


def simulate(mode):
    temperature = np.zeros(count)
    output = np.zeros(count)
    temperature[0] = ambient
    previous_output = 0.0
    integral = 0.0
    previous_error = setpoint - ambient - sensor_noise[0]

    for k in range(count - 1):
        error = setpoint - (temperature[k] + sensor_noise[k])
        if mode == "온오프":
            if temperature[k] + sensor_noise[k] < setpoint - band:
                command = 100.0
            elif temperature[k] + sensor_noise[k] > setpoint + band:
                command = 0.0
            else:
                command = previous_output
            integrate = False
        elif mode == "P":
            command = np.clip(kp * error, 0.0, 100.0)
            integrate = False
        else:
            derivative = (error - previous_error) / dt
            candidate_integral = integral + error * dt
            candidate = kp * (error + candidate_integral / ti + td * derivative)
            if 0.0 <= candidate <= 100.0:
                command = candidate
                integrate = True
            else:
                command = np.clip(kp * (error + integral / ti + td * derivative), 0.0, 100.0)
                integrate = False
            previous_error = error

        if heater_slew > 0.0:
            max_change = heater_slew * dt
            output[k] = np.clip(command, previous_output - max_change, previous_output + max_change)
        else:
            output[k] = command
        output[k] = np.clip(output[k], 0.0, 100.0)
        if mode == "PID" and integrate and np.isclose(output[k], candidate):
            integral = candidate_integral
        previous_output = output[k]

        delayed_output = output[k - delay_steps] if k >= delay_steps else 0.0
        disturbance_now = disturbance if times[k] >= disturbance_start else 0.0
        temperature[k + 1] = temperature[k] + dt / tau * (-(temperature[k] - ambient) + gain * delayed_output + disturbance_now)

    output[-1] = output[-2] if count > 1 else output[0]
    return temperature, output


onoff_t, onoff_u = simulate("온오프")
p_t, p_u = simulate("P")
pid_t, pid_u = simulate("PID")
minutes = times / 60.0


def response_metrics(temperature, output):
    step = setpoint - ambient
    direction = np.sign(step) or 1.0
    overshoot = max(0.0, float(np.max(direction * (temperature - setpoint))))
    overshoot_pct = 100.0 * overshoot / abs(step) if step else 0.0
    projected = direction * (temperature - ambient)
    reached_10 = np.flatnonzero(projected >= 0.1 * abs(step))
    reached_90 = np.flatnonzero(projected >= 0.9 * abs(step))
    rise_time = times[reached_90[0]] - times[reached_10[0]] if len(reached_10) and len(reached_90) else None
    tolerance = max(abs(step) * 0.02, 0.1)
    outside = np.flatnonzero(np.abs(temperature - setpoint) > tolerance)
    settling_time = times[outside[-1] + 1] if len(outside) and outside[-1] < count - 1 else (times[0] if not len(outside) else None)
    error = np.abs(setpoint - temperature)
    iae = np.sum((error[:-1] + error[1:]) * 0.5 * np.diff(times))
    heater_use = np.sum((output[:-1] + output[1:]) * 0.5 * np.diff(times))
    heater_movement = np.sum(np.abs(np.diff(output)))
    return overshoot_pct, rise_time, settling_time, iae, heater_use, heater_movement


metrics = {
    "온오프": response_metrics(onoff_t, onoff_u),
    "P": response_metrics(p_t, p_u),
    "PID": response_metrics(pid_t, pid_u),
}

graph_tab, table_tab = st.tabs(["그래프 화면", "수치표"])

with graph_tab:
    chart1, chart2 = st.columns(2)
    with chart1:
        st.subheader("1. 온오프 전체")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        plot_onoff_1 = ax1.plot([], [], color="#e76f51", label="온도")[0] if show_onoff else None
        if show_setpoint:
            ax1.axhline(setpoint, color="#264653", linestyle="--", label="설정값")
        ax1.set(xlabel="시간 (분)", ylabel="온도 (℃)", xlim=(0, minutes[-1]))
        ypad = max(float(np.ptp(onoff_t)) * 0.05, 1.0)
        ax1.set_ylim(float(np.min(onoff_t)) - ypad, float(np.max(onoff_t)) + ypad)
        ax1.grid(alpha=0.25)
        if ax1.lines:
            ax1.legend(fontsize=8)
        fig1.tight_layout()
        plot1 = st.empty()

    with chart2:
        st.subheader("2. 온오프 확대 (40~70분)")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        zoom = (minutes >= 40) & (minutes <= 70)
        zoom_indices = np.flatnonzero(zoom)
        plot_onoff_2 = ax2.plot([], [], color="#e76f51", label="온도")[0] if show_onoff else None
        if show_onoff:
            ax2.axhline(setpoint - band, color="#888", linestyle="--", linewidth=1, label="히스테리시스")
            ax2.axhline(setpoint + band, color="#888", linestyle="--", linewidth=1)
        ax2.set(xlabel="시간 (분)", ylabel="온도 (℃)", xlim=(40, 70))
        ax2.grid(alpha=0.25)
        if show_setpoint:
            ax2.axhline(setpoint, color="#264653", linestyle="--", label="설정값")
        ax2r = ax2.twinx()
        plot_heater = ax2r.plot([], [], color="#2a9d8f", label="히터 출력")[0] if show_heater else None
        ax2r.set(ylabel="히터 출력 (%)", ylim=(-5, 105))
        zoom_data = onoff_t[zoom] if len(zoom_indices) else np.array([ambient, setpoint])
        zoom_pad = max(float(np.ptp(zoom_data)) * 0.1, 1.0)
        ax2.set_ylim(float(np.min(zoom_data)) - zoom_pad, float(np.max(zoom_data)) + zoom_pad)
        lines = ax2.get_lines() + (ax2r.get_lines() if show_heater else [])
        if lines:
            ax2.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=8)
        fig2.tight_layout()
        plot2 = st.empty()

    st.subheader("3. 세 제어기 온도 비교")
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    plot_lines = []
    for visible, label, values, color in (
        (show_onoff, "온오프", onoff_t, "#4c9be8"),
        (show_p, "P", p_t, "#ff8c42"),
        (show_pid, "PID", pid_t, "#2ca02c"),
    ):
        line = ax3.plot([], [], label=label, color=color, linewidth=1.2)[0] if visible else None
        plot_lines.append(line)
    if show_setpoint:
        ax3.axhline(setpoint, color="#555", linestyle="--", label="설정값")
    ax3.set(xlabel="시간 (분)", ylabel="온도 (℃)", xlim=(0, minutes[-1]))
    shown = [values for visible, values in ((show_onoff, onoff_t), (show_p, p_t), (show_pid, pid_t)) if visible]
    shown.extend([np.array([ambient, setpoint])])
    low = min(float(np.min(values)) for values in shown)
    high = max(float(np.max(values)) for values in shown)
    pad = max((high - low) * 0.05, 1.0)
    if process_preset == "반도체 열처리로 (기본)" and setpoint == 1000 and ambient == 25 and disturbance == 0:
        ax3.set_ylim(750, 1100)
    else:
        ax3.set_ylim(low - pad, high + pad)
    ax3.grid(alpha=0.25)
    if ax3.lines:
        ax3.legend(fontsize=9)
    fig3.tight_layout()
    plot3 = st.empty()

    def draw_plots():
        x = minutes
        if plot_onoff_1 is not None:
            plot_onoff_1.set_data(x, onoff_t)
        if plot_onoff_2 is not None:
            plot_onoff_2.set_data(minutes[zoom_indices], onoff_t[zoom_indices])
        if plot_heater is not None:
            plot_heater.set_data(np.repeat(minutes[zoom_indices], 2)[1:], np.repeat(onoff_u[zoom_indices], 2)[:-1])
        for line, values in zip(plot_lines, (onoff_t, p_t, pid_t)):
            if line is not None:
                line.set_data(x, values)
        plot1.pyplot(fig1, width="stretch")
        plot2.pyplot(fig2, width="stretch")
        plot3.pyplot(fig3, width="stretch")

    draw_plots()
    for fig in (fig1, fig2, fig3):
        plt.close(fig)

    last = times >= total_time - 1800
    st.subheader("시뮬레이션 결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("온오프 마지막 30분 진동 폭", f"{np.ptp(onoff_t[last]):.2f} ℃")
    c2.metric("P 잔류편차", f"{setpoint - p_t[-1]:.2f} ℃", f"최종 온도 {p_t[-1]:.2f} ℃")
    c3.metric("PID 최종 오차", f"{setpoint - pid_t[-1]:.6f} ℃", f"최종 온도 {pid_t[-1]:.2f} ℃")

with table_tab:
    st.subheader("제어 성능 지표")
    metric_labels = ["최대 오버슈트 (%)", "상승시간 10–90% (s)", "정착시간 ±2% (s)", "IAE (℃·s)", "히터 사용량 (%·s)", "히터 총 변화량 (%)"]
    metric_rows = list(zip(*metrics.values()))
    metric_values = {
        "지표": metric_labels,
        "온오프": [f"{v:.2f}" if v is not None else "미도달" for v in metric_rows[0]],
        "P": [f"{v:.2f}" if v is not None else "미도달" for v in metric_rows[1]],
        "PID": [f"{v:.2f}" if v is not None else "미도달" for v in metric_rows[2]],
    }
    st.dataframe(metric_values, hide_index=True, width="stretch")
    st.caption("IAE는 목표 온도 오차의 절대값 누적량입니다. 외란이 있으면 원래 설정값 기준 정착시간이 미도달일 수 있습니다.")

    st.subheader("시간별 시뮬레이션 수치")
    disturbance_trace = np.where(times >= disturbance_start, disturbance, 0.0)
    st.dataframe({
        "시간 (s)": times,
        "시간 (분)": minutes,
        "센서 잡음 (℃)": sensor_noise,
        "열부하 외란 (등가 ℃)": disturbance_trace,
        "온오프 온도 (℃)": onoff_t,
        "온오프 히터 출력 (%)": onoff_u,
        "P 온도 (℃)": p_t,
        "P 히터 출력 (%)": p_u,
        "PID 온도 (℃)": pid_t,
        "PID 히터 출력 (%)": pid_u,
    }, hide_index=True, height=600, width="stretch")

    csv_data = io.StringIO(newline="")
    writer = csv.writer(csv_data)
    settings = [setpoint, ambient, tau, gain, delay, total_time, dt, band, kp, ti, td, disturbance_start, disturbance, noise_std, heater_slew, process_preset]
    writer.writerow(["time_s", "time_min", "sensor_noise_C", "disturbance_equiv_C", "onoff_temp_C", "onoff_heater_pct", "p_temp_C", "p_heater_pct", "pid_temp_C", "pid_heater_pct", "setpoint_C", "ambient_C", "tau_s", "gain", "delay_s", "total_time_s", "dt_s", "hysteresis_C", "Kc", "tauI_s", "tauD_s", "disturbance_start_s", "disturbance_C", "sensor_noise_std_C", "heater_slew_pct_s", "process_preset"])
    writer.writerows([t, t / 60, noise, d, ot, ou, pt, pu, it, iu, *settings] for t, noise, d, ot, ou, pt, pu, it, iu in zip(times, sensor_noise, disturbance_trace, onoff_t, onoff_u, p_t, p_u, pid_t, pid_u))
    st.download_button("CSV로 결과 저장", csv_data.getvalue().encode("utf-8-sig"), file_name="temperature_control_simulation.csv", mime="text/csv")
