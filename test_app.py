from streamlit.testing.v1 import AppTest


def smoke_test():
    app = AppTest.from_file("app.py").run(timeout=60)
    assert not app.exception

    preset = app.selectbox(key="process_preset")
    preset.set_value(preset.options[1])
    app.number_input(key="total_time").set_value(60)
    app.number_input(key="disturbance").set_value(-10.0)
    app.number_input(key="noise_std").set_value(0.2)
    app.number_input(key="heater_slew").set_value(5.0)
    app.run(timeout=60)

    assert not app.exception
    assert app.number_input(key="setpoint").value == 250.0
    assert len(app.dataframe) == 2


if __name__ == "__main__":
    smoke_test()
    print("PASS: preset, disturbance, noise, rate limit, and metrics")
