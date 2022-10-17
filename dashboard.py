from pickletools import optimize
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from smartforecast import *

st.set_page_config(layout="wide")
st.markdown(
    f"""
            <style>
                .reportview-container .sidebar-content {{
                    padding-top: {1}rem;
                }}
                .reportview-container .main .block-container {{
                    padding-top: {1}rem;
                }}
            </style>
            """,
    unsafe_allow_html=True,
)
# ------------------------------ Title ------------------------------------------------
st.title("SmartForecasting")
# -------------------------------------------------------------------------------------
seasonality_list = []
# ------------------------------ Side Bar ---------------------------------------------
with st.sidebar:
    add_trend = st.checkbox("Activate trend", value=True)
    trend_mode = st.radio("Mode", ("linear", "exponential"))
    col1, col2 = st.columns(2)
    with col1:
        level = st.number_input("Level", value=0.0)
    with col2:
        coeff = st.number_input("Coefficient", value=10.0, step=0.1)

    trend = {
        "mode": "multiplicative",
        "coefficient": coeff,
        "type": trend_mode,
    }

    add_cyclic = st.checkbox("Activate Cyclicity", value=True)
    col1, col2 = st.columns(2)
    with col1:
        freq_1 = st.slider("Frequency", step=0.5, min_value=0.5, value=2.0)
    with col2:
        amp_1 = st.slider("Amplitude", 1, 10, 4)
    if add_cyclic:
        seasonality_list += [
            {
                "frequency": freq_1,
                "mode": "additive",
                "amplitude": amp_1,
                "phi": 0,
            }
        ]

    st.markdown("""---""")
    add_season_1 = st.checkbox("Activate 1st Seasonality", value=True)
    col1, col2 = st.columns(2)
    with col1:
        season_freq_1 = st.slider(
            "Frequency n°1", step=0.5, min_value=freq_1 + 0.5, value=8.0
        )
    with col2:
        season_amp_1 = st.slider("Amplitude n°1", 1, max_value=amp_1, value=2)
    season_mode_1 = st.radio("Type n°1", ("additive", "multiplicative"))
    if add_season_1:
        seasonality_list += [
            {
                "frequency": season_freq_1,
                "mode": season_mode_1,
                "amplitude": season_amp_1,
                "phi": 0,
            }
        ]

    st.markdown("""---""")
    add_season_2 = st.checkbox("Activate 2nd Seasonality", value=True)
    col1, col2 = st.columns(2)
    with col1:
        season_freq_2 = st.slider(
            "Frequency n°2", step=0.5, min_value=freq_1 + 0.5, value=8.0
        )
    with col2:
        season_amp_2 = st.slider("Amplitude n°2", 1, max_value=amp_1, value=3)
    season_mode_2 = st.radio("Type n°2", ("additive", "multiplicative"))
    if add_season_2:
        seasonality_list += [
            {
                "frequency": season_freq_2,
                "mode": season_mode_2,
                "amplitude": season_amp_2,
                "phi": 0,
            }
        ]

    st.markdown("""---""")
    st.header("Noise")
    add_noise = st.checkbox("Add noise", value=True)
    noise_std = st.slider("Amplitude", 0.1, 2.0, 1.25)
# -------------------------------------------------------------------------------------
print(seasonality_list)

forecast = None
timeseries = Timeseries(
    start=1,
    end=5,
    dt=0.005,
    trend=trend,
    seasonalities=seasonality_list,
    add_noise=add_noise,
    gaussian_noise={"mu": 0, "sigma": noise_std},
    level=level,
)


# ------------------------------ TABS --------------------------------------------------
Data, Analyse, Model = st.tabs(["TimeSeries", "Analyses", "Model"])

# ------------------------------ TimeSeries --------------------------------------------
with Data:
    fig_timeseries = go.Figure(
        data=go.Scatter(x=timeseries.time, y=timeseries.ts, name="Timeseries")
    )

    if st.button("Decompose"):
        fig_timeseries.add_trace(
            go.Scatter(
                x=timeseries.time,
                y=timeseries.generate_trend(),
                name="Trend",
            ),
        )
        i = 0
        for season in timeseries.seasonalities:
            i += 1
            fig_timeseries.add_trace(
                go.Scatter(
                    x=timeseries.time,
                    y=timeseries.generate_seasonality(
                        coeff=season["amplitude"],
                        frequency=season["frequency"],
                        phi=season["phi"],
                    ),
                    name=f"Seasonality {i}",
                )
            )
        if add_noise:
            fig_timeseries.add_trace(
                go.Scatter(
                    x=timeseries.time,
                    y=timeseries._noise,
                    name="Noise",
                ),
            )
    fig_timeseries.update_layout(
        title="Generated TimeSeries",
        xaxis_title="Time",
        yaxis_title="Value",
    )
    st.plotly_chart(fig_timeseries, use_container_width=True, sharing="streamlit")
# -------------------------------------------------------------------------------------

# ------------------------------ Analyses ----------------------------------------------
with Analyse:
    smoothing_window = st.slider(
        "Smoothing window", step=1, min_value=3, max_value=100, value=100
    )
    fig_timeseries_2 = go.Figure(
        data=go.Scatter(
            x=timeseries.time,
            y=timeseries.ts,
            name="Original Series",
        )
    )
    fig_timeseries_2.add_trace(
        go.Scatter(
            x=timeseries.time[smoothing_window - 1 :],
            y=timeseries.moving_average(w=smoothing_window),
            name="Moving average",
        ),
    )
    fig_timeseries_2.add_trace(
        go.Scatter(
            x=timeseries.time[smoothing_window - 1 :],
            y=timeseries.ts[smoothing_window - 1 :]
            - timeseries.moving_average(w=smoothing_window)
            + np.mean(timeseries.ts),
            name="Detrended Timeseries",
        ),
    )
    fig_timeseries_2.update_layout(
        title="Trend Analysis",
        xaxis_title="Time",
        yaxis_title="Value",
    )
    st.plotly_chart(fig_timeseries_2, use_container_width=True, sharing="streamlit")

    fig_autocorr = go.Figure(
        data=go.Scatter(
            x=np.arange(0, len(timeseries.autocorrelation), timeseries.dt),
            y=timeseries.autocorrelation,
        )
    )
    fig_autocorr.update_layout(
        title="Time Autocorrelation Function (ACF)",
        xaxis_title="Lag",
        yaxis_title="Correlation",
    )
    st.plotly_chart(fig_autocorr, use_container_width=True, sharing="streamlit")
    x, y = timeseries.periodogram()
    fig_periodogram = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
        )
    )
    fig_periodogram.update_layout(
        title="Periodogram",
        xaxis_title="Period",
        yaxis_title="Intensity",
    )
    st.plotly_chart(fig_periodogram, use_container_width=True, sharing="streamlit")
# -------------------------------------------------------------------------------------

# ------------------------------ Holt Winters ------------------------------------------
with Model:
    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        st.subheader("Model")
        horizon = st.slider("Forecast horizon", 1, len(timeseries.time), 200)
        alpha = st.slider("alpha", 0.0, 1.0, 0.5)
        beta = st.slider("beta", 0.0, 1.0, 0.5)
        gamma = st.slider("gamma", 0.0, 1.0, 0.5)

    with col2:
        st.subheader("Timeseries")
        season_mode = st.radio("Season", ("additive", "multiplicative"))
        n_occ = (1 / freq_1) / timeseries.dt
        seasonal_period_1 = st.number_input(
            "Period",
            min_value=2,
            max_value=1000,
            step=1,
            value=int(n_occ),
        )

        st.write(seasonal_period_1)
        if st.button("Run"):
            model = ExponentialSmoothing(
                timeseries.ts,
                trend="additive",
                seasonal=season_mode,
                seasonal_periods=n_occ,
            ).fit(
                smoothing_level=alpha,
                smoothing_trend=beta,
                smoothing_seasonal=gamma,
                optimized=False,
                use_brute=True,
            )
            forecast = model.forecast(horizon)

        fig_forecast = go.Figure(
            data=go.Scatter(x=timeseries.time, y=timeseries.ts, name="History")
        )
        with col3:
            st.subheader("Forecast")
            if forecast is not None:
                fig_forecast.add_trace(
                    go.Scatter(
                        x=np.arange(
                            timeseries.time[-1] + timeseries.dt,
                            timeseries.time[-1] + horizon * timeseries.dt,
                            timeseries.dt,
                        ),
                        y=forecast,
                        name="Forecast",
                    )
                )
            fig_forecast.update_layout(
                title="Forecast",
                xaxis_title="Time",
                yaxis_title="Value",
            )
            st.plotly_chart(fig_forecast, use_container_width=True, sharing="streamlit")
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
