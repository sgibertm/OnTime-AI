"""
home.py — OnTime AI · Home Dashboard
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from Logic.utils_ml import *

TMB_RED = "#CE1126"


def home_page():
    
    data = load_data()

    ROUTES = build_routes(data)

    st.title("Home Dashboard")
    st.write("Tell me about your destination and a bus station. I'll use Supervised Learning to predict your timming of arrival!")
    st.divider()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Journey Parameters")
        st.divider()

        route_names = list(ROUTES.keys())
        route_name = st.selectbox("Metro Line", route_names)

        route_data = ROUTES[route_name]
        stop = st.selectbox("Stop", route_data["stops"])

        day = st.selectbox("Day of Week", DAYS)

        model_choice = st.selectbox("Algorithm", ["Linear Regression", "Random Forest"])

        # Persist the selected algorithm in session state so the sidebar can read it
        st.session_state["model_choice"] = model_choice

        st.markdown("")
        predict_buton = st.button("Predict Delay")

    with col2:
        st.subheader("Calculation")
        st.divider()

        if predict_buton:

            st.subheader("Input summary")

            st.write("Route:", route_name)
            st.write("Stop:", stop)
            st.write("Day:", day)
            st.write("Model:", model_choice)

            mask = (
                data["stop_name"].str.contains(stop, case=False, na=False) &
                (data["route_short_name"] == route_name.split()[0])
            )

            subset = data[mask]

            if subset.empty:
                st.error("Stop not found in dataset")

            elif model_choice == "Linear Regression":
                #pred, x_line, y_line, rank, model = run_prediction(subset, data)
                # ── Linear Regression path ─────────────────────────────────────
                pred_seconds, feature_coefficients, model, mae = run_prediction(subset, data)

                minutes = pred_seconds / 60

                st.metric(
                    label="Estimated travel time",
                    value=f"{minutes:.1f} min"
                )

                # Top-8 absolute coefficients as feature importance chart
                sorted_fc = sorted(feature_coefficients.items(), key=lambda x: x[1], reverse=True)[:8]
                feat_names  = [item[0] for item in sorted_fc]
                feat_scores = [item[1] for item in sorted_fc]

                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.barh(feat_names[::-1], feat_scores[::-1], color=TMB_RED)
                ax.set_xlabel("Absolute coefficient")
                ax.set_title("Top-8 Feature Coefficients")
                ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
                fig.tight_layout()

                st.pyplot(fig)

                st.caption(f"Mean Absolute Error (cv=3): ±{mae/60:.1f} min")
                """
                st.metric(
                    label="Predicted delay (min)",
                    value=f"{pred:.2f}"
                )

                fig, ax = plt.subplots()

                ax.scatter(
                    data["stop_rank_normalized"],
                    data["delay_minutes"],
                    s=2,
                    alpha=0.2
                )

                ax.plot(x_line, y_line, color="red", linewidth=2)

                ax.set_xlabel("Stop rank")
                ax.set_ylabel("Delay (min)")
                ax.set_title("Stop position vs delay")

                st.pyplot(fig)
                """

            else:
                # ── Random Forest path ─────────────────────────────────────────
                with st.spinner("Training Random Forest…"):
                    pred_seconds, feature_importances, model, mae = run_random_forest_regressor(subset, data)
                # Estimated time probability metric
                minutes = pred_seconds / 60

                st.metric(
                    label="Estimated travel time",
                    value=f"{minutes:.1f} min"
                )

                # Top-8 feature importances — horizontal bar chart
                sorted_fi = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:8]
                feat_names = [item[0] for item in sorted_fi]
                feat_scores = [item[1] for item in sorted_fi]

                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.barh(feat_names[::-1], feat_scores[::-1], color=TMB_RED)
                ax.set_xlabel("Importance")
                ax.set_title("Top-8 Feature Importances")
                ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
                fig.tight_layout()

                st.pyplot(fig)

                # Cross-validated accuracy caption
                st.caption(f"Mean Absolute Error (cv=5): ±{mae/60:.1f} min")
                
        else:
            st.info("Need to choose a Metro Line, the station you are located at and the day of the week before seeing the results.")
