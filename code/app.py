"""
NutriAI Streamlit frontend.

Loads the offline snapshot, applies clinical filters, and generates a 7-day meal plan.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from filtering import PERSONAS, apply_clinical_filters, load_snapshot
from meal_planner import generate_7_day_plan, swap_meals_for_day

PERSONA_OPTIONS = [
    "IBS + vegetarian",
    "GERD + gluten-free",
    "T2 diabetes + vegan",
    "hypertension + pescatarian",
]

DAY_OPTIONS = [f"Day {day}" for day in range(1, 8)]
DISPLAY_COLUMNS = ["Day", "Meal_Type", "Food_Description", "Calories", "Protein", "Fat", "Carbs"]

st.set_page_config(page_title="NutriAI Meal Planner", page_icon="🥗", layout="wide")
st.title("NutriAI Clinical Meal Planner")
st.caption("Generate a personalized 7-day meal plan from the USDA offline snapshot.")

if "feedback_log" not in st.session_state:
    st.session_state["feedback_log"] = []

with st.sidebar:
    st.header("Persona Settings")
    selected_persona = st.selectbox(
        "Select clinical persona",
        options=PERSONA_OPTIONS,
        index=0,
    )
    daily_calorie_target = st.number_input(
        "Daily calorie target",
        min_value=1200,
        max_value=3500,
        value=2000,
        step=50,
    )
    generate_clicked = st.button("Generate Meal Plan", type="primary", use_container_width=True)

if generate_clicked:
    try:
        start_time = time.time()

        snapshot = load_snapshot()
        filtered_df, why_excluded = apply_clinical_filters(snapshot, selected_persona)
        plan_df, summary = generate_7_day_plan(filtered_df, daily_calorie_target=daily_calorie_target)

        elapsed_seconds = time.time() - start_time

        st.session_state["plan_df"] = plan_df
        st.session_state["summary"] = summary
        st.session_state["why_excluded"] = why_excluded
        st.session_state["elapsed_seconds"] = elapsed_seconds
        st.session_state["persona"] = selected_persona
        st.session_state["filtered_df"] = filtered_df
        st.session_state["daily_calorie_target"] = daily_calorie_target
        st.session_state["swap_count"] = 0
    except Exception as exc:
        st.error(f"Failed to generate meal plan: {exc}")

if "plan_df" in st.session_state:
    elapsed = st.session_state["elapsed_seconds"]
    persona = st.session_state["persona"]
    why_excluded = st.session_state["why_excluded"]
    plan_df = st.session_state["plan_df"]
    summary = st.session_state["summary"]
    display_plan = plan_df[DISPLAY_COLUMNS]

    st.success(f"Meal plan generated for **{persona}**")
    st.metric(
        label="Total execution time",
        value=f"{elapsed:.2f}s",
        help="Includes loading snapshot, clinical filtering, and 7-day plan generation.",
    )

    if elapsed < 60:
        st.info("Sub-60s generation target met.")

    st.subheader("Why Excluded?")
    st.write(
        f"**{why_excluded['excluded_count']:,}** of **{why_excluded['initial_count']:,}** foods "
        f"were excluded for **{why_excluded['persona']}** "
        f"({why_excluded['diet']} diet + {why_excluded['condition']})."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Foods passed filter", f"{why_excluded['final_count']:,}")
    col2.metric("Foods excluded", f"{why_excluded['excluded_count']:,}")
    col3.metric("Unique foods in plan", summary["unique_foods_used"])

    exclusion_counts = why_excluded.get("exclusion_reason_counts", {})
    if exclusion_counts:
        exclusion_df = pd.DataFrame(
            [{"Reason": reason, "Count": count} for reason, count in exclusion_counts.items()]
        )
        st.dataframe(exclusion_df, use_container_width=True, hide_index=True)
    else:
        st.write("No exclusion reasons recorded.")

    with st.expander("Excluded food samples"):
        samples = why_excluded.get("excluded_samples", [])
        if samples:
            sample_rows = []
            for item in samples:
                sample_rows.append(
                    {
                        "fdc_id": item.get("fdc_id"),
                        "description": item.get("description"),
                        "reasons": "; ".join(item.get("reasons", [])),
                    }
                )
            st.dataframe(pd.DataFrame(sample_rows), use_container_width=True, hide_index=True)
        else:
            st.write("No excluded samples available.")

    st.subheader("Daily Macro Averages")
    averages = summary["daily_macro_averages"]
    avg_cols = st.columns(4)
    avg_cols[0].metric("Calories", f"{averages['Calories']:.1f}")
    avg_cols[1].metric("Protein (g)", f"{averages['Protein']:.1f}")
    avg_cols[2].metric("Fat (g)", f"{averages['Fat']:.1f}")
    avg_cols[3].metric("Carbs (g)", f"{averages['Carbs']:.1f}")

    daily_totals_df = pd.DataFrame(summary["daily_totals"])
    st.dataframe(daily_totals_df, use_container_width=True, hide_index=True)

    st.subheader("7-Day Meal Plan")
    st.dataframe(display_plan, use_container_width=True, hide_index=True)

    csv_data = display_plan.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download meal plan as CSV",
        data=csv_data,
        file_name=f"nutriai_meal_plan_{persona.replace(' ', '_').replace('+', 'plus')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("User Feedback & Adjustments")

    feedback_col1, feedback_col2 = st.columns([2, 1])
    with feedback_col1:
        selected_day_label = st.selectbox(
            "Select day to swap meals",
            options=DAY_OPTIONS,
            index=0,
            key="swap_day_select",
        )
    with feedback_col2:
        plan_rating = st.selectbox(
            "Rate this meal plan",
            options=["Not rated", "1 - Poor", "2 - Fair", "3 - Good", "4 - Very Good", "5 - Excellent"],
            index=0,
            key="plan_rating",
        )

    swap_clicked = st.button("Swap Meals for this Day", use_container_width=False)

    if swap_clicked:
        try:
            swap_day = int(selected_day_label.split()[-1])
            updated_plan, updated_summary = swap_meals_for_day(
                filtered_df=st.session_state["filtered_df"],
                plan_df=st.session_state["plan_df"],
                day=swap_day,
                daily_calorie_target=int(st.session_state["daily_calorie_target"]),
            )
            st.session_state["plan_df"] = updated_plan
            st.session_state["summary"] = updated_summary
            st.session_state["swap_count"] = st.session_state.get("swap_count", 0) + 1
            st.success(f"Swapped all meals for {selected_day_label}. Daily macros recalculated.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not swap meals: {exc}")

    general_feedback = st.text_area(
        "General Feedback",
        placeholder="Tell us what you liked, what you'd change, or any dietary notes...",
        height=120,
        key="general_feedback_text",
    )

    submit_clicked = st.button("Submit Feedback", type="primary")

    if submit_clicked:
        feedback_entry = {
            "persona": persona,
            "rating": plan_rating,
            "feedback": general_feedback.strip(),
            "swap_count": st.session_state.get("swap_count", 0),
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.session_state["feedback_log"].append(feedback_entry)
        st.success("Feedback submitted successfully! Thank you for helping improve NutriAI.")

    if st.session_state["feedback_log"]:
        with st.expander("Submitted feedback history"):
            st.dataframe(
                pd.DataFrame(st.session_state["feedback_log"]),
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info("Select a persona in the sidebar and click **Generate Meal Plan** to begin.")

    st.markdown("### Supported personas")
    for persona_name, config in PERSONAS.items():
        st.markdown(f"- **{persona_name}** — {config['diet']} + {config['condition']}")
