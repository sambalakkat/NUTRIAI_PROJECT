"""
7-day meal planner for NutriAI clinical personas.

Builds diverse meal plans from filtered USDA snapshot data.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
import pandas as pd

from filtering import apply_clinical_filters, load_snapshot

ENERGY_COL = "Energy"
PROTEIN_COL = "Protein"
FAT_COL = "Total lipid (fat)"
CARB_COL = "Carbohydrate, by difference"
FIBER_COL = "Fiber, total dietary"

MEAL_TYPES = ["Breakfast", "Lunch", "Dinner"]
MEAL_SPLITS = {"Breakfast": 0.28, "Lunch": 0.34, "Dinner": 0.38}
DAYS = list(range(1, 8))

MAX_USES_PER_FOOD = 2
MAX_PORTION_GRAMS = 300.0
MIN_PORTION_GRAMS = 100.0
TOP_CANDIDATE_POOL = 1000
CALORIE_TOLERANCE = 0.10

NON_MEAL_PATTERN = re.compile(
    r"\b(?:seasoning|salt|spice mix|bouillon|extract|flavoring|food coloring|"
    r"leavening|baking powder|baking soda|yeast|starch|cornstarch|"
    r"shortening|lard|oil spray|flour|powder|matrix|protein supplement)\b"
)


def _prepare_food_pool(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Score and rank foods using vectorized nutritional-density metrics."""
    pool = filtered_df.copy()

    required = ["description", "fdc_id", ENERGY_COL, PROTEIN_COL, FAT_COL, CARB_COL]
    missing = [col for col in required if col not in pool.columns]
    if missing:
        raise ValueError(f"filtered_df is missing required columns: {missing}")

    pool = pool.dropna(subset=[ENERGY_COL, "description"])
    pool = pool[pool[ENERGY_COL] > 0]

    protein = pool[PROTEIN_COL].fillna(0)
    fiber = pool[FIBER_COL].fillna(0) if FIBER_COL in pool.columns else 0
    energy = pool[ENERGY_COL]

    # Keep realistic whole-food style entries (exclude concentrates/seasonings).
    pool = pool[
        (energy >= 40)
        & (energy <= 450)
        & (protein <= 35)
        & (~pool["description"].str.contains(NON_MEAL_PATTERN, regex=True, na=False))
    ]

    if pool.empty:
        raise ValueError("filtered_df has no usable foods after realistic-meal filtering.")

    protein_arr = protein.loc[pool.index].to_numpy(dtype=float)
    fiber_arr = (
        fiber.loc[pool.index].to_numpy(dtype=float)
        if isinstance(fiber, pd.Series)
        else np.zeros(len(pool), dtype=float)
    )
    energy_arr = energy.loc[pool.index].to_numpy(dtype=float)

    protein_density = protein_arr / energy_arr
    fiber_density = fiber_arr / energy_arr
    pool = pool.copy()
    pool["density_score"] = 0.60 * protein_density + 0.40 * fiber_density

    return pool.sort_values("density_score", ascending=False).reset_index(drop=True)


def _nutrients_for_portion(row: pd.Series, grams: float) -> dict[str, float]:
    scale = grams / 100.0
    return {
        "Calories": float(row[ENERGY_COL]) * scale,
        "Protein": float(row[PROTEIN_COL]) * scale if pd.notna(row[PROTEIN_COL]) else 0.0,
        "Fat": float(row[FAT_COL]) * scale if pd.notna(row[FAT_COL]) else 0.0,
        "Carbs": float(row[CARB_COL]) * scale if pd.notna(row[CARB_COL]) else 0.0,
    }


def _eligible_mask(
    pool: pd.DataFrame,
    usage_counts: dict[Any, int],
) -> np.ndarray:
    fdc_ids = pool["fdc_id"].to_numpy()
    return np.array([usage_counts.get(fdc_id, 0) < MAX_USES_PER_FOOD for fdc_id in fdc_ids])


def _rotated_candidates(
    pool: pd.DataFrame,
    eligible: np.ndarray,
    slot_index: int,
) -> np.ndarray:
    if eligible.size == 0:
        return eligible

    top = eligible[: min(TOP_CANDIDATE_POOL, eligible.size)]
    offset = slot_index % top.size
    return np.concatenate([top[offset:], top[:offset]])


def _pick_portion_grams(
    energy_per_100g: float,
    desired_calories: float,
    max_calories: float,
) -> float:
    if desired_calories <= 0 or max_calories <= 0:
        return 0.0

    target_calories = min(desired_calories, max_calories)
    grams = (target_calories / energy_per_100g) * 100.0
    grams = min(MAX_PORTION_GRAMS, grams)

    if grams < MIN_PORTION_GRAMS:
        return 0.0
    return grams


def _append_meal_item(
    plan_rows: list[dict[str, Any]],
    usage_counts: dict[Any, int],
    day: int,
    meal_type: str,
    row: pd.Series,
    grams: float,
) -> dict[str, float]:
    nutrients = _nutrients_for_portion(row, grams)
    plan_rows.append(
        {
            "Day": day,
            "Meal_Type": meal_type,
            "fdc_id": row["fdc_id"],
            "Food_Description": row["description"],
            "Calories": round(nutrients["Calories"], 1),
            "Protein": round(nutrients["Protein"], 1),
            "Fat": round(nutrients["Fat"], 1),
            "Carbs": round(nutrients["Carbs"], 1),
        }
    )
    fdc_id = row["fdc_id"]
    usage_counts[fdc_id] = usage_counts.get(fdc_id, 0) + 1
    return nutrients


def _build_plan_summary(
    plan_df: pd.DataFrame,
    daily_calorie_target: int,
    usage_counts: dict[Any, int],
) -> dict[str, Any]:
    max_daily_calories = daily_calorie_target * (1.0 + CALORIE_TOLERANCE)

    daily_totals = (
        plan_df.groupby("Day", as_index=False)[["Calories", "Protein", "Fat", "Carbs"]]
        .sum()
        .round(1)
    )
    daily_totals["calorie_target"] = daily_calorie_target
    daily_totals["max_allowed_calories"] = round(max_daily_calories, 1)
    daily_totals["within_target"] = daily_totals["Calories"] <= max_daily_calories

    reuse_by_fdc = pd.Series(list(usage_counts.values()))
    return {
        "daily_calorie_target": daily_calorie_target,
        "max_daily_calories": round(max_daily_calories, 1),
        "daily_macro_averages": {
            "Calories": round(float(daily_totals["Calories"].mean()), 1),
            "Protein": round(float(daily_totals["Protein"].mean()), 1),
            "Fat": round(float(daily_totals["Fat"].mean()), 1),
            "Carbs": round(float(daily_totals["Carbs"].mean()), 1),
        },
        "daily_totals": daily_totals.to_dict(orient="records"),
        "unique_foods_used": int(len(usage_counts)),
        "max_food_reuse": int(reuse_by_fdc.max()) if not reuse_by_fdc.empty else 0,
    }


def _usage_counts_from_plan(plan_df: pd.DataFrame, exclude_day: int | None = None) -> dict[Any, int]:
    usage_counts: dict[Any, int] = {}
    subset = plan_df if exclude_day is None else plan_df[plan_df["Day"] != exclude_day]

    if "fdc_id" not in subset.columns:
        return usage_counts

    for fdc_id in subset["fdc_id"].dropna():
        usage_counts[fdc_id] = usage_counts.get(fdc_id, 0) + 1
    return usage_counts


def _generate_day_meals(
    pool: pd.DataFrame,
    day: int,
    daily_calorie_target: int,
    usage_counts: dict[Any, int],
    slot_index: int,
) -> list[dict[str, Any]]:
    max_daily_calories = daily_calorie_target * (1.0 + CALORIE_TOLERANCE)
    day_calories = 0.0
    plan_rows: list[dict[str, Any]] = []

    for meal_type in MEAL_TYPES:
        meal_target = daily_calorie_target * MEAL_SPLITS[meal_type]
        meal_calories = 0.0
        items_added = 0
        max_items_per_meal = 2
        meal_fdc_ids: set[Any] = set()

        while items_added < max_items_per_meal:
            remaining_meal = meal_target - meal_calories
            remaining_day = max_daily_calories - day_calories
            if remaining_meal <= 30 or remaining_day <= 30:
                break

            eligible = _eligible_mask(pool, usage_counts)
            eligible_idx = np.flatnonzero(eligible)
            candidates = _rotated_candidates(pool, eligible_idx, slot_index + items_added)

            added = False
            for idx in candidates:
                row = pool.iloc[int(idx)]
                if row["fdc_id"] in meal_fdc_ids:
                    continue

                desired = remaining_meal if items_added == 0 else remaining_meal * 0.85
                grams = _pick_portion_grams(
                    float(row[ENERGY_COL]),
                    desired,
                    remaining_day,
                )
                if grams <= 0:
                    continue

                nutrients = _nutrients_for_portion(row, grams)
                if day_calories + nutrients["Calories"] > max_daily_calories:
                    grams = _pick_portion_grams(
                        float(row[ENERGY_COL]),
                        remaining_day,
                        remaining_day,
                    )
                    if grams <= 0:
                        continue
                    nutrients = _nutrients_for_portion(row, grams)
                    if day_calories + nutrients["Calories"] > max_daily_calories:
                        continue

                nutrients = _append_meal_item(
                    plan_rows, usage_counts, day, meal_type, row, grams
                )
                meal_fdc_ids.add(row["fdc_id"])
                meal_calories += nutrients["Calories"]
                day_calories += nutrients["Calories"]
                items_added += 1
                added = True
                break

            if not added:
                break

        slot_index += 1

    while day_calories < daily_calorie_target * 0.97:
        remaining_day = max_daily_calories - day_calories
        if remaining_day <= 30:
            break

        eligible = _eligible_mask(pool, usage_counts)
        candidates = _rotated_candidates(pool, np.flatnonzero(eligible), slot_index)
        added = False

        for idx in candidates[:80]:
            row = pool.iloc[int(idx)]
            grams = _pick_portion_grams(
                float(row[ENERGY_COL]),
                daily_calorie_target - day_calories,
                remaining_day,
            )
            if grams <= 0:
                continue

            nutrients = _nutrients_for_portion(row, grams)
            if day_calories + nutrients["Calories"] > max_daily_calories:
                continue

            nutrients = _append_meal_item(
                plan_rows, usage_counts, day, "Dinner", row, grams
            )
            day_calories += nutrients["Calories"]
            added = True
            break

        if not added:
            break
        slot_index += 1

    return plan_rows


def swap_meals_for_day(
    filtered_df: pd.DataFrame,
    plan_df: pd.DataFrame,
    day: int,
    daily_calorie_target: int = 2000,
    slot_index: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace all meals for one day with newly generated alternatives."""
    if day not in DAYS:
        raise ValueError(f"Day must be between 1 and 7, received {day}.")

    pool = _prepare_food_pool(filtered_df)
    usage_counts = _usage_counts_from_plan(plan_df, exclude_day=day)

    if slot_index is None:
        slot_index = int(time.time() * 1000) % 997

    new_day_rows = _generate_day_meals(
        pool, day, daily_calorie_target, usage_counts, slot_index
    )

    remaining = plan_df[plan_df["Day"] != day]
    updated_plan = pd.concat(
        [remaining, pd.DataFrame(new_day_rows)],
        ignore_index=True,
    )

    meal_order = pd.CategoricalDtype(categories=MEAL_TYPES, ordered=True)
    updated_plan["Meal_Type"] = updated_plan["Meal_Type"].astype(meal_order)
    updated_plan = updated_plan.sort_values(["Day", "Meal_Type"]).reset_index(drop=True)
    updated_plan["Meal_Type"] = updated_plan["Meal_Type"].astype(str)

    full_usage = _usage_counts_from_plan(updated_plan)
    summary = _build_plan_summary(updated_plan, daily_calorie_target, full_usage)
    return updated_plan, summary


def generate_7_day_plan(
    filtered_df: pd.DataFrame,
    daily_calorie_target: int = 2000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generate a 7-day meal plan with Breakfast, Lunch, and Dinner.

    Uses greedy selection on nutritionally dense foods while enforcing
    diversity (max 2 uses per food) and a daily calorie ceiling of +10%.
    """
    pool = _prepare_food_pool(filtered_df)
    usage_counts: dict[Any, int] = {}
    plan_rows: list[dict[str, Any]] = []
    slot_index = 0

    for day in DAYS:
        day_rows = _generate_day_meals(
            pool, day, daily_calorie_target, usage_counts, slot_index
        )
        plan_rows.extend(day_rows)
        slot_index += len(MEAL_TYPES) + 1

    plan_df = pd.DataFrame(
        plan_rows,
        columns=[
            "Day",
            "Meal_Type",
            "fdc_id",
            "Food_Description",
            "Calories",
            "Protein",
            "Fat",
            "Carbs",
        ],
    )

    summary = _build_plan_summary(plan_df, daily_calorie_target, usage_counts)
    return plan_df, summary


if __name__ == "__main__":
    persona = "IBS + vegetarian"
    snapshot = load_snapshot()
    filtered_df, exclusion_report = apply_clinical_filters(snapshot, persona)

    start = time.perf_counter()
    plan, summary = generate_7_day_plan(filtered_df, daily_calorie_target=2000)
    elapsed = time.perf_counter() - start

    print(f"Persona: {persona}")
    print(f"Filtered pool: {len(filtered_df)} foods")
    print(f"Generated plan in {elapsed:.2f}s")
    print(f"Meals planned: {len(plan)}")
    print(f"Unique foods: {summary['unique_foods_used']}")
    print(f"Max reuse: {summary['max_food_reuse']} (limit {MAX_USES_PER_FOOD})")
    print("\nDaily macro averages:")
    for macro, value in summary["daily_macro_averages"].items():
        print(f"  {macro}: {value}")

    print("\nDaily totals:")
    print(pd.DataFrame(summary["daily_totals"]).to_string(index=False))

    print("\nSample meals (Day 1):")
    print(plan[plan["Day"] == 1].to_string(index=False))
