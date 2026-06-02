"""
Clinical filtering pipeline for NutriAI offline snapshot.

Applies diet and medical-condition exclusions for four user personas.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT = PROJECT_ROOT / "data" / "offline_snapshot.csv"

CARB_COL = "Carbohydrate, by difference"
FIBER_COL = "Fiber, total dietary"

# ---------------------------------------------------------------------------
# Persona registry
# ---------------------------------------------------------------------------

PERSONAS = {
    "IBS + vegetarian": {"condition": "IBS", "diet": "vegetarian"},
    "GERD + gluten-free": {"condition": "GERD", "diet": "gluten-free"},
    "T2 diabetes + vegan": {"condition": "T2 diabetes", "diet": "vegan"},
    "hypertension + pescatarian": {"condition": "hypertension", "diet": "pescatarian"},
}

# ---------------------------------------------------------------------------
# Diet keyword lists (matched against lowercase description)
# ---------------------------------------------------------------------------

MEAT_POULTRY_KEYWORDS = [
    "beef", "pork", "lamb", "veal", "steak", "bacon", "ham", "sausage",
    "salami", "pepperoni", "prosciutto", "chorizo", "hot dog", "frankfurter",
    "chicken", "turkey", "duck", "goose", "quail", "cornish hen", "poultry",
    "venison", "bison", "buffalo", "rabbit", "game meat", "meat", "liver",
    "kidney", "tripe", "giblets", "lard", "suety", "jerky", "paté", "pate",
    "blood sausage", "oxtail", "short ribs", "brisket", "meatball", "meat loaf",
]

FISH_SEAFOOD_KEYWORDS = [
    "fish", "salmon", "tuna", "cod", "haddock", "halibut", "trout", "sardine",
    "anchovy", "anchovies", "mackerel", "herring", "tilapia", "catfish",
    "shrimp", "prawn", "crab", "lobster", "crayfish", "crawfish", "oyster",
    "mussel", "clam", "scallop", "squid", "calamari", "octopus", "caviar",
    "roe", "seafood", "shellfish", "surimi", "fish oil",
]

DAIRY_EGG_KEYWORDS = [
    "milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "whey", "casein",
    "ghee", "paneer", "ricotta", "mozzarella", "cheddar", "parmesan", "brie",
    "feta", "gouda", "swiss cheese", "cottage cheese", "ice cream", "custard",
    "half-and-half", "buttermilk", "sour cream", "cream cheese", "egg", "eggs",
    "albumen", "mayonnaise", "mayo", "meringue", "hollandaise", "lactose",
]

GLUTEN_KEYWORDS = [
    "wheat", "gluten", "barley", "rye", "spelt", "semolina", "bulgur", "farro",
    "triticale", "malt", "seitan", "couscous", "pasta", "noodle", "macaroni",
    "bread", "flour", "cracker", "cookie", "biscuit", "cake", "pastry", "bagel",
    "muffin", "croissant", "pita", "tortilla", "wheat bran", "wheat germ",
    "beer", "ale", "lager", "stout", "soy sauce", "teriyaki",
]

# ---------------------------------------------------------------------------
# Medical condition keyword lists
# ---------------------------------------------------------------------------

IBS_FODMAP_KEYWORDS = [
    "onion", "garlic", "leek", "shallot", "scallion", "chive", "wheat",
    "apple", "pear", "mango", "watermelon", "cherry", "plum", "peach",
    "apricot", "nectarine", "honey", "agave", "high fructose", "inulin",
    "chicory", "cauliflower", "mushroom", "asparagus", "artichoke",
    "beans", "lentil", "chickpea", "garbanzo", "legume", "black bean",
    "kidney bean", "pinto bean", "split pea", "soybean", "edamame",
    "cashew", "pistachio", "soft cheese", "cream cheese", "ricotta",
    "milk", "yogurt", "yoghurt", "sorbitol", "mannitol", "xylitol",
]

GERD_TRIGGER_KEYWORDS = [
    "tomato", "marinara", "salsa", "ketchup", "citrus", "orange", "lemon",
    "lime", "grapefruit", "tangerine", "mandarin", "spicy", "chili", "chilli",
    "jalapeño", "jalapeno", "cayenne", "hot sauce", "sriracha", "curry",
    "chocolate", "cocoa", "coffee", "espresso", "mint", "peppermint",
    "alcohol", "wine", "beer", "liquor", "vinegar", "pickle", "fried",
    "deep-fried", "greasy", "fatty", "onion", "garlic", "pepperoni",
    "carbonated", "soda", "cola", "energy drink",
]

HYPERTENSION_SODIUM_KEYWORDS = [
    "salt", "salted", "soy sauce", "pickle", "pickled", "cured", "brine",
    "brined", "bacon", "ham", "salami", "pepperoni", "prosciutto", "anchovy",
    "anchovies", "bouillon", "broth", "stock cube", "monosodium glutamate",
    "msg", "pretzel", "chips", "crisps", "sauerkraut", "olives", "cheese",
    "processed cheese", "smoked", "jerky", "tamari", "miso", "sodium",
    "corned", "hot dog", "frankfurter", "ramen", "instant soup",
]

T2_DIABETES_SUGAR_KEYWORDS = [
    "sugar", "syrup", "honey", "molasses", "candy", "soda", "cola",
    "sweetened", "dessert", "cake", "cookie", "pastry", "donut", "doughnut",
    "jam", "jelly", "marmalade", "frosting", "icing", "sucrose", "fructose",
    "corn syrup", "high fructose", "sweetener", "syrupy", "pie filling",
    "chocolate", "caramel", "toffee", "sherbet", "sorbet", "popsicle",
]

# T2 diabetes macro thresholds (USDA snapshot values are per 100 g)
T2_MAX_NET_CARBS_G = 10.0
T2_MAX_TOTAL_CARBS_G = 15.0
T2_MAX_ENERGY_KCAL = 250.0


def _normalize_description(description: Any) -> str:
    if description is None or (isinstance(description, float) and pd.isna(description)):
        return ""
    return str(description).lower()


def _find_keyword_matches(description: str, keywords: list[str]) -> list[str]:
    """Return keywords found in description using word-boundary matching."""
    matches: list[str] = []
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        if re.search(pattern, description):
            matches.append(keyword)
    return matches


def _diet_exclusion_reasons(description: str, diet: str) -> list[str]:
    reasons: list[str] = []

    if diet == "vegetarian":
        for keyword in _find_keyword_matches(description, MEAT_POULTRY_KEYWORDS):
            reasons.append(f"diet (vegetarian): contains meat/poultry keyword '{keyword}'")
        for keyword in _find_keyword_matches(description, FISH_SEAFOOD_KEYWORDS):
            reasons.append(f"diet (vegetarian): contains fish/seafood keyword '{keyword}'")

    elif diet == "vegan":
        for keyword in _find_keyword_matches(description, MEAT_POULTRY_KEYWORDS):
            reasons.append(f"diet (vegan): contains meat/poultry keyword '{keyword}'")
        for keyword in _find_keyword_matches(description, FISH_SEAFOOD_KEYWORDS):
            reasons.append(f"diet (vegan): contains fish/seafood keyword '{keyword}'")
        for keyword in _find_keyword_matches(description, DAIRY_EGG_KEYWORDS):
            reasons.append(f"diet (vegan): contains dairy/egg keyword '{keyword}'")

    elif diet == "pescatarian":
        for keyword in _find_keyword_matches(description, MEAT_POULTRY_KEYWORDS):
            reasons.append(f"diet (pescatarian): contains meat/poultry keyword '{keyword}'")

    elif diet == "gluten-free":
        for keyword in _find_keyword_matches(description, GLUTEN_KEYWORDS):
            reasons.append(f"diet (gluten-free): contains gluten keyword '{keyword}'")

    else:
        raise ValueError(f"Unknown diet type: {diet}")

    return reasons


def _condition_exclusion_reasons(row: pd.Series, condition: str) -> list[str]:
    description = _normalize_description(row.get("description"))
    reasons: list[str] = []

    if condition == "IBS":
        for keyword in _find_keyword_matches(description, IBS_FODMAP_KEYWORDS):
            reasons.append(f"condition (IBS): high-FODMAP keyword '{keyword}'")

    elif condition == "GERD":
        for keyword in _find_keyword_matches(description, GERD_TRIGGER_KEYWORDS):
            reasons.append(f"condition (GERD): trigger keyword '{keyword}'")

    elif condition == "hypertension":
        for keyword in _find_keyword_matches(description, HYPERTENSION_SODIUM_KEYWORDS):
            reasons.append(
                f"condition (hypertension): high-sodium keyword '{keyword}' "
                "(sodium column not in snapshot; keyword filter applied)"
            )

    elif condition == "T2 diabetes":
        for keyword in _find_keyword_matches(description, T2_DIABETES_SUGAR_KEYWORDS):
            reasons.append(f"condition (T2 diabetes): high-sugar keyword '{keyword}'")

        carbs = row.get(CARB_COL)
        fiber = row.get(FIBER_COL)

        if pd.notna(carbs):
            if float(carbs) > T2_MAX_TOTAL_CARBS_G:
                reasons.append(
                    f"condition (T2 diabetes): total carbs {float(carbs):.1f}g "
                    f"exceeds limit of {T2_MAX_TOTAL_CARBS_G}g per 100g"
                )

            net_carbs = float(carbs)
            if pd.notna(fiber):
                net_carbs = max(float(carbs) - float(fiber), 0.0)

            if net_carbs > T2_MAX_NET_CARBS_G:
                reasons.append(
                    f"condition (T2 diabetes): net carbs {net_carbs:.1f}g "
                    f"exceeds limit of {T2_MAX_NET_CARBS_G}g per 100g"
                )

        energy = row.get("Energy")
        if pd.notna(energy) and float(energy) > T2_MAX_ENERGY_KCAL:
            if pd.notna(carbs) and float(carbs) > T2_MAX_TOTAL_CARBS_G * 0.5:
                reasons.append(
                    f"condition (T2 diabetes): energy {float(energy):.1f} kcal "
                    f"with elevated carbs exceeds diabetic energy-density threshold"
                )

    else:
        raise ValueError(f"Unknown medical condition: {condition}")

    return reasons


def _parse_persona(persona: str) -> dict[str, str]:
    normalized = persona.strip()
    for key, config in PERSONAS.items():
        if normalized.lower() == key.lower():
            return config

    raise ValueError(
        f"Unknown persona '{persona}'. Supported personas: {', '.join(PERSONAS)}"
    )


def _summarize_exclusions(exclusion_log: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in exclusion_log:
        for reason in entry["reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def apply_clinical_filters(
    df: pd.DataFrame, persona: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Filter the offline snapshot for a clinical persona.

    Parameters
    ----------
    df : pd.DataFrame
        USDA offline snapshot with at least a 'description' column.
    persona : str
        One of: 'IBS + vegetarian', 'GERD + gluten-free',
        'T2 diabetes + vegan', 'hypertension + pescatarian'.

    Returns
    -------
    filtered_df : pd.DataFrame
        Rows that pass all diet and condition filters.
    why_excluded : dict
        Transparency report with counts and sample exclusions.
    """
    if "description" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'description' column.")

    config = _parse_persona(persona)
    diet = config["diet"]
    condition = config["condition"]

    keep_indices: list[Any] = []
    exclusion_log: list[dict[str, Any]] = []

    for index, row in df.iterrows():
        description = _normalize_description(row.get("description"))
        reasons = _diet_exclusion_reasons(description, diet)
        reasons.extend(_condition_exclusion_reasons(row, condition))

        if reasons:
            exclusion_log.append(
                {
                    "fdc_id": row.get("fdc_id"),
                    "description": row.get("description"),
                    "reasons": reasons,
                }
            )
        else:
            keep_indices.append(index)

    filtered_df = df.loc[keep_indices].copy().reset_index(drop=True)

    why_excluded: dict[str, Any] = {
        "persona": persona,
        "diet": diet,
        "condition": condition,
        "initial_count": len(df),
        "final_count": len(filtered_df),
        "excluded_count": len(df) - len(filtered_df),
        "exclusion_reason_counts": _summarize_exclusions(exclusion_log),
        "excluded_samples": exclusion_log[:15],
    }

    return filtered_df, why_excluded


def load_snapshot(path: Path | str = DEFAULT_SNAPSHOT) -> pd.DataFrame:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
    return pd.read_csv(snapshot_path)


if __name__ == "__main__":
    snapshot = load_snapshot()
    print(f"Loaded snapshot: {len(snapshot)} items\n")

    for persona_name in PERSONAS:
        filtered, report = apply_clinical_filters(snapshot, persona_name)
        print(f"=== {persona_name} ===")
        print(f"  Passed: {report['final_count']} / {report['initial_count']}")
        print(f"  Excluded: {report['excluded_count']}")
        top_reasons = list(report["exclusion_reason_counts"].items())[:5]
        if top_reasons:
            print("  Top exclusion reasons:")
            for reason, count in top_reasons:
                print(f"    - {reason}: {count}")
        print()
