from collections import Counter

import pandas as pd

from src.battle_state import STAT_NAMES, get_features_from_battle


TYPE_NAMES = (
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
)
KNOWN_TYPES = frozenset(TYPE_NAMES)
MISSING_CATEGORY = "__missing__"


def get_category(value):
    if value is None:
        return MISSING_CATEGORY
    return str(value).strip().lower()


def get_value(value):
    if value is None:
        return 0.0
    return float(value)


def create_type_dict(type_chart):
    type_dict = {}
    for row in type_chart.itertuples(index=False):
        attacking_type = str(row.attacking_type)
        defending_type = str(row.defending_type)
        type_dict[(attacking_type, defending_type)] = get_value(row.multiplier)
    return type_dict


def effectiveness(attacking_types, defending_types, type_dict):
    """Return the best multiplier an attacker's types can reach against a defender.

    Each attacking type is a separate option, so the options are compared and
    not multiplied together. Defending types do multiply. The supplied chart
    lists only non-neutral pairs, so a missing pair means 1.0.
    """
    attacks = []
    for attack_type in attacking_types or []:
        if attack_type and attack_type != "notype":
            attacks.append(attack_type)

    defenses = []
    for defense_type in defending_types or []:
        if defense_type and defense_type != "notype":
            defenses.append(defense_type)

    if not attacks:
        return 1.0
    best = 0.0
    for attack_type in attacks:
        multiplier = 1.0
        for defense_type in defenses:
            multiplier *= type_dict.get((str(attack_type), str(defense_type)), 1.0)
        best = max(best, multiplier)
    return best


def pokemon_types(pokemon):
    return pokemon["types"]


def add_pokemon_features(prefix, pokemon, features, with_item=True):
    features[f"{prefix}_name"] = get_category(pokemon.get("name"))
    if with_item:
        features[f"{prefix}_item"] = get_category(pokemon.get("item"))
    types = []
    for type_name in pokemon_types(pokemon):
        if type_name != "notype":
            types.append(type_name)

    features[f"{prefix}_type_1"] = get_category(types[0] if types else None)
    features[f"{prefix}_type_2"] = get_category(types[1] if len(types) > 1 else None)
    for stat in STAT_NAMES:
        features[f"{prefix}_base_{stat}"] = get_value(pokemon.get(f"base_{stat}"))


def add_numeric_summary(prefix, values, features):
    if not values:
        for suffix in ("mean", "min", "max", "std", "sum"):
            features[f"{prefix}_{suffix}"] = 0.0
        return
    series = pd.Series(values, dtype="float64")
    features[f"{prefix}_mean"] = float(series.mean())
    features[f"{prefix}_min"] = float(series.min())
    features[f"{prefix}_max"] = float(series.max())
    features[f"{prefix}_std"] = float(series.std(ddof=0))
    features[f"{prefix}_sum"] = float(series.sum())


def add_team_features(battle, type_dict, features):
    """Add the pre-battle features: team stats, lead comparison, type matchups."""
    team = battle["p1_team_details"][:6]
    p2_lead = battle["p2_lead_details"]

    for stat in STAT_NAMES:
        values = []
        for pokemon in team:
            values.append(get_value(pokemon.get(f"base_{stat}")))
        add_numeric_summary(f"p1_team_base_{stat}", values, features)

    p1_lead = team[0]
    add_pokemon_features("p1_lead", p1_lead, features)
    add_pokemon_features("p2_lead", p2_lead, features, with_item=False)

    for stat in STAT_NAMES:
        p1_value = get_value(p1_lead.get(f"base_{stat}"))
        p2_value = get_value(p2_lead.get(f"base_{stat}"))
        features[f"lead_base_{stat}_difference"] = p1_value - p2_value
        features[f"lead_base_{stat}_ratio"] = p1_value / p2_value if p2_value else 0.0
        team_mean = features[f"p1_team_base_{stat}_mean"]
        features[f"team_mean_base_{stat}_difference"] = team_mean - p2_value

    team_types = []
    for pokemon in team:
        for type_name in pokemon_types(pokemon):
            if type_name != "notype":
                team_types.append(type_name)
    type_counts = Counter(team_types)
    for type_name in TYPE_NAMES:
        features[f"p1_team_type_count_{type_name}"] = type_counts[type_name]

    p2_types = pokemon_types(p2_lead)
    p1_lead_types = pokemon_types(p1_lead)
    team_effectiveness = []
    for pokemon in team:
        pokemon_effectiveness = effectiveness(
            pokemon_types(pokemon),
            p2_types,
            type_dict,
        )
        team_effectiveness.append(pokemon_effectiveness)
    add_numeric_summary("p1_team_vs_p2_lead_type", team_effectiveness, features)
    features["p1_lead_vs_p2_lead_type"] = effectiveness(p1_lead_types, p2_types, type_dict)
    features["p2_lead_vs_p1_lead_type"] = effectiveness(p2_types, p1_lead_types, type_dict)


def add_last_turn_categories(turns, features):
    """Add the last active Pokemon, status and move as categories for one-hot encoding."""
    last_turn = turns[-1] if turns else {}
    for side in ("p1", "p2"):
        state = last_turn.get(f"{side}_pokemon_state")
        state = state if isinstance(state, dict) else {}
        features[f"{side}_last_active_name"] = get_category(state.get("name"))
        features[f"{side}_last_status"] = get_category(state.get("status"))
        move = last_turn.get(f"{side}_move_details")
        move = move if isinstance(move, dict) else {}
        features[f"{side}_last_move_name"] = get_category(move.get("name"))
        features[f"{side}_last_move_type"] = get_category(move.get("type"))


def extract_battle_features(battle, type_chart, species):
    """Extract pre-battle and reconstructed end-of-observation features for one battle."""
    features = {}
    type_dict = type_chart if isinstance(type_chart, dict) else create_type_dict(type_chart)
    add_team_features(battle, type_dict, features)
    turns = battle.get("battle_timeline")
    turns = turns if isinstance(turns, list) else []
    add_last_turn_categories(turns, features)
    get_features_from_battle(battle, type_dict, species, features, KNOWN_TYPES)
    features["battle_id"] = battle.get("battle_id")
    if "player_won" in battle:
        features["player_won"] = int(bool(battle["player_won"]))
    return features


def build_features(battles, type_chart, species):
    type_dict = (
        type_chart
        if isinstance(type_chart, dict)
        else create_type_dict(type_chart)
    )
    rows = (extract_battle_features(battle, type_dict, species) for battle in battles)
    dataset = pd.DataFrame.from_records(rows)

    return dataset.fillna(0)
