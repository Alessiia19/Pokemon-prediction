from collections import Counter

STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")
BOOST_NAMES = ("atk", "def", "spa", "spd", "spe")
DEBILITATING = {"slp": "slp", "par": "par", "brn": "brn", "psn": "psn", "tox": "psn", "frz": "frz"}
STATUS_KINDS = ("slp", "par", "brn", "psn", "frz")
WEATHER_MOVES = {"raindance": "rain", "sunnyday": "sun", "sandstorm": "sand"}
SCREEN_MOVES = ("reflect", "lightscreen")
TRACKED_EFFECTS = ("substitute", "leechseed", "confusion", "trapped", "perish", "encore")
CHECKPOINTS = (5, 10, 15, 20, 25)
LOW_HP = 0.5
VERY_LOW_HP = 0.25
UNDAMAGED_TOLERANCE = 1e-9
MIN_TURNS_LEFT = 5
RECENT_TURNS = 10
UNSEEN_P2_STRENGTH = 500.0

def is_hidden_power(move):
    return move["name"].startswith("hiddenpower")


def clean_types(types):
    return [type_name for type_name in types if type_name != "notype"]


def create_species_dict(battles):
    """Map each pokemon name to its types and base stats"""
    species = {}
    for battle in battles:
        players_pokemons = battle["p1_team_details"] + [battle["p2_lead_details"]]
        for pokemon in players_pokemons:
            name = pokemon["name"]
            types = clean_types(pokemon["types"])
            if types and name not in species:
                species[name] = {
                    "types": types,
                    "stats": {stat: float(pokemon.get(f"base_{stat}") or 0.0) for stat in STAT_NAMES},
                }
    return species


def multiplier(attack_type, defending_types, type_dict, known_types):
    """Return the type multiplier"""
    if not attack_type or attack_type not in known_types or not defending_types:
        return None
    
    result = 1.0
    for defending_type in defending_types:
        result *= type_dict.get((attack_type, defending_type), 1.0)
    return result


def move_multiplier(move, defending_types, type_dict, known_types):
    if is_hidden_power(move):
        return None
    return multiplier(move["type"], defending_types, type_dict, known_types)


def get_best_multiplier(attacker_types, defending_types, type_dict, known_types):
    """Return the best matchup multiplier among the attacker's types."""
    values = []
    for attack_type in attacker_types or []:
        value = multiplier(attack_type, defending_types, type_dict, known_types)
        if value is not None:
            values.append(value)

    if values:
        return max(values)
    return None


def scan_side(turns, side):
    """Track the last observed HP and status of every Pokemon of one player."""
    last_hp, last_status, fainted = {}, {}, set()
    actives, hp_totals, exits = [], [], []
    previous_name, previous_hp, previous_status = None, None, None
    damage_taken, healing, switches, first_seen_switches, returning_switches = 0.0, 0.0, 0, 0, 0
    for index, turn in enumerate(turns):
        state = turn.get(f"{side}_pokemon_state") or {}
        name, hp, status = state["name"], float(state.get("hp_pct") or 0.0), state["status"]
        known_before = last_hp.get(name, 1.0)

        # Count a switch when the active Pokemon changes, and track whether it is a returning Pokemon or a first-seen Pokemon
        if name != previous_name and previous_name is not None:
            switches += 1
            if name in last_hp:
                returning_switches += 1
            else:
                first_seen_switches += 1
            exits.append({
                "turn": index, "out": previous_name, "out_hp": previous_hp, "out_status": previous_status,
                "in": name, "in_hp": hp, "in_known_hp": known_before,
            })
        elif previous_name is not None:
            damage_taken += max(previous_hp - hp, 0.0)
            healing += max(hp - previous_hp, 0.0)
        if status == "fnt":
            fainted.add(name)
            hp = 0.0
        last_hp[name], last_status[name] = hp, status
        actives.append(name)
        hp_totals.append(sum(last_hp.values()) + max(6 - len(last_hp), 0))
        previous_name, previous_hp, previous_status = name, hp, status
    return {
        "last_hp": last_hp, "last_status": last_status, "fainted": fainted, "actives": actives,
        "hp_totals": hp_totals, "exits": exits, "damage_taken": damage_taken, "healing": healing,
        "switches": switches, "first_seen_switches": first_seen_switches, "returning_switches": returning_switches,
        "last_state": (turns[-1].get(f"{side}_pokemon_state") or {}) if turns else {},
    }


def get_used_moves(turns, side, actives):
    for index, turn in enumerate(turns):
        move = turn.get(f"{side}_move_details")
        if move is not None and (index == 0 or actives[index] == actives[index - 1]):
            yield index, move


def get_suspected_p2_ko(turns, p1, p2, species, type_dict, known_types):
    """Find P2 exits that look like hidden KOs.

    Signature: P1 used a damaging move that could hit the newcomer, yet the
    newcomer took no damage, so the move most likely struck the Pokemon that
    left. On train.jsonl, with at least 10 turns left, 92% of such exits below
    25% HP never return, against 18% for ordinary exits at high HP.
    """
    used_moves = dict(get_used_moves(turns, "p1", p1["actives"]))
    p2_active_pokemon = p2["actives"]

    # Pokemon that left without returning, and the subset that left at low HP or very low HP.
    gone_any, gone_low, gone_very_low = set(), set(), set()

    # Hidden-KO candidates with a move signature, split by HP evidence and remaining battle time.
    suspected_ko = set()
    suspected_ko_at_low_hp = set()
    suspected_ko_at_low_hp_with_time_remaining = set()
    returned_pokemon = 0

    gone_buckets = Counter()
    last_exit = {}

    for event in p2["exits"]:

        # Known KOs
        if event["out_status"] == "fnt":
            continue

        pokemon_out, index = event["out"], event["turn"]
        never_returns = pokemon_out not in p2_active_pokemon[index:]
        
        if never_returns:
            last_exit[pokemon_out] = index

            # Divide the gone Pokemon into buckets by their last observed HP
            gone_buckets[f"b{min(int(event['out_hp'] * 4), 3)}"] += 1

            gone_any.add(pokemon_out)
            if event["out_hp"] < LOW_HP:
                gone_low.add(pokemon_out)
            if event["out_hp"] < VERY_LOW_HP:
                gone_very_low.add(pokemon_out)


        move = used_moves.get(index)

        # Skip if p1 didn't use a damaging move
        if move is None or move["category"] == "status":
            continue

        # Skip if the move couldn't hit the incoming Pokemon
        incoming_types = species.get(event["in"], {}).get("types")
        if move_multiplier(move, incoming_types, type_dict, known_types) == 0:
            continue

        # Skip if the incoming Pokemon took damage
        if event["in_hp"] < event["in_known_hp"] - UNDAMAGED_TOLERANCE:
            continue

        if not never_returns:
            returned_pokemon += 1
            continue

        suspected_ko.add(pokemon_out)
        gone_buckets[f"sig_b{min(int(event['out_hp'] * 4), 3)}"] += 1
        if event["out_hp"] < LOW_HP:
            suspected_ko_at_low_hp.add(pokemon_out)
            if len(turns) - index > MIN_TURNS_LEFT:
                suspected_ko_at_low_hp_with_time_remaining.add(pokemon_out)

    return {
        "gone": gone_any, "gone_low": gone_low, "gone_very_low": gone_very_low,
        "suspected_ko": suspected_ko,
        "suspected_ko_at_low_hp": suspected_ko_at_low_hp,
        "suspected_ko_at_low_hp_with_time_remaining": suspected_ko_at_low_hp_with_time_remaining,
        "returned_pokemon": returned_pokemon,
        "gone_buckets": gone_buckets, "last_exit": last_exit,
    }


def get_moves_effectiveness_counts(turns, side, attacker, defender, species, type_dict, known_types):
    counts = Counter()
    powers, weighted = [], []

    for index, move in get_used_moves(turns, side, attacker["actives"]):
        counts["used"] += 1

        # Status moves
        if move.get("category") == "status":
            counts["status"] += 1
            continue

        # Offensive moves
        counts["attack"] += 1
        hidden = is_hidden_power(move)
        power = float(move.get("base_power") or 0.0)
        if not hidden:
            powers.append(power)
        value = move_multiplier(move, species.get(defender["actives"][index], {}).get("types"), type_dict, known_types)
        
        if value is None:
            counts["unknown"] += 1
        elif value == 0:
            counts["immune"] += 1
        elif value < 1:
            counts["resisted"] += 1
        elif value == 1:
            counts["neutral"] += 1
        else:
            counts["super"] += 1

        if value is not None and not hidden:
            weighted.append(power * value)

    return {
        "moves_used": counts["used"], "moves_status": counts["status"], "moves_attack": counts["attack"],
        "eff_unknown": counts["unknown"], "eff_immune": counts["immune"], "eff_resisted": counts["resisted"],
        "eff_neutral": counts["neutral"], "eff_super": counts["super"],
        "power_mean": sum(powers) / len(powers) if powers else 0.0,
        "power_x_eff_sum": sum(weighted),
    }


def get_team_strength(names, scan, species):
    """Base-stat total of known Pokemon, weighted by their remaining HP."""
    total = 0.0
    for name in names:
        stats = species[name]["stats"]
        total += sum(stats.values()) * scan["last_hp"].get(name, 1.0)
    return total


def get_status_counts(scan):
    return Counter(
        DEBILITATING[status] for name, status in scan["last_status"].items()
        if status in DEBILITATING and name not in scan["fainted"]
    )


def get_field_state(turns):
    spikes = {"p1": 0, "p2": 0}
    counts = Counter()
    last_weather, last_weather_turn = "", None

    # Scan the battles to track spikes, weather, and screens
    for index, turn in enumerate(turns):
        for side, opponent in (("p1", "p2"), ("p2", "p1")):
            move = turn.get(f"{side}_move_details")
            name = move["name"] if move is not None else ""

            if name == "spikes":
                spikes[opponent] = 1
                counts[f"{side}_spikes_used"] += 1
            elif name == "rapidspin":
                spikes[side] = 0
                counts[f"{side}_rapidspin_used"] += 1
            elif name in WEATHER_MOVES:
                last_weather, last_weather_turn = WEATHER_MOVES[name], index
                counts[f"{side}_weather_used"] += 1
            elif name in SCREEN_MOVES:
                counts[f"{side}_screens_used"] += 1
    return spikes, counts, last_weather, last_weather_turn


def get_features_from_battle(battle, type_dict, species, features, known_types):
    """
    Extract main features from a single battle, like:
    - Observed and suspected KOs
    - HP left and status
    - Moves used and their effectiveness
    - Field hazards and weather
    """

    turns = battle["battle_timeline"]

    p1 = scan_side(turns, "p1")
    p2 = scan_side(turns, "p2")
    team = []
    for pokemon in battle.get("p1_team_details"):
        team.append(pokemon["name"])

    # ---  KO Counts, both observed and estimated ---

    # Observed KO counts
    features["state_p1_ko_observed"] = len(p1["fainted"])
    features["state_p2_ko_observed"] = len(p2["fainted"])
    
    suspects = get_suspected_p2_ko(turns, p1, p2, species, type_dict, known_types)
    p2_tot_ko_estimated = p2["fainted"] | suspects["suspected_ko_at_low_hp"] # Union of observed and most suspected KO

    # Suspected KO counts in various categories, excluding already fainted Pokemon
    # - Strict: suspected KO at low HP
    # - Settled: suspected KO at low HP with time remaining
    # - Loose: suspected KO in general
    features["state_p2_ko_suspected_strict"] = len(suspects["suspected_ko_at_low_hp"] - p2["fainted"])
    features["state_p2_ko_suspected_settled"] = len(suspects["suspected_ko_at_low_hp_with_time_remaining"] - p2["fainted"])
    features["state_p2_ko_suspected_loose"] = len(suspects["suspected_ko"] - p2["fainted"])

    # Suspected KO that actually returned to the battle
    features["state_p2_signature_but_returned"] = suspects["returned_pokemon"]

    # Pokemon that left the battle without returning, excluding already fainted Pokemon
    features["state_p2_gone"] = len(suspects["gone"] - p2["fainted"])
    features["state_p2_gone_low_hp"] = len(suspects["gone_low"] - p2["fainted"])
    features["state_p2_gone_very_low_hp"] = len(suspects["gone_very_low"] - p2["fainted"])

    # Estimated KO count and balance compared with P1
    features["state_p2_ko_estimated"] = len(p2_tot_ko_estimated)
    features["state_ko_estimated_balance"] = len(p2_tot_ko_estimated) - len(p1["fainted"])

    # Gone pokemon grouped into buckets by their last observed HP
    for bucket in ("b0", "b1", "b2", "b3", "sig_b0", "sig_b1", "sig_b2", "sig_b3"):
        features[f"state_p2_gone_{bucket}"] = suspects["gone_buckets"][bucket]

    # Broader estimation of P2 pokemon that are likely out of the battle 
    # including fainted, gone at low HP and suspected KO at low HP
    p2_likely_out = (p2["fainted"] | suspects["gone_low"] | suspects["suspected_ko_at_low_hp"])
    features["state_p2_ko_likely"] = len(p2_likely_out)
    features["state_ko_likely_balance"] = len(p2_likely_out) - len(p1["fainted"])

    # Recent KO counts
    recent_ko = len(turns) - RECENT_TURNS    
    first_ko_turn = {}
    for index, turn in enumerate(turns):
        state = turn.get("p1_pokemon_state") or {}
        if state.get("status") == "fnt":
            first_ko_turn.setdefault(state["name"], index)

    recent_p1_ko = 0
    for index in first_ko_turn.values():
        if index >= recent_ko:
            recent_p1_ko += 1
    features["recent_p1_ko"] = recent_p1_ko

    recent_p2_likely_ko = 0
    for name, index in suspects["last_exit"].items():
        if name in p2_likely_out and index >= recent_ko:
            recent_p2_likely_ko += 1
    features["recent_p2_likely_ko"] = recent_p2_likely_ko
    features["recent_ko_balance"] = features["recent_p2_likely_ko"] - features["recent_p1_ko"]
    
    # Recent turns since the last KO
    features["recent_turns_since_p1_ko"] = (len(turns)-1 - max(first_ko_turn.values()) if first_ko_turn else len(turns))


    # --- Team HP estimation 

    # Adjust P2's last observed HP by setting it to 0 for suspected KO
    p2_adjusted, p2_likely = {}, {}
    for name, hp in p2["last_hp"].items():
        if name in p2_tot_ko_estimated:
            p2_adjusted[name] = 0.0
        else:
            p2_adjusted[name] = hp

        if name in p2_likely_out:
            p2_likely[name] = 0.0
        else:
            p2_likely[name] = hp

    # Mean Team HP 
    p1_seen, p2_seen = len(p1["last_hp"]), len(p2["last_hp"])
    p1_team_hp = (sum(p1["last_hp"].values()) + max(6 - p1_seen, 0)) / 6
    p2_team_hp = (sum(p2["last_hp"].values()) + max(6 - p2_seen, 0)) / 6
    p2_team_hp_adjusted = (sum(p2_adjusted.values()) + max(6 - p2_seen, 0)) / 6
    p2_team_hp_likely = (sum(p2_likely.values()) + max(6 - p2_seen, 0)) / 6

    features["state_p1_seen"] = p1_seen
    features["state_p2_seen"] = p2_seen
    features["state_p1_team_hp"] = p1_team_hp
    features["state_p2_team_hp"] = p2_team_hp
    features["state_p2_team_hp_adjusted"] = p2_team_hp_adjusted
    features["state_team_hp_balance"] = p1_team_hp - p2_team_hp
    features["state_team_hp_balance_adjusted"] = p1_team_hp - p2_team_hp_adjusted
    features["state_p1_seen_hp_mean"] = sum(p1["last_hp"].values()) / p1_seen
    features["state_p2_seen_hp_mean"] = sum(p2["last_hp"].values()) / p2_seen

    # Pokemon with at least 50% HP
    features["state_p1_healthy"] = sum(hp >= LOW_HP for hp in p1["last_hp"].values()) + max(6 - p1_seen, 0)
    features["state_p2_healthy"] = sum(hp >= LOW_HP for hp in p2_adjusted.values()) + max(6 - p2_seen, 0)
    features["state_healthy_balance"] = features["state_p1_healthy"] - features["state_p2_healthy"]
    features["state_p2_team_hp_likely"] = p2_team_hp_likely
    features["state_team_hp_balance_likely"] = p1_team_hp - p2_team_hp_likely
    features["state_p2_healthy_likely"] = sum(hp >= LOW_HP for hp in p2_likely.values()) + max(6 - p2_seen, 0)
    features["state_healthy_balance_likely"] = features["state_p1_healthy"] - features["state_p2_healthy_likely"]

    # Pokemon probably alive in each team
    features["state_p2_alive_likely"] = 6 - len(p2_likely_out)
    features["state_p1_alive"] = 6 - len(p1["fainted"])
    features["state_alive_balance_likely"] = features["state_p1_alive"] - features["state_p2_alive_likely"]

    # Track team HP balance at checkpoints to estimate advantage over time
    for checkpoint in CHECKPOINTS:
        reached = min(checkpoint, len(turns)) - 1
        features[f"trend_hp_balance_t{checkpoint}"] = (p1["hp_totals"][reached] - p2["hp_totals"][reached]) / 6
    last_balance = (p1["hp_totals"][-1] - p2["hp_totals"][-1]) / 6
    
    features["trend_hp_momentum_last10"] = last_balance - features["trend_hp_balance_t20"]
    features["trend_hp_momentum_last5"] = last_balance - features["trend_hp_balance_t25"]

    # Damage taken and healing 
    features["trend_p1_damage_taken"] = p1["damage_taken"]
    features["trend_p2_damage_taken"] = p2["damage_taken"]
    features["trend_p1_healing"] = p1["healing"]
    features["trend_p2_healing"] = p2["healing"]
    features["trend_damage_balance"] = p2["damage_taken"] - p1["damage_taken"]

    # Switches 
    for side, scan in (("p1", p1), ("p2", p2)):
        features[f"state_{side}_switches"] = scan["switches"]
        features[f"state_{side}_first_seen_switches"] = scan["first_seen_switches"]
        features[f"state_{side}_returning_switches"] = scan["returning_switches"]

    # --- Matchup advantage analysis 
    p1_type_adv_turns, p1_type_disadv_turns, p1_faster_turns = 0, 0, 0
    active_hp_gap, recent_hp_gap, recent_p1_damage, recent_p2_damage = 0.0, 0.0, 0.0, 0.0

    for index, turn in enumerate(turns):
        a_name, b_name = p1["actives"][index], p2["actives"][index]
        a_info, b_info = species.get(a_name, {}), species.get(b_name, {})

        # Get the best multiplier for each active Pokemon against the other
        a_stab = get_best_multiplier(a_info.get("types"), b_info.get("types"), type_dict, known_types)
        b_stab = get_best_multiplier(b_info.get("types"), a_info.get("types"), type_dict, known_types)
        if a_stab is not None and b_stab is not None:
            p1_type_adv_turns += a_stab > b_stab
            p1_type_disadv_turns += a_stab < b_stab

        # Count the number of turns where p1's active Pokemon is faster than p2's
        p1_faster_turns += a_info.get("stats", {}).get("spe", 0.0) > b_info.get("stats", {}).get("spe", 0.0)

        # HP gap between the two active Pokemon
        a_hp = float(turn["p1_pokemon_state"].get("hp_pct") or 0.0)
        b_hp = float(turn["p2_pokemon_state"].get("hp_pct") or 0.0)
        active_hp_gap += a_hp - b_hp

        # HP gap and damage taken in the last 10 turns
        if index >= recent_ko:
            recent_hp_gap += a_hp - b_hp

            if index and p1["actives"][index - 1] == a_name:
                previous_state = turns[index - 1]["p1_pokemon_state"]
                previous_hp = float(previous_state.get("hp_pct") or 0.0)
                recent_p1_damage += max(previous_hp - a_hp, 0.0)

            if index and p2["actives"][index - 1] == b_name:
                previous_state = turns[index - 1]["p2_pokemon_state"]
                previous_hp = float(previous_state.get("hp_pct") or 0.0)
                recent_p2_damage += max(previous_hp - b_hp, 0.0)
                
    features["matchup_favourable_turns"] = p1_type_adv_turns
    features["matchup_unfavourable_turns"] = p1_type_disadv_turns
    features["matchup_balance"] = p1_type_adv_turns - p1_type_disadv_turns
    features["matchup_faster_turns"] = p1_faster_turns
    features["matchup_active_hp_gap_mean"] = active_hp_gap/ len(turns)
    features["recent_active_hp_gap_mean"] = recent_hp_gap/ max(len(turns) - max(recent_ko, 0), 1)
    features["recent_damage_balance"] = recent_p2_damage - recent_p1_damage

    # Count status conditions across each team
    p1_status, p2_status = get_status_counts(p1), get_status_counts(p2)
    for kind in STATUS_KINDS:
        features[f"status_p1_{kind}"] = p1_status[kind]
        features[f"status_p2_{kind}"] = p2_status[kind]
    features["status_p1_total"] = sum(p1_status.values())
    features["status_p2_total"] = sum(p2_status.values())
    features["status_balance"] = features["status_p2_total"] - features["status_p1_total"]

    # Count boosts 
    for side, scan in (("p1", p1), ("p2", p2)):
        boosts = scan["last_state"].get("boosts") or {}
        values = [float(boosts.get(boost) or 0.0) for boost in BOOST_NAMES]
        features[f"boost_{side}_sum"] = sum(values)
        features[f"boost_{side}_positive"] = sum(max(value, 0.0) for value in values)
        features[f"boost_{side}_negative"] = sum(min(value, 0.0) for value in values)
        effects = set(scan["last_state"].get("effects") or [])
        for effect in TRACKED_EFFECTS:
            features[f"effect_{side}_{effect}"] = int(effect in effects)
    features["boost_balance"] = features["boost_p1_sum"] - features["boost_p2_sum"]

    # Count field hazards and weather
    spikes, field_counts, weather, weather_turn = get_field_state(turns)
    features["field_spikes_on_p1_side"] = spikes["p1"]
    features["field_spikes_on_p2_side"] = spikes["p2"]
    for side in ("p1", "p2"):
        for action in ("spikes_used", "rapidspin_used", "weather_used", "screens_used"):
            features[f"field_{side}_{action}"] = field_counts[f"{side}_{action}"]
    for kind in ("rain", "sun", "sand"):
        features[f"field_last_weather_{kind}"] = int(weather == kind)
    features["field_turns_since_weather"] = len(turns) - 1 - weather_turn if weather_turn is not None else -1

    # Moves effectiveness counts
    p1_moves = get_moves_effectiveness_counts(turns, "p1", p1, p2, species, type_dict, known_types)
    p2_moves = get_moves_effectiveness_counts(turns, "p2", p2, p1, species, type_dict, known_types)
    for key in p1_moves:
        features[f"moves_p1_{key}"] = p1_moves[key]
        features[f"moves_p2_{key}"] = p2_moves[key]

    # Super-effective moves balance
    features["moves_super_balance"] = p1_moves["eff_super"] - p2_moves["eff_super"]

    # Moves power overall effectiveness balance
    features["moves_power_balance"] = p1_moves["power_x_eff_sum"] - p2_moves["power_x_eff_sum"]

    # --- Last turn analysis
    p1_active, p2_active = p1["actives"][-1], p2["actives"][-1]
    p1_types = species.get(p1_active, {}).get("types")
    p2_types = species.get(p2_active, {}).get("types")
    p1_stats, p2_stats = species.get(p1_active, {}).get("stats", {}), species.get(p2_active, {}).get("stats", {})
    p1_stab = get_best_multiplier(p1_types, p2_types, type_dict, known_types)
    p2_stab = get_best_multiplier(p2_types, p1_types, type_dict, known_types)
    features["end_p1_stab_vs_p2"] = p1_stab if p1_stab is not None else -1.0
    features["end_p2_stab_vs_p1"] = p2_stab if p2_stab is not None else -1.0
    features["end_speed_difference"] = p1_stats.get("spe", 0.0) - p2_stats.get("spe", 0.0)
    features["end_p1_active_hp"] = p1["last_hp"].get(p1_active, 0.0)
    features["end_p2_active_hp"] = p2["last_hp"].get(p2_active, 0.0)
    features["end_active_hp_balance"] = features["end_p1_active_hp"] - features["end_p2_active_hp"]
    features["end_p1_active_status"] = int(p1["last_status"].get(p1_active) in DEBILITATING)
    features["end_p2_active_status"] = int(p2["last_status"].get(p2_active) in DEBILITATING)

    # Team strength estimation
    p2_alive = [name for name in p2["last_hp"] if name not in p2_tot_ko_estimated]
    p1_alive = [name for name in team if name not in p1["fainted"]]
    features["strength_p1_left"] = get_team_strength(team, p1, species)
    features["strength_p2_seen_left"] = sum(sum(species.get(name, {}).get("stats", {}).values()) * p2_adjusted[name] for name in p2_alive)
    
    features["strength_p2_seen_alive"] = len(p2_alive)
    features["strength_p1_alive"] = len(p1_alive)
    
    features["strength_p1_team_total"] = sum(sum(species.get(name, {}).get("stats", {}).values()) for name in team)
    p2_seen_total = [sum(species.get(name, {}).get("stats", {}).values()) for name in p2["last_hp"]]
    features["strength_p2_seen_mean"] = sum(p2_seen_total) / max(len(p2_seen_total), 1)

    # Estimation of P2's total strength, including unseen Pokemon
    p2_seen_strength = 0.0
    for name in p2["last_hp"]:
        pokemon_strength = sum(species.get(name, {}).get("stats", {}).values())
        p2_seen_strength += pokemon_strength * p2_likely[name]

    p2_unseen_count = max(6 - p2_seen, 0)
    p2_unseen_strength = p2_unseen_count * UNSEEN_P2_STRENGTH
    p2_estimated_strength = p2_seen_strength + p2_unseen_strength
    features["strength_balance_estimated"] = features["strength_p1_left"] - p2_estimated_strength

    # Count how many Pokemon are alive with a super-effective matchup against the other team's active Pokemon
    p1_alive_supereff = []
    for name in p1_alive:
        value = get_best_multiplier(species.get(name, {}).get("types"), p2_types, type_dict, known_types)
        if value is not None:
            p1_alive_supereff.append(value)
    features["end_p1_alive_with_super_vs_active"] = sum(value > 1 for value in p1_alive_supereff)
    
    p2_alive_supereff = []
    for name in p2_alive:
        value = get_best_multiplier(species.get(name, {}).get("types"), p1_types, type_dict, known_types)
        if value is not None:
            p2_alive_supereff.append(value)
    features["end_p2_alive_with_super_vs_active"] = sum(value > 1 for value in p2_alive_supereff)

    # Survivors' HP-weighted stats
    for stat in STAT_NAMES:
        p1_total = sum(species.get(n, {}).get("stats", {}).get(stat, 0.0) * p1["last_hp"].get(n, 1.0) for n in p1_alive)
        p2_total = sum(species.get(n, {}).get("stats", {}).get(stat, 0.0) * p2_likely[n] for n in p2["last_hp"])
        features[f"survivors_p1_{stat}"] = p1_total
        features[f"survivors_p2_{stat}"] = p2_total
    pair_values, reverse_values = [], []

    # Survivors type matchup coverage
    for attacker in p1_alive:
        for defender in p2["last_hp"]: 
            if defender in p2_likely_out:
                continue
            forward = get_best_multiplier(species.get(attacker, {}).get("types"), species.get(defender, {}).get("types"), type_dict, known_types)
            backward = get_best_multiplier(species.get(defender, {}).get("types"), species.get(attacker, {}).get("types"), type_dict, known_types)
            if forward is not None and backward is not None:
                pair_values.append(forward)
                reverse_values.append(backward)
    features["survivors_coverage_p1"] = sum(pair_values) / len(pair_values) if pair_values else 1.0
    features["survivors_coverage_p2"] = sum(reverse_values) / len(reverse_values) if reverse_values else 1.0
    features["survivors_coverage_balance"] = features["survivors_coverage_p1"] - features["survivors_coverage_p2"]

    # Pokemon still alive in each team
    for name in p1_alive:
        features[f"alive_p1_{name}"] = 1
    for name in p2_alive:
        features[f"alive_p2_{name}"] = 1
