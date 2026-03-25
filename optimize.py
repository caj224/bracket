"""
Bracket EV optimization engine for a March Madness pool with upset bonuses.

This script:
1. Builds a full bracket using a chosen picker (greedy, chalk, or random).
2. Scores brackets by exact expected value under a matchup probability grid.
3. Improves brackets via one-flip local search / hill climbing.
4. Runs a multi-start overnight search with random, greedy, and elite restarts.
5. Saves the current best bracket and a top-k leaderboard to disk.

Dependencies:
- `teams_64`: list of 64 team dicts
- `grid`: NxN probability matrix where grid[i][j] = P(team i beats team j)

These are imported from `data.py` so the script stays easy to rerun.
"""

import copy
import json
import os
import random
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple

from data import load_bracket_data


# ============================================================
# CONFIG
# ============================================================

ROUND_PARAMS = {
    1: (1, 1),    # Round of 64
    2: (2, 2),    # Round of 32
    3: (4, 3),    # Sweet 16
    4: (8, 4),    # Elite 8
    5: (16, 5),   # Final Four
    6: (32, 6),   # National Final
}

BRACKET_SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
REGION_ORDER = ["East", "West", "South", "Midwest"]

ROUND_NAME = {
    1: "R64",
    2: "R32",
    3: "Sweet16",
    4: "Elite8",
    5: "FinalFour",
    6: "Title",
}

SAVE_PATH = "best_bracket.json"
TOP_K_PATH = "top_brackets.json"

MAX_HILL_ITERS = 100
PRINT_EVERY = 25
TOP_K = 10

# Mixed restart strategy
P_RANDOM_RESTART = 0.10
P_GREEDY_RESTART = 0.0
ELITE_POOL_SIZE = 5
ELITE_MIN_FLIPS = 2
ELITE_MAX_FLIPS = 5


# ============================================================
# BASIC HELPERS
# ============================================================

def optimize_round(team1: dict, team2: dict, prob1: float, prob2: float, round_num: int) -> dict:
    """Choose the immediate-EV winner for a single matchup."""
    base, bonus = ROUND_PARAMS[round_num]

    seed1 = team1["seed"]
    seed2 = team2["seed"]

    ev1 = prob1 * (base + bonus * max(0, seed1 - seed2))
    ev2 = prob2 * (base + bonus * max(0, seed2 - seed1))

    return team1 if ev1 >= ev2 else team2


def greedy_pick(t1: dict, t2: dict, prob_grid, name_to_idx: Dict[str, int], round_num: int) -> dict:
    """Pick the higher immediate-EV team in a matchup."""
    i = name_to_idx[t1["name"]]
    j = name_to_idx[t2["name"]]
    p1 = prob_grid[i][j]
    p2 = prob_grid[j][i]
    return optimize_round(t1, t2, p1, p2, round_num)


def favorite_pick(t1: dict, t2: dict, prob_grid, name_to_idx: Dict[str, int]) -> dict:
    """Pick the team with the higher raw win probability."""
    i = name_to_idx[t1["name"]]
    j = name_to_idx[t2["name"]]
    return t1 if prob_grid[i][j] >= prob_grid[j][i] else t2


def weighted_pick(t1: dict, t2: dict, prob_grid, name_to_idx: Dict[str, int], temp: float = 1.0) -> dict:
    """
    Stochastic matchup picker.

    temp < 1 spreads probabilities toward 50/50 more aggressively.
    temp > 1 sharpens the preference for the favorite.
    """
    i = name_to_idx[t1["name"]]
    j = name_to_idx[t2["name"]]
    p1 = prob_grid[i][j]

    a = p1 ** temp
    b = (1 - p1) ** temp
    p = a / (a + b)

    return t1 if random.random() < p else t2


def other_team_in_matchup(t1: dict, t2: dict, picked: dict) -> dict:
    """Return the non-picked team from a two-team matchup."""
    if picked["name"] == t1["name"]:
        return t2
    if picked["name"] == t2["name"]:
        return t1

    raise ValueError(
        f"Picked team {picked['name']} not in matchup {t1['name']} vs {t2['name']}"
    )


def build_region_teams(teams: List[dict]) -> Dict[str, List[dict]]:
    """Group teams by region and order them in standard bracket seed order."""
    regions = {}

    for region in REGION_ORDER:
        region_teams = [t for t in teams if t["region"] == region]
        seed_to_team = {t["seed"]: t for t in region_teams}
        ordered = [seed_to_team[seed] for seed in BRACKET_SEED_ORDER]
        regions[region] = ordered

    return regions


def print_bracket(full_bracket: dict) -> None:
    """Pretty-print a bracket for inspection."""
    for region, data in full_bracket["regions"].items():
        print(f"\n{'=' * 40}")
        print(f"  {region.upper()} REGION")
        print(f"{'=' * 40}")

        print("  R64:")
        for team in data["round_64_winners"]:
            print(f"    ({team['seed']}) {team['name']}")

        print("  R32:")
        for team in data["round_32_winners"]:
            print(f"    ({team['seed']}) {team['name']}")

        print("  Sweet 16:")
        for team in data["round_16_winners"]:
            print(f"    ({team['seed']}) {team['name']}")

        champ = data["region_champion"]
        print(f"  Elite 8 winner: ({champ['seed']}) {champ['name']}")

    print(f"\n{'=' * 40}")
    print("  FINAL FOUR")
    print(f"{'=' * 40}")

    for team in full_bracket["final_four_winners"]:
        print(f"  ({team['seed']}) {team['name']}")

    champ = full_bracket["champion"]
    print(f"\n  CHAMPION: ({champ['seed']}) {champ['name']}")


# ============================================================
# BRACKET BUILDERS
# ============================================================

def build_region_bracket(
    region_teams: List[dict],
    picker: str,
    prob_grid=None,
    name_to_idx: Optional[Dict[str, int]] = None,
) -> dict:
    """Build a single region's bracket using the requested picker."""
    def pick(t1: dict, t2: dict, round_num: int) -> dict:
        if picker == "greedy":
            return greedy_pick(t1, t2, prob_grid, name_to_idx, round_num)
        if picker == "chalk":
            return favorite_pick(t1, t2, prob_grid, name_to_idx)
        if picker == "random":
            temp = random.choice([0.7, 0.85, 1.0, 1.15, 1.3])
            return weighted_pick(t1, t2, prob_grid, name_to_idx, temp)

        raise ValueError(f"Unknown picker: {picker}")

    r64 = [pick(region_teams[i], region_teams[i + 1], 1) for i in range(0, 16, 2)]
    r32 = [pick(r64[i], r64[i + 1], 2) for i in range(0, 8, 2)]
    r16 = [pick(r32[i], r32[i + 1], 3) for i in range(0, 4, 2)]
    champ = pick(r16[0], r16[1], 4)

    return {
        "teams": region_teams,
        "round_64_winners": r64,
        "round_32_winners": r32,
        "round_16_winners": r16,
        "region_champion": champ,
    }


def build_full_bracket(
    all_region_teams: Dict[str, List[dict]],
    prob_grid,
    name_to_idx: Dict[str, int],
    picker: str = "greedy",
) -> dict:
    """
    Build a full 63-game bracket from regional team lists.

    `picker` controls how winners are selected:
    - greedy: maximize immediate expected value at each game
    - chalk: always take the more likely winner
    - random: probabilistic weighted selection
    """
    regions = {}

    for region_name, teams in all_region_teams.items():
        regions[region_name] = build_region_bracket(
            teams,
            picker=picker,
            prob_grid=prob_grid,
            name_to_idx=name_to_idx,
        )

    east = regions["East"]["region_champion"]
    south = regions["South"]["region_champion"]
    west = regions["West"]["region_champion"]
    midwest = regions["Midwest"]["region_champion"]

    if picker == "greedy":
        semi1 = greedy_pick(east, south, prob_grid, name_to_idx, 5)
        semi2 = greedy_pick(west, midwest, prob_grid, name_to_idx, 5)
        champ = greedy_pick(semi1, semi2, prob_grid, name_to_idx, 6)
    elif picker == "chalk":
        semi1 = favorite_pick(east, south, prob_grid, name_to_idx)
        semi2 = favorite_pick(west, midwest, prob_grid, name_to_idx)
        champ = favorite_pick(semi1, semi2, prob_grid, name_to_idx)
    elif picker == "random":
        semi1 = weighted_pick(east, south, prob_grid, name_to_idx)
        semi2 = weighted_pick(west, midwest, prob_grid, name_to_idx)
        champ = weighted_pick(semi1, semi2, prob_grid, name_to_idx)
    else:
        raise ValueError(f"Unknown picker: {picker}")

    return {
        "regions": regions,
        "final_four_winners": [semi1, semi2],
        "champion": champ,
    }


# ============================================================
# EXACT EV SCORER
# ============================================================

def score_bracket_exact(full_bracket: dict, prob_grid, name_to_idx: Dict[str, int]) -> float:
    """
    Compute the exact expected score of a fully specified bracket.

    The bracket is converted into a binary game tree. For each node,
    winner distributions are computed recursively, then expected points
    are summed using the user's upset-bonus scoring rules.
    """
    team_lookup = {}
    for region_data in full_bracket["regions"].values():
        for team in region_data["teams"]:
            team_lookup[team["name"]] = team

    node_lookup = {}

    def make_leaf(team: dict) -> dict:
        return {"type": "leaf", "team": team}

    def make_game(left: dict, right: dict, round_num: int, picked_winner: dict) -> dict:
        return {
            "type": "game",
            "left": left,
            "right": right,
            "round_num": round_num,
            "picked_winner": picked_winner,
        }

    def register_node(node: dict) -> None:
        node_lookup[id(node)] = node
        if node["type"] == "game":
            register_node(node["left"])
            register_node(node["right"])

    region_roots = {}

    for region_name, region_data in full_bracket["regions"].items():
        teams = region_data["teams"]
        r64 = region_data["round_64_winners"]
        r32 = region_data["round_32_winners"]
        r16 = region_data["round_16_winners"]
        champ = region_data["region_champion"]

        leaves = [make_leaf(team) for team in teams]

        round1_games = [
            make_game(leaves[i], leaves[i + 1], 1, r64[i // 2])
            for i in range(0, 16, 2)
        ]
        round2_games = [
            make_game(round1_games[i], round1_games[i + 1], 2, r32[i // 2])
            for i in range(0, 8, 2)
        ]
        round3_games = [
            make_game(round2_games[i], round2_games[i + 1], 3, r16[i // 2])
            for i in range(0, 4, 2)
        ]
        region_final = make_game(round3_games[0], round3_games[1], 4, champ)

        region_roots[region_name] = region_final
        register_node(region_final)

    semi1 = make_game(
        region_roots["East"],
        region_roots["South"],
        5,
        full_bracket["final_four_winners"][0],
    )
    semi2 = make_game(
        region_roots["West"],
        region_roots["Midwest"],
        5,
        full_bracket["final_four_winners"][1],
    )
    title_game = make_game(semi1, semi2, 6, full_bracket["champion"])

    register_node(semi1)
    register_node(semi2)
    register_node(title_game)

    winner_dist_cache = {}

    def winner_distribution(node_id: int) -> Dict[str, float]:
        if node_id in winner_dist_cache:
            return winner_dist_cache[node_id]

        node = node_lookup[node_id]

        if node["type"] == "leaf":
            result = {node["team"]["name"]: 1.0}
        else:
            left_dist = winner_distribution(id(node["left"]))
            right_dist = winner_distribution(id(node["right"]))
            result = {}

            for left_name, p_left in left_dist.items():
                i = name_to_idx[left_name]
                for right_name, p_right in right_dist.items():
                    j = name_to_idx[right_name]
                    p_match = p_left * p_right

                    result[left_name] = result.get(left_name, 0.0) + p_match * prob_grid[i][j]
                    result[right_name] = result.get(right_name, 0.0) + p_match * prob_grid[j][i]

        winner_dist_cache[node_id] = result
        return result

    def expected_points_for_game(node: dict) -> float:
        if node["type"] == "leaf":
            return 0.0

        picked = node["picked_winner"]
        picked_name = picked["name"]
        round_num = node["round_num"]
        base, bonus = ROUND_PARAMS[round_num]

        left_dist = winner_distribution(id(node["left"]))
        right_dist = winner_distribution(id(node["right"]))

        ev = 0.0

        for left_name, p_left in left_dist.items():
            left_team = team_lookup[left_name]
            i = name_to_idx[left_name]

            for right_name, p_right in right_dist.items():
                right_team = team_lookup[right_name]
                j = name_to_idx[right_name]
                p_match = p_left * p_right

                if picked_name == left_name:
                    p_win = prob_grid[i][j]
                    points = base + bonus * max(0, left_team["seed"] - right_team["seed"])
                    ev += p_match * p_win * points
                elif picked_name == right_name:
                    p_win = prob_grid[j][i]
                    points = base + bonus * max(0, right_team["seed"] - left_team["seed"])
                    ev += p_match * p_win * points

        return ev

    def total_ev(node: dict) -> float:
        if node["type"] == "leaf":
            return 0.0
        return total_ev(node["left"]) + total_ev(node["right"]) + expected_points_for_game(node)

    return total_ev(title_game)


# ============================================================
# LOCAL SEARCH / HILL CLIMB
# ============================================================

def rebuild_region_after_flip(
    region_data: dict,
    prob_grid,
    name_to_idx: Dict[str, int],
    target_round: int,
    target_slot: int,
) -> dict:
    """
    Flip one regional pick, then greedily repair downstream rounds when needed.

    Example:
    - Flip one R64 winner in a region
    - Recompute later rounds that depend on that winner
    - Preserve unaffected earlier choices
    """
    teams = region_data["teams"]

    curr_r64 = region_data["round_64_winners"]
    curr_r32 = region_data["round_32_winners"]
    curr_r16 = region_data["round_16_winners"]
    curr_champ = region_data["region_champion"]

    new_r64 = []
    for slot in range(8):
        t1 = teams[2 * slot]
        t2 = teams[2 * slot + 1]

        if target_round == 1 and target_slot == slot:
            picked = other_team_in_matchup(t1, t2, curr_r64[slot])
        else:
            picked = curr_r64[slot]

        new_r64.append(picked)

    new_r32 = []
    for slot in range(4):
        t1 = new_r64[2 * slot]
        t2 = new_r64[2 * slot + 1]

        if target_round > 2:
            picked = curr_r32[slot]
        elif target_round == 2 and target_slot == slot:
            picked = other_team_in_matchup(t1, t2, curr_r32[slot])
        elif target_round < 2:
            picked = greedy_pick(t1, t2, prob_grid, name_to_idx, 2)
        else:
            picked = curr_r32[slot]

        new_r32.append(picked)

    new_r16 = []
    for slot in range(2):
        t1 = new_r32[2 * slot]
        t2 = new_r32[2 * slot + 1]

        if target_round > 3:
            picked = curr_r16[slot]
        elif target_round == 3 and target_slot == slot:
            picked = other_team_in_matchup(t1, t2, curr_r16[slot])
        elif target_round < 3:
            picked = greedy_pick(t1, t2, prob_grid, name_to_idx, 3)
        else:
            picked = curr_r16[slot]

        new_r16.append(picked)

    t1 = new_r16[0]
    t2 = new_r16[1]

    if target_round == 4 and target_slot == 0:
        new_champ = other_team_in_matchup(t1, t2, curr_champ)
    elif target_round < 4:
        new_champ = greedy_pick(t1, t2, prob_grid, name_to_idx, 4)
    else:
        new_champ = curr_champ

    return {
        "teams": teams,
        "round_64_winners": new_r64,
        "round_32_winners": new_r32,
        "round_16_winners": new_r16,
        "region_champion": new_champ,
    }


def rebuild_final_four_greedy(full_bracket: dict, prob_grid, name_to_idx: Dict[str, int]) -> dict:
    """Recompute Final Four and title picks greedily from regional champions."""
    rc = {region: full_bracket["regions"][region]["region_champion"] for region in REGION_ORDER}

    semi1 = greedy_pick(rc["East"], rc["South"], prob_grid, name_to_idx, 5)
    semi2 = greedy_pick(rc["West"], rc["Midwest"], prob_grid, name_to_idx, 5)
    champ = greedy_pick(semi1, semi2, prob_grid, name_to_idx, 6)

    full_bracket["final_four_winners"] = [semi1, semi2]
    full_bracket["champion"] = champ
    return full_bracket


def validate_bracket(full_bracket: dict) -> None:
    """Sanity-check that the chosen champion is actually in the title game."""
    ff = full_bracket["final_four_winners"]
    champ = full_bracket["champion"]

    if champ["name"] not in {ff[0]["name"], ff[1]["name"]}:
        raise ValueError(
            f"Champion {champ['name']} not in title game {ff[0]['name']} vs {ff[1]['name']}"
        )


def rebuild_ff_after_flip(
    full_bracket: dict,
    prob_grid,
    name_to_idx: Dict[str, int],
    target_round: int,
    target_slot: int,
) -> dict:
    """Flip one Final Four or title pick while keeping the bracket valid."""
    bracket = copy.deepcopy(full_bracket)

    rc = {region: bracket["regions"][region]["region_champion"] for region in REGION_ORDER}
    curr_ff = bracket["final_four_winners"]
    curr_champ = bracket["champion"]

    east = rc["East"]
    south = rc["South"]
    west = rc["West"]
    midwest = rc["Midwest"]

    if target_round == 5 and target_slot == 0:
        semi1 = other_team_in_matchup(east, south, curr_ff[0])
    else:
        semi1 = curr_ff[0]

    if target_round == 5 and target_slot == 1:
        semi2 = other_team_in_matchup(west, midwest, curr_ff[1])
    else:
        semi2 = curr_ff[1]

    if target_round == 6:
        if curr_champ["name"] not in {semi1["name"], semi2["name"]}:
            champ = greedy_pick(semi1, semi2, prob_grid, name_to_idx, 6)
        else:
            champ = other_team_in_matchup(semi1, semi2, curr_champ)
    elif target_round == 5:
        if curr_champ["name"] in {semi1["name"], semi2["name"]}:
            champ = curr_champ
        else:
            champ = greedy_pick(semi1, semi2, prob_grid, name_to_idx, 6)
    else:
        champ = curr_champ

    bracket["final_four_winners"] = [semi1, semi2]
    bracket["champion"] = champ
    return bracket


def generate_one_flip_neighbors(
    full_bracket: dict,
    prob_grid,
    name_to_idx: Dict[str, int],
) -> Generator[Tuple[str, dict], None, None]:
    """
    Enumerate all valid one-flip neighbors of the current bracket.

    Regional moves flip exactly one pick in one region and then greedily repair downstream rounds.
    Final Four moves flip one semifinal or the title pick directly.
    """
    round_slots = {1: 8, 2: 4, 3: 2, 4: 1}

    for region in REGION_ORDER:
        for round_num, nslots in round_slots.items():
            for slot in range(nslots):
                candidate = copy.deepcopy(full_bracket)
                candidate["regions"][region] = rebuild_region_after_flip(
                    candidate["regions"][region],
                    prob_grid,
                    name_to_idx,
                    round_num,
                    slot,
                )
                candidate = rebuild_final_four_greedy(candidate, prob_grid, name_to_idx)
                desc = f"{region} {ROUND_NAME[round_num]} slot {slot}"
                yield desc, candidate

    for slot in [0, 1]:
        candidate = rebuild_ff_after_flip(full_bracket, prob_grid, name_to_idx, 5, slot)
        validate_bracket(candidate)
        yield f"FinalFour semi slot {slot}", candidate

    candidate = rebuild_ff_after_flip(full_bracket, prob_grid, name_to_idx, 6, 0)
    validate_bracket(candidate)
    yield "Title flip", candidate


def best_one_flip_improvement(full_bracket: dict, prob_grid, name_to_idx: Dict[str, int]):
    """Return the best improving one-flip neighbor, if any."""
    current_ev = score_bracket_exact(full_bracket, prob_grid, name_to_idx)

    best_ev = current_ev
    best_desc = None
    best_bracket = None

    for desc, candidate in generate_one_flip_neighbors(full_bracket, prob_grid, name_to_idx):
        try:
            validate_bracket(candidate)
            ev = score_bracket_exact(candidate, prob_grid, name_to_idx)
        except Exception:
            continue

        if ev > best_ev:
            best_ev = ev
            best_desc = desc
            best_bracket = candidate

    return best_bracket, best_ev, best_desc


def improve_bracket_hill_climb(
    start_bracket: dict,
    prob_grid,
    name_to_idx: Dict[str, int],
    max_iters: int = 50,
    verbose: bool = False,
):
    """
    Repeatedly take the best improving one-flip move until no improvement remains.
    """
    current = copy.deepcopy(start_bracket)
    current_ev = score_bracket_exact(current, prob_grid, name_to_idx)
    history = [float(current_ev)]

    if verbose:
        print(f"Starting EV: {current_ev:.4f}")

    for it in range(1, max_iters + 1):
        best_neighbor, best_neighbor_ev, best_desc = best_one_flip_improvement(
            current, prob_grid, name_to_idx
        )

        if best_neighbor is None:
            if verbose:
                print(f"No improving 1-flip neighbor found at iteration {it}. Final EV: {current_ev:.4f}")
            break

        if verbose:
            improvement = best_neighbor_ev - current_ev
            print(f"Iter {it}: {best_desc} improved EV by {improvement:.4f} -> {best_neighbor_ev:.4f}")

        current = best_neighbor
        current_ev = best_neighbor_ev
        history.append(float(current_ev))

    return current, float(current_ev), history


# ============================================================
# SAVE / LOAD
# ============================================================

def atomic_json_save(payload: dict, path: str) -> None:
    """Safely write JSON to disk by using a temporary file and replace."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def load_json(path: str, default=None):
    """Load JSON if present, otherwise return default."""
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def save_best_result(best_bracket: dict, best_ev: float, path: str, extra: Optional[dict] = None) -> None:
    """Save the current best bracket and metadata to disk."""
    payload = {
        "timestamp": datetime.now().isoformat(),
        "best_ev": float(best_ev),
        "extra": extra or {},
        "best_bracket": best_bracket,
    }
    atomic_json_save(payload, path)


def maybe_update_top_k(
    bracket: dict,
    ev: float,
    source_mode: str,
    history: List[float],
    top_k_path: str = TOP_K_PATH,
    k: int = TOP_K,
) -> None:
    """Add a candidate bracket to the top-k leaderboard with deduplication."""
    board = load_json(top_k_path, default=[])
    board.append(
        {
            "timestamp": datetime.now().isoformat(),
            "ev": float(ev),
            "source_mode": source_mode,
            "history": history,
            "bracket": bracket,
        }
    )

    board = sorted(board, key=lambda x: x["ev"], reverse=True)

    deduped = []
    seen = set()

    for row in board:
        key = json.dumps(row["bracket"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    deduped = deduped[:k]
    atomic_json_save(deduped, top_k_path)


# ============================================================
# ELITE RESTART HELPERS
# ============================================================

def choose_elite_restart(
    top_k_path: str = TOP_K_PATH,
    elite_pool_size: int = ELITE_POOL_SIZE,
):
    """Sample a restart bracket from the current top-k board."""
    board = load_json(top_k_path, default=[])

    if not board:
        return None

    usable = board[:elite_pool_size]
    picked = random.choice(usable)

    source_mode = picked["source_mode"]
    while source_mode.startswith("elite:"):
        source_mode = source_mode[len("elite:"):]

    return source_mode, copy.deepcopy(picked["bracket"])


def mutate_one_random_flip(full_bracket: dict, prob_grid, name_to_idx: Dict[str, int]) -> dict:
    """
    Apply one random valid move to a bracket.

    Move space:
    - 60 regional flips
    - 2 semifinal flips
    - 1 title flip
    """
    candidate = copy.deepcopy(full_bracket)
    move_type = random.randint(0, 62)

    if move_type < 60:
        region_idx = move_type // 15
        offset = move_type % 15
        region = REGION_ORDER[region_idx]

        if offset < 8:
            round_num, slot = 1, offset
        elif offset < 12:
            round_num, slot = 2, offset - 8
        elif offset < 14:
            round_num, slot = 3, offset - 12
        else:
            round_num, slot = 4, 0

        candidate["regions"][region] = rebuild_region_after_flip(
            candidate["regions"][region],
            prob_grid,
            name_to_idx,
            round_num,
            slot,
        )
        candidate = rebuild_final_four_greedy(candidate, prob_grid, name_to_idx)

    elif move_type < 62:
        slot = move_type - 60
        candidate = rebuild_ff_after_flip(candidate, prob_grid, name_to_idx, 5, slot)
        validate_bracket(candidate)

    else:
        candidate = rebuild_ff_after_flip(candidate, prob_grid, name_to_idx, 6, 0)
        validate_bracket(candidate)

    return candidate


def perturb_bracket(
    bracket: dict,
    prob_grid,
    name_to_idx: Dict[str, int],
    min_flips: int = 1,
    max_flips: int = 3,
) -> dict:
    """Apply a random number of random valid flips to perturb a bracket."""
    out = copy.deepcopy(bracket)
    n_flips = random.randint(min_flips, max_flips)

    for _ in range(n_flips):
        out = mutate_one_random_flip(out, prob_grid, name_to_idx)

    return out


def build_restart_bracket(
    all_region_teams: Dict[str, List[dict]],
    prob_grid,
    name_to_idx: Dict[str, int],
):
    """
    Build a restart bracket according to the configured multi-start strategy:
    random restart, greedy restart, or elite restart.
    """
    u = random.random()

    if u < P_RANDOM_RESTART:
        bracket = build_full_bracket(all_region_teams, prob_grid, name_to_idx, picker="random")
        return "random", bracket

    if u < P_RANDOM_RESTART + P_GREEDY_RESTART:
        bracket = build_full_bracket(all_region_teams, prob_grid, name_to_idx, picker="greedy")
        return "greedy", bracket

    elite_pick = choose_elite_restart()
    if elite_pick is not None:
        elite_source_mode, elite_bracket = elite_pick
        return f"elite:{elite_source_mode}", elite_bracket

    bracket = build_full_bracket(all_region_teams, prob_grid, name_to_idx, picker="random")
    return "random", bracket


# ============================================================
# OVERNIGHT SEARCH
# ============================================================

def overnight_search(
    all_region_teams: Dict[str, List[dict]],
    prob_grid,
    name_to_idx: Dict[str, int],
):
    """
    Run an indefinite multi-start local search until interrupted.

    The current best result is persisted to disk, and a top-k board is maintained
    for elite restarts and later inspection.
    """
    saved = load_json(SAVE_PATH)

    if saved is not None:
        best_ev = float(saved["best_ev"])
        best_bracket = saved["best_bracket"]
        print(f"Loaded previous best from disk: {best_ev:.4f}")
    else:
        best_bracket = build_full_bracket(all_region_teams, prob_grid, name_to_idx, picker="greedy")
        best_ev = score_bracket_exact(best_bracket, prob_grid, name_to_idx)
        save_best_result(best_bracket, best_ev, SAVE_PATH, extra={"init": "greedy"})
        maybe_update_top_k(best_bracket, best_ev, "greedy_init", [float(best_ev)])
        print(f"Initialized best EV from greedy bracket: {best_ev:.4f}")

    starts_completed = 0

    try:
        while True:
            starts_completed += 1

            start_mode, start_bracket = build_restart_bracket(
                all_region_teams,
                prob_grid,
                name_to_idx,
            )

            if start_mode.startswith("elite:"):
                start_bracket = perturb_bracket(
                    start_bracket,
                    prob_grid,
                    name_to_idx,
                    min_flips=ELITE_MIN_FLIPS,
                    max_flips=ELITE_MAX_FLIPS,
                )

            start_ev = score_bracket_exact(start_bracket, prob_grid, name_to_idx)

            final_bracket, final_ev, history = improve_bracket_hill_climb(
                start_bracket,
                prob_grid,
                name_to_idx,
                max_iters=MAX_HILL_ITERS,
                verbose=False,
            )

            maybe_update_top_k(final_bracket, final_ev, start_mode, history)

            if final_ev > best_ev:
                best_ev = final_ev
                best_bracket = final_bracket

                save_best_result(
                    best_bracket,
                    best_ev,
                    SAVE_PATH,
                    extra={
                        "starts_completed": starts_completed,
                        "start_mode": start_mode,
                        "start_ev": float(start_ev),
                        "history": history,
                    },
                )

                champ = best_bracket["champion"]["name"]
                print(
                    f"[{starts_completed}] NEW BEST | "
                    f"mode={start_mode:<14} | "
                    f"start_ev={start_ev:.4f} | "
                    f"best_ev={best_ev:.4f} | "
                    f"champ={champ}"
                )

            elif starts_completed % PRINT_EVERY == 0:
                print(
                    f"[{starts_completed}] no new best | "
                    f"mode={start_mode:<14} | "
                    f"start_ev={start_ev:.4f} | "
                    f"final_ev={final_ev:.4f} | "
                    f"current_best={best_ev:.4f}"
                )

    except KeyboardInterrupt:
        print("\nStopped by user.")
        print(f"Best EV saved in {SAVE_PATH}: {best_ev:.4f}")
        return best_bracket, best_ev


# ============================================================
# MORNING HELPERS
# ============================================================

def show_saved_best() -> None:
    """Print the currently saved best bracket from disk."""
    saved = load_json(SAVE_PATH)
    if saved is None:
        print("No saved best bracket found.")
        return

    print(f"Saved timestamp: {saved['timestamp']}")
    print(f"Saved best EV:   {saved['best_ev']:.4f}")
    print_bracket(saved["best_bracket"])


def show_top_k() -> None:
    """Print the saved top-k leaderboard."""
    board = load_json(TOP_K_PATH, default=[])
    if not board:
        print("No top-bracket file found.")
        return

    for i, row in enumerate(board, 1):
        champ = row["bracket"]["champion"]["name"]
        print(
            f"{i:>2}. EV={row['ev']:.4f} | "
            f"champ={champ} | "
            f"mode={row['source_mode']} | "
            f"time={row['timestamp']}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    teams_64, grid = load_bracket_data()
    all_region_teams = build_region_teams(teams_64)
    name_to_idx = {team["name"]: i for i, team in enumerate(teams_64)}

    print("Starting overnight bracket search...")
    print(f"Results will be saved to: {SAVE_PATH}")
    print(f"Top {TOP_K} brackets will be saved to: {TOP_K_PATH}")
    print("Press Ctrl+C in the morning to stop.\n")

    overnight_search(all_region_teams, grid, name_to_idx)


if __name__ == "__main__":
    main()