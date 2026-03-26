"""
Partial-round bracket optimization.

For each cutoff round k, this script finds the bracket that maximizes
expected points earned through round k only.

Examples:
- cutoff=1: optimize only Round of 64 picks
- cutoff=2: optimize through Round of 32
- cutoff=3: optimize through Sweet 16
- cutoff=4: optimize through Elite 8
- cutoff=5: optimize through Final Four
- cutoff=6: optimize full bracket through Title

After the cutoff round, no picks are stored or optimized.
"""

import copy
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from data import load_bracket_data
from optimize import (
    ELITE_MAX_FLIPS,
    ELITE_MIN_FLIPS,
    MAX_HILL_ITERS,
    P_GREEDY_RESTART,
    P_RANDOM_RESTART,
    PRINT_EVERY,
    ROUND_PARAMS,
    ROUND_NAME,
    REGION_ORDER,
    BRACKET_SEED_ORDER,
    SAVE_PATH as FULL_BRACKET_SAVE_PATH,
    TOP_K,
    atomic_json_save,
    load_json,
    greedy_pick,
    favorite_pick,
    weighted_pick,
    other_team_in_matchup,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PARTIAL_RESULTS_DIR = RESULTS_DIR / "partial"

PARTIAL_SAVE_PATH_TEMPLATE = str(PARTIAL_RESULTS_DIR / "best_partial_round_{cutoff}.json")
PARTIAL_TOP_K_PATH_TEMPLATE = str(PARTIAL_RESULTS_DIR / "top_partial_round_{cutoff}.json")

# Bias deeper horizons harder because the early-round searches are cheaper and
# tend to stabilize sooner.
PARTIAL_SEARCH_RUN_WEIGHTS = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 10,
    6: 0,
}

# For Elite 8 / Final Four, some "random" restarts begin near the saved
# full-title bracket and then get lightly perturbed before hill climbing.
TITLE_GUIDED_RANDOM_RESTART_PROB = {
    4: 0.45,
    5: 0.70,
}
TITLE_GUIDED_MIN_FLIPS = {
    4: 1,
    5: 1,
}
TITLE_GUIDED_MAX_FLIPS = {
    4: 3,
    5: 4,
}

# For deeper horizons, spend some restarts refining the current incumbent best
# instead of always starting from scratch.
INCUMBENT_GUIDED_RESTART_PROB = {
    4: 0.35,
    5: 0.55,
    6: 0.25,
}
INCUMBENT_GUIDED_MIN_FLIPS = {
    4: 1,
    5: 1,
    6: 1,
}
INCUMBENT_GUIDED_MAX_FLIPS = {
    4: 2,
    5: 2,
    6: 2,
}


# ============================================================
# PARTIAL BRACKET BUILDERS
# ============================================================

def build_region_teams(teams: List[dict]) -> Dict[str, List[dict]]:
    regions = {}
    for region in REGION_ORDER:
        region_teams = [t for t in teams if t["region"] == region]
        seed_to_team = {t["seed"]: t for t in region_teams}
        ordered = [seed_to_team[seed] for seed in BRACKET_SEED_ORDER]
        regions[region] = ordered
    return regions


def build_partial_region_bracket(
    region_teams: List[dict],
    cutoff_round: int,
    picker: str,
    prob_grid,
    name_to_idx: Dict[str, int],
) -> dict:
    def pick(t1: dict, t2: dict, round_num: int) -> dict:
        if picker == "greedy":
            return greedy_pick(t1, t2, prob_grid, name_to_idx, round_num)
        if picker == "chalk":
            return favorite_pick(t1, t2, prob_grid, name_to_idx)
        if picker == "random":
            return weighted_pick(t1, t2, prob_grid, name_to_idx)
        raise ValueError(f"Unknown picker: {picker}")

    out = {"teams": region_teams}

    if cutoff_round >= 1:
        r64 = [pick(region_teams[i], region_teams[i + 1], 1) for i in range(0, 16, 2)]
        out["round_64_winners"] = r64
    else:
        return out

    if cutoff_round >= 2:
        r32 = [pick(r64[i], r64[i + 1], 2) for i in range(0, 8, 2)]
        out["round_32_winners"] = r32
    else:
        return out

    if cutoff_round >= 3:
        r16 = [pick(r32[i], r32[i + 1], 3) for i in range(0, 4, 2)]
        out["round_16_winners"] = r16
    else:
        return out

    if cutoff_round >= 4:
        champ = pick(r16[0], r16[1], 4)
        out["region_champion"] = champ
    else:
        return out

    return out


def build_partial_bracket(
    all_region_teams: Dict[str, List[dict]],
    cutoff_round: int,
    prob_grid,
    name_to_idx: Dict[str, int],
    picker: str = "greedy",
) -> dict:
    regions = {}

    for region_name, teams in all_region_teams.items():
        regions[region_name] = build_partial_region_bracket(
            region_teams=teams,
            cutoff_round=min(cutoff_round, 4),
            picker=picker,
            prob_grid=prob_grid,
            name_to_idx=name_to_idx,
        )

    out = {"cutoff_round": cutoff_round, "regions": regions}

    if cutoff_round < 5:
        return out

    east = regions["East"]["region_champion"]
    south = regions["South"]["region_champion"]
    west = regions["West"]["region_champion"]
    midwest = regions["Midwest"]["region_champion"]

    if picker == "greedy":
        semi1 = greedy_pick(east, south, prob_grid, name_to_idx, 5)
        semi2 = greedy_pick(west, midwest, prob_grid, name_to_idx, 5)
    elif picker == "chalk":
        semi1 = favorite_pick(east, south, prob_grid, name_to_idx)
        semi2 = favorite_pick(west, midwest, prob_grid, name_to_idx)
    elif picker == "random":
        semi1 = weighted_pick(east, south, prob_grid, name_to_idx)
        semi2 = weighted_pick(west, midwest, prob_grid, name_to_idx)
    else:
        raise ValueError(f"Unknown picker: {picker}")

    out["final_four_winners"] = [semi1, semi2]

    if cutoff_round < 6:
        return out

    if picker == "greedy":
        champ = greedy_pick(semi1, semi2, prob_grid, name_to_idx, 6)
    elif picker == "chalk":
        champ = favorite_pick(semi1, semi2, prob_grid, name_to_idx)
    else:
        champ = weighted_pick(semi1, semi2, prob_grid, name_to_idx)

    out["champion"] = champ
    return out

def rebuild_partial_final_rounds_greedy(partial_bracket: dict, prob_grid, name_to_idx: Dict[str, int]) -> dict:
    """
    After changing a regional pick in a partial bracket, rebuild downstream
    Final Four / title picks greedily if those rounds are included in the cutoff.
    """
    cutoff_round = partial_bracket["cutoff_round"]

    if cutoff_round < 5:
        return partial_bracket

    rc = {r: partial_bracket["regions"][r]["region_champion"] for r in REGION_ORDER}

    semi1 = greedy_pick(rc["East"], rc["South"], prob_grid, name_to_idx, 5)
    semi2 = greedy_pick(rc["West"], rc["Midwest"], prob_grid, name_to_idx, 5)
    partial_bracket["final_four_winners"] = [semi1, semi2]

    if cutoff_round >= 6:
        partial_bracket["champion"] = greedy_pick(semi1, semi2, prob_grid, name_to_idx, 6)

    return partial_bracket
# ============================================================
# PARTIAL EXACT EV SCORER
# ============================================================

def score_partial_bracket_exact(partial_bracket: dict, prob_grid, name_to_idx: Dict[str, int]) -> float:
    cutoff_round = partial_bracket["cutoff_round"]

    team_lookup = {}
    for region_data in partial_bracket["regions"].values():
        for team in region_data["teams"]:
            team_lookup[team["name"]] = team

    total_ev = 0.0

    def matchup_ev(left_dist: Dict[str, float], right_dist: Dict[str, float], picked_name: str, round_num: int) -> float:
        base, bonus = ROUND_PARAMS[round_num]
        ev = 0.0

        for left_name, p_left in left_dist.items():
            i = name_to_idx[left_name]
            left_team = team_lookup[left_name]

            for right_name, p_right in right_dist.items():
                j = name_to_idx[right_name]
                right_team = team_lookup[right_name]
                p_match = p_left * p_right

                if picked_name == left_name:
                    p_win = prob_grid[i][j]
                    pts = base + bonus * max(0, left_team["seed"] - right_team["seed"])
                    ev += p_match * p_win * pts
                elif picked_name == right_name:
                    p_win = prob_grid[j][i]
                    pts = base + bonus * max(0, right_team["seed"] - left_team["seed"])
                    ev += p_match * p_win * pts

        return ev

    def winner_dist_two_teams(t1: dict, t2: dict) -> Tuple[Dict[str, float], Dict[str, float]]:
        return ({t1["name"]: 1.0}, {t2["name"]: 1.0})

    region_champ_dists = {}

    for region_name, region_data in partial_bracket["regions"].items():
        teams = region_data["teams"]

        r64_inputs = [(teams[i], teams[i + 1]) for i in range(0, 16, 2)]
        r64_winner_dists = []

        if cutoff_round >= 1:
            for slot, picked in enumerate(region_data["round_64_winners"]):
                left_dist, right_dist = winner_dist_two_teams(*r64_inputs[slot])
                total_ev += matchup_ev(left_dist, right_dist, picked["name"], 1)

                t1, t2 = r64_inputs[slot]
                i = name_to_idx[t1["name"]]
                j = name_to_idx[t2["name"]]
                r64_winner_dists.append({
                    t1["name"]: prob_grid[i][j],
                    t2["name"]: prob_grid[j][i],
                })

        if cutoff_round >= 2:
            r32_winner_dists = []
            for slot, picked in enumerate(region_data["round_32_winners"]):
                left_dist = r64_winner_dists[2 * slot]
                right_dist = r64_winner_dists[2 * slot + 1]
                total_ev += matchup_ev(left_dist, right_dist, picked["name"], 2)

                out = {}
                for left_name, p_left in left_dist.items():
                    i = name_to_idx[left_name]
                    for right_name, p_right in right_dist.items():
                        j = name_to_idx[right_name]
                        p_match = p_left * p_right
                        out[left_name] = out.get(left_name, 0.0) + p_match * prob_grid[i][j]
                        out[right_name] = out.get(right_name, 0.0) + p_match * prob_grid[j][i]
                r32_winner_dists.append(out)

        if cutoff_round >= 3:
            r16_winner_dists = []
            for slot, picked in enumerate(region_data["round_16_winners"]):
                left_dist = r32_winner_dists[2 * slot]
                right_dist = r32_winner_dists[2 * slot + 1]
                total_ev += matchup_ev(left_dist, right_dist, picked["name"], 3)

                out = {}
                for left_name, p_left in left_dist.items():
                    i = name_to_idx[left_name]
                    for right_name, p_right in right_dist.items():
                        j = name_to_idx[right_name]
                        p_match = p_left * p_right
                        out[left_name] = out.get(left_name, 0.0) + p_match * prob_grid[i][j]
                        out[right_name] = out.get(right_name, 0.0) + p_match * prob_grid[j][i]
                r16_winner_dists.append(out)

        if cutoff_round >= 4:
            picked = region_data["region_champion"]
            left_dist = r16_winner_dists[0]
            right_dist = r16_winner_dists[1]
            total_ev += matchup_ev(left_dist, right_dist, picked["name"], 4)

            out = {}
            for left_name, p_left in left_dist.items():
                i = name_to_idx[left_name]
                for right_name, p_right in right_dist.items():
                    j = name_to_idx[right_name]
                    p_match = p_left * p_right
                    out[left_name] = out.get(left_name, 0.0) + p_match * prob_grid[i][j]
                    out[right_name] = out.get(right_name, 0.0) + p_match * prob_grid[j][i]
            region_champ_dists[region_name] = out

    if cutoff_round >= 5:
        ff = partial_bracket["final_four_winners"]

        left_dist = region_champ_dists["East"]
        right_dist = region_champ_dists["South"]
        total_ev += matchup_ev(left_dist, right_dist, ff[0]["name"], 5)

        semi1_dist = {}
        for left_name, p_left in left_dist.items():
            i = name_to_idx[left_name]
            for right_name, p_right in right_dist.items():
                j = name_to_idx[right_name]
                p_match = p_left * p_right
                semi1_dist[left_name] = semi1_dist.get(left_name, 0.0) + p_match * prob_grid[i][j]
                semi1_dist[right_name] = semi1_dist.get(right_name, 0.0) + p_match * prob_grid[j][i]

        left_dist = region_champ_dists["West"]
        right_dist = region_champ_dists["Midwest"]
        total_ev += matchup_ev(left_dist, right_dist, ff[1]["name"], 5)

        semi2_dist = {}
        for left_name, p_left in left_dist.items():
            i = name_to_idx[left_name]
            for right_name, p_right in right_dist.items():
                j = name_to_idx[right_name]
                p_match = p_left * p_right
                semi2_dist[left_name] = semi2_dist.get(left_name, 0.0) + p_match * prob_grid[i][j]
                semi2_dist[right_name] = semi2_dist.get(right_name, 0.0) + p_match * prob_grid[j][i]

    if cutoff_round >= 6:
        champ = partial_bracket["champion"]
        total_ev += matchup_ev(semi1_dist, semi2_dist, champ["name"], 6)

    return total_ev


# ============================================================
# PARTIAL NEIGHBORS
# ============================================================

def rebuild_region_after_flip_partial(
    region_data: dict,
    cutoff_round: int,
    prob_grid,
    name_to_idx: Dict[str, int],
    target_round: int,
    target_slot: int,
) -> dict:
    teams = region_data["teams"]
    out = {"teams": teams}

    curr_r64 = region_data.get("round_64_winners")
    curr_r32 = region_data.get("round_32_winners")
    curr_r16 = region_data.get("round_16_winners")
    curr_champ = region_data.get("region_champion")

    if cutoff_round >= 1:
        new_r64 = []
        for slot in range(8):
            t1, t2 = teams[2 * slot], teams[2 * slot + 1]
            if target_round == 1 and target_slot == slot:
                picked = other_team_in_matchup(t1, t2, curr_r64[slot])
            else:
                picked = curr_r64[slot]
            new_r64.append(picked)
        out["round_64_winners"] = new_r64

    if cutoff_round >= 2:
        new_r32 = []
        for slot in range(4):
            t1, t2 = out["round_64_winners"][2 * slot], out["round_64_winners"][2 * slot + 1]
            if target_round > 2:
                picked = curr_r32[slot]
            elif target_round == 2 and target_slot == slot:
                picked = other_team_in_matchup(t1, t2, curr_r32[slot])
            elif target_round < 2:
                picked = greedy_pick(t1, t2, prob_grid, name_to_idx, 2)
            else:
                picked = curr_r32[slot]
            new_r32.append(picked)
        out["round_32_winners"] = new_r32

    if cutoff_round >= 3:
        new_r16 = []
        for slot in range(2):
            t1, t2 = out["round_32_winners"][2 * slot], out["round_32_winners"][2 * slot + 1]
            if target_round > 3:
                picked = curr_r16[slot]
            elif target_round == 3 and target_slot == slot:
                picked = other_team_in_matchup(t1, t2, curr_r16[slot])
            elif target_round < 3:
                picked = greedy_pick(t1, t2, prob_grid, name_to_idx, 3)
            else:
                picked = curr_r16[slot]
            new_r16.append(picked)
        out["round_16_winners"] = new_r16

    if cutoff_round >= 4:
        t1, t2 = out["round_16_winners"][0], out["round_16_winners"][1]
        if target_round == 4:
            new_champ = other_team_in_matchup(t1, t2, curr_champ)
        elif target_round < 4:
            new_champ = greedy_pick(t1, t2, prob_grid, name_to_idx, 4)
        else:
            new_champ = curr_champ
        out["region_champion"] = new_champ

    return out


def generate_one_flip_neighbors_partial(
    partial_bracket: dict,
    prob_grid,
    name_to_idx: Dict[str, int],
) -> Generator[Tuple[str, dict], None, None]:
    cutoff_round = partial_bracket["cutoff_round"]
    regional_round_slots = {1: 8, 2: 4, 3: 2, 4: 1}

    for region in REGION_ORDER:
        for round_num, nslots in regional_round_slots.items():
            if round_num > min(cutoff_round, 4):
                continue

            for slot in range(nslots):
                candidate = copy.deepcopy(partial_bracket)
                candidate["regions"][region] = rebuild_region_after_flip_partial(
                    candidate["regions"][region],
                    cutoff_round=min(cutoff_round, 4),
                    prob_grid=prob_grid,
                    name_to_idx=name_to_idx,
                    target_round=round_num,
                    target_slot=slot,
                )
                candidate = rebuild_partial_final_rounds_greedy(candidate, prob_grid, name_to_idx)
                yield f"{region} {ROUND_NAME[round_num]} slot {slot}", candidate

    if cutoff_round >= 5:
        ff = partial_bracket["final_four_winners"]
        rc = {r: partial_bracket["regions"][r]["region_champion"] for r in REGION_ORDER}

        cand = copy.deepcopy(partial_bracket)
        cand["final_four_winners"] = [
            other_team_in_matchup(rc["East"], rc["South"], ff[0]),
            ff[1],
        ]
        yield "FinalFour semi slot 0", cand

        cand = copy.deepcopy(partial_bracket)
        cand["final_four_winners"] = [
            ff[0],
            other_team_in_matchup(rc["West"], rc["Midwest"], ff[1]),
        ]
        yield "FinalFour semi slot 1", cand

    if cutoff_round >= 6:
        cand = copy.deepcopy(partial_bracket)
        semi1, semi2 = cand["final_four_winners"]
        current_champ = cand["champion"]
        # Champion may be stale if finalists changed (e.g. after a semi flip)
        if current_champ["name"] not in (semi1["name"], semi2["name"]):
            current_champ = greedy_pick(semi1, semi2, prob_grid, name_to_idx, 6)
            cand["champion"] = current_champ
        cand["champion"] = other_team_in_matchup(semi1, semi2, current_champ)
        yield "Title flip", cand


def best_one_flip_improvement_partial(partial_bracket: dict, prob_grid, name_to_idx: Dict[str, int]):
    current_ev = score_partial_bracket_exact(partial_bracket, prob_grid, name_to_idx)

    best_ev = current_ev
    best_desc = None
    best_bracket = None

    for desc, candidate in generate_one_flip_neighbors_partial(partial_bracket, prob_grid, name_to_idx):
        ev = score_partial_bracket_exact(candidate, prob_grid, name_to_idx)
        if ev > best_ev:
            best_ev = ev
            best_desc = desc
            best_bracket = candidate

    return best_bracket, best_ev, best_desc


def improve_partial_bracket_hill_climb(
    start_bracket: dict,
    prob_grid,
    name_to_idx: Dict[str, int],
    max_iters: int = 100,
    verbose: bool = False,
):
    current = copy.deepcopy(start_bracket)
    current_ev = score_partial_bracket_exact(current, prob_grid, name_to_idx)
    history = [float(current_ev)]

    if verbose:
        print(f"Starting EV: {current_ev:.4f}")

    for it in range(1, max_iters + 1):
        best_neighbor, best_neighbor_ev, best_desc = best_one_flip_improvement_partial(
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
# REPORTING
# ============================================================

def summarize_partial_bracket(partial_bracket: dict) -> dict:
    cutoff_round = partial_bracket["cutoff_round"]
    out = {"cutoff_round": cutoff_round, "regions": {}}

    for region, data in partial_bracket["regions"].items():
        region_out = {}
        if cutoff_round >= 1:
            region_out["round_64_winners"] = [t["name"] for t in data["round_64_winners"]]
        if cutoff_round >= 2:
            region_out["round_32_winners"] = [t["name"] for t in data["round_32_winners"]]
        if cutoff_round >= 3:
            region_out["round_16_winners"] = [t["name"] for t in data["round_16_winners"]]
        if cutoff_round >= 4:
            region_out["region_champion"] = data["region_champion"]["name"]
        out["regions"][region] = region_out

    if cutoff_round >= 5:
        out["final_four_winners"] = [t["name"] for t in partial_bracket["final_four_winners"]]
    if cutoff_round >= 6:
        out["champion"] = partial_bracket["champion"]["name"]

    return out

def partial_save_path(cutoff_round: int) -> str:
    return PARTIAL_SAVE_PATH_TEMPLATE.format(cutoff=cutoff_round)


def partial_top_k_path(cutoff_round: int) -> str:
    return PARTIAL_TOP_K_PATH_TEMPLATE.format(cutoff=cutoff_round)


def load_title_round_from_full_results() -> dict:
    saved = load_json(FULL_BRACKET_SAVE_PATH)
    if saved is None:
        raise FileNotFoundError(
            f"Full-bracket result not found at {FULL_BRACKET_SAVE_PATH}. Run scripts/optimize.py first."
        )

    best_bracket = copy.deepcopy(saved["best_bracket"])
    best_bracket["cutoff_round"] = 6

    return {
        "timestamp": saved.get("timestamp"),
        "best_ev": float(saved["best_ev"]),
        "extra": saved.get("extra", {}),
        "best_bracket": best_bracket,
        "source_path": FULL_BRACKET_SAVE_PATH,
    }


def project_title_bracket_to_cutoff(cutoff_round: int) -> Optional[dict]:
    if cutoff_round < 4 or cutoff_round > 5:
        return None

    try:
        title_bracket = load_title_round_from_full_results()["best_bracket"]
    except FileNotFoundError:
        return None

    out = {"cutoff_round": cutoff_round, "regions": {}}

    for region in REGION_ORDER:
        src = title_bracket["regions"][region]
        region_out = {"teams": copy.deepcopy(src["teams"])}
        if cutoff_round >= 1:
            region_out["round_64_winners"] = copy.deepcopy(src["round_64_winners"])
        if cutoff_round >= 2:
            region_out["round_32_winners"] = copy.deepcopy(src["round_32_winners"])
        if cutoff_round >= 3:
            region_out["round_16_winners"] = copy.deepcopy(src["round_16_winners"])
        if cutoff_round >= 4:
            region_out["region_champion"] = copy.deepcopy(src["region_champion"])
        out["regions"][region] = region_out

    if cutoff_round >= 5:
        out["final_four_winners"] = copy.deepcopy(title_bracket["final_four_winners"])

    return out


def save_best_partial_result(
    best_bracket: dict,
    best_ev: float,
    cutoff_round: int,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "best_ev": float(best_ev),
        "extra": extra or {},
        "best_bracket": best_bracket,
    }
    atomic_json_save(payload, partial_save_path(cutoff_round))


def save_best_title_result(
    best_bracket: dict,
    best_ev: float,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "best_ev": float(best_ev),
        "extra": extra or {},
        "best_bracket": copy.deepcopy(best_bracket),
    }
    payload["best_bracket"].pop("cutoff_round", None)
    atomic_json_save(payload, FULL_BRACKET_SAVE_PATH)


def maybe_update_partial_top_k(
    bracket: dict,
    ev: float,
    source_mode: str,
    history: List[float],
    cutoff_round: int,
    k: int = TOP_K,
) -> None:
    top_k_path = partial_top_k_path(cutoff_round)
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

    atomic_json_save(deduped[:k], top_k_path)


def choose_partial_elite_restart(cutoff_round: int, elite_pool_size: int = 5):
    board = load_json(partial_top_k_path(cutoff_round), default=[])
    if not board:
        return None

    usable = board[:elite_pool_size]
    picked = random.choice(usable)
    source_mode = picked["source_mode"]
    while source_mode.startswith("elite:"):
        source_mode = source_mode[len("elite:"):]

    return source_mode, copy.deepcopy(picked["bracket"])


def choose_title_guided_partial_restart(
    cutoff_round: int,
    prob_grid,
    name_to_idx: Dict[str, int],
):
    base = project_title_bracket_to_cutoff(cutoff_round)
    if base is None:
        return None

    return perturb_partial_bracket(
        base,
        prob_grid,
        name_to_idx,
        min_flips=TITLE_GUIDED_MIN_FLIPS[cutoff_round],
        max_flips=TITLE_GUIDED_MAX_FLIPS[cutoff_round],
    )


def choose_incumbent_guided_partial_restart(
    incumbent_bracket: Optional[dict],
    prob_grid,
    name_to_idx: Dict[str, int],
):
    if incumbent_bracket is None:
        return None

    cutoff_round = incumbent_bracket["cutoff_round"]
    guided_prob = INCUMBENT_GUIDED_RESTART_PROB.get(cutoff_round, 0.0)
    if guided_prob <= 0 or random.random() >= guided_prob:
        return None

    return perturb_partial_bracket(
        incumbent_bracket,
        prob_grid,
        name_to_idx,
        min_flips=INCUMBENT_GUIDED_MIN_FLIPS[cutoff_round],
        max_flips=INCUMBENT_GUIDED_MAX_FLIPS[cutoff_round],
    )


def mutate_one_random_flip_partial(partial_bracket: dict, prob_grid, name_to_idx: Dict[str, int]) -> dict:
    candidate = copy.deepcopy(partial_bracket)
    cutoff_round = candidate["cutoff_round"]

    regional_moves = sum({1: 8, 2: 4, 3: 2, 4: 1}[r] for r in range(1, min(cutoff_round, 4) + 1))
    total_moves = 4 * regional_moves
    if cutoff_round >= 5:
        total_moves += 2
    if cutoff_round >= 6:
        total_moves += 1

    move_idx = random.randrange(total_moves)

    regional_move_counts = []
    for round_num in range(1, min(cutoff_round, 4) + 1):
        regional_move_counts.append((round_num, {1: 8, 2: 4, 3: 2, 4: 1}[round_num]))

    regional_total = 4 * regional_moves
    if move_idx < regional_total:
        region_idx = move_idx // regional_moves
        offset = move_idx % regional_moves
        region = REGION_ORDER[region_idx]

        for round_num, slots in regional_move_counts:
            if offset < slots:
                slot = offset
                candidate["regions"][region] = rebuild_region_after_flip_partial(
                    candidate["regions"][region],
                    cutoff_round=min(cutoff_round, 4),
                    prob_grid=prob_grid,
                    name_to_idx=name_to_idx,
                    target_round=round_num,
                    target_slot=slot,
                )
                candidate = rebuild_partial_final_rounds_greedy(candidate, prob_grid, name_to_idx)
                return candidate
            offset -= slots

    ff_offset = move_idx - regional_total
    if cutoff_round >= 5 and ff_offset < 2:
        ff = candidate["final_four_winners"]
        rc = {r: candidate["regions"][r]["region_champion"] for r in REGION_ORDER}

        if ff_offset == 0:
            semi1 = other_team_in_matchup(rc["East"], rc["South"], ff[0])
            candidate["final_four_winners"] = [semi1, ff[1]]
        else:
            semi2 = other_team_in_matchup(rc["West"], rc["Midwest"], ff[1])
            candidate["final_four_winners"] = [ff[0], semi2]

        if cutoff_round >= 6:
            semi1, semi2 = candidate["final_four_winners"]
            champ = candidate["champion"]
            if champ["name"] not in {semi1["name"], semi2["name"]}:
                candidate["champion"] = greedy_pick(semi1, semi2, prob_grid, name_to_idx, 6)
        return candidate

    if cutoff_round >= 6:
        semi1, semi2 = candidate["final_four_winners"]
        champ = candidate["champion"]
        if champ["name"] not in {semi1["name"], semi2["name"]}:
            champ = greedy_pick(semi1, semi2, prob_grid, name_to_idx, 6)
        candidate["champion"] = other_team_in_matchup(semi1, semi2, champ)
        return candidate

    return candidate


def perturb_partial_bracket(
    bracket: dict,
    prob_grid,
    name_to_idx: Dict[str, int],
    min_flips: int = 1,
    max_flips: int = 3,
) -> dict:
    out = copy.deepcopy(bracket)
    n_flips = random.randint(min_flips, max_flips)

    for _ in range(n_flips):
        out = mutate_one_random_flip_partial(out, prob_grid, name_to_idx)

    return out


def build_partial_restart_bracket(
    all_region_teams,
    cutoff_round: int,
    prob_grid,
    name_to_idx,
    incumbent_bracket: Optional[dict] = None,
):
    incumbent_guided = choose_incumbent_guided_partial_restart(
        incumbent_bracket,
        prob_grid,
        name_to_idx,
    )
    if incumbent_guided is not None:
        return "incumbent_guided", incumbent_guided

    u = random.random()

    if u < P_RANDOM_RESTART:
        guided_prob = TITLE_GUIDED_RANDOM_RESTART_PROB.get(cutoff_round, 0.0)
        if guided_prob > 0 and random.random() < guided_prob:
            guided = choose_title_guided_partial_restart(cutoff_round, prob_grid, name_to_idx)
            if guided is not None:
                return "title_guided_random", guided
        bracket = build_partial_bracket(all_region_teams, cutoff_round, prob_grid, name_to_idx, picker="random")
        return "random", bracket

    if u < P_RANDOM_RESTART + P_GREEDY_RESTART:
        bracket = build_partial_bracket(all_region_teams, cutoff_round, prob_grid, name_to_idx, picker="greedy")
        return "greedy", bracket

    elite_pick = choose_partial_elite_restart(cutoff_round)
    if elite_pick is not None:
        elite_source_mode, elite_bracket = elite_pick
        return f"elite:{elite_source_mode}", elite_bracket

    bracket = build_partial_bracket(all_region_teams, cutoff_round, prob_grid, name_to_idx, picker="random")
    return "random", bracket


def initialize_partial_search_for_round(
    all_region_teams,
    cutoff_round: int,
    prob_grid,
    name_to_idx,
):
    if cutoff_round == 6:
        title_payload = load_title_round_from_full_results()
        best_ev = float(title_payload["best_ev"])
        if PARTIAL_SEARCH_RUN_WEIGHTS.get(6, 0) <= 0:
            print(f"{ROUND_NAME[cutoff_round]:<10} using optimize.py result: {best_ev:.4f}")
            return {
                "cutoff_round": cutoff_round,
                "best_bracket": title_payload["best_bracket"],
                "best_ev": best_ev,
                "starts_completed": 0,
                "external_source": True,
            }

        print(f"{ROUND_NAME[cutoff_round]:<10} initialized from optimize.py result: {best_ev:.4f}")
        maybe_update_partial_top_k(
            title_payload["best_bracket"],
            best_ev,
            "full_init",
            [float(best_ev)],
            cutoff_round,
        )
        return {
            "cutoff_round": cutoff_round,
            "best_bracket": title_payload["best_bracket"],
            "best_ev": best_ev,
            "starts_completed": 0,
        }

    save_path = partial_save_path(cutoff_round)
    saved = load_json(save_path)

    if saved is not None:
        best_ev = float(saved["best_ev"])
        best_bracket = saved["best_bracket"]
        print(f"{ROUND_NAME[cutoff_round]:<10} loaded previous best: {best_ev:.4f}")
    else:
        best_bracket = build_partial_bracket(
            all_region_teams,
            cutoff_round,
            prob_grid,
            name_to_idx,
            picker="greedy",
        )
        best_ev = score_partial_bracket_exact(best_bracket, prob_grid, name_to_idx)
        save_best_partial_result(best_bracket, best_ev, cutoff_round, extra={"init": "greedy"})
        maybe_update_partial_top_k(best_bracket, best_ev, "greedy_init", [float(best_ev)], cutoff_round)
        print(f"{ROUND_NAME[cutoff_round]:<10} initialized best EV: {best_ev:.4f}")

    return {
        "cutoff_round": cutoff_round,
        "best_bracket": best_bracket,
        "best_ev": best_ev,
        "starts_completed": 0,
    }


def run_partial_search_step(
    state: dict,
    all_region_teams,
    prob_grid,
    name_to_idx,
) -> None:
    if state.get("external_source"):
        title_payload = load_title_round_from_full_results()
        state["best_bracket"] = title_payload["best_bracket"]
        state["best_ev"] = float(title_payload["best_ev"])
        return

    cutoff_round = state["cutoff_round"]
    state["starts_completed"] += 1
    starts_completed = state["starts_completed"]

    start_mode, start_bracket = build_partial_restart_bracket(
        all_region_teams,
        cutoff_round,
        prob_grid,
        name_to_idx,
        incumbent_bracket=state["best_bracket"],
    )

    if start_mode.startswith("elite:"):
        start_bracket = perturb_partial_bracket(
            start_bracket,
            prob_grid,
            name_to_idx,
            min_flips=ELITE_MIN_FLIPS,
            max_flips=ELITE_MAX_FLIPS,
        )

    start_ev = score_partial_bracket_exact(start_bracket, prob_grid, name_to_idx)
    final_bracket, final_ev, history = improve_partial_bracket_hill_climb(
        start_bracket,
        prob_grid,
        name_to_idx,
        max_iters=MAX_HILL_ITERS,
        verbose=False,
    )

    maybe_update_partial_top_k(final_bracket, final_ev, start_mode, history, cutoff_round)

    if final_ev > state["best_ev"]:
        state["best_ev"] = final_ev
        state["best_bracket"] = final_bracket
        save_extra = {
            "starts_completed": starts_completed,
            "start_mode": start_mode,
            "start_ev": float(start_ev),
            "history": history,
        }
        if cutoff_round == 6:
            save_best_title_result(final_bracket, final_ev, extra=save_extra)
        else:
            save_best_partial_result(final_bracket, final_ev, cutoff_round, extra=save_extra)
        print(
            f"{ROUND_NAME[cutoff_round]:<10} [{starts_completed}] NEW BEST | "
            f"mode={start_mode:<14} | start_ev={start_ev:.4f} | best_ev={final_ev:.4f}"
        )
    elif starts_completed % PRINT_EVERY == 0:
        print(
            f"{ROUND_NAME[cutoff_round]:<10} [{starts_completed}] no new best | "
            f"mode={start_mode:<14} | start_ev={start_ev:.4f} | "
            f"final_ev={final_ev:.4f} | current_best={state['best_ev']:.4f}"
        )


def main():
    teams_64, grid = load_bracket_data()
    all_region_teams = build_region_teams(teams_64)
    name_to_idx = {team["name"]: i for i, team in enumerate(teams_64)}
    print("Starting ongoing partial-round searches...")
    print("Rounds 1-5 keep their own partial results; Title uses the saved full bracket unless weight 6 is set above 0.")
    print("Press Ctrl+C to stop.\n")

    try:
        for cutoff_round in range(1, 6):
            print(
                f"{ROUND_NAME[cutoff_round]:<10} files: "
                f"{partial_save_path(cutoff_round)}, {partial_top_k_path(cutoff_round)}"
            )
        print(f"{ROUND_NAME[6]:<10} file:  {FULL_BRACKET_SAVE_PATH}")
        print("")

        states = []
        for cutoff_round in range(1, 7):
            states.append(
                initialize_partial_search_for_round(
                    all_region_teams=all_region_teams,
                    cutoff_round=cutoff_round,
                    prob_grid=grid,
                    name_to_idx=name_to_idx,
                )
            )

        weighted_states = []
        for state in states:
            cutoff_round = state["cutoff_round"]
            run_weight = PARTIAL_SEARCH_RUN_WEIGHTS.get(cutoff_round, 0)
            for _ in range(run_weight):
                weighted_states.append(state)

        print("Per-cycle search weights:")
        for cutoff_round in range(1, 7):
            print(f"  {ROUND_NAME[cutoff_round]:<10} x{PARTIAL_SEARCH_RUN_WEIGHTS[cutoff_round]}")
        print("")

        while True:
            for state in weighted_states:
                run_partial_search_step(
                    state=state,
                    all_region_teams=all_region_teams,
                    prob_grid=grid,
                    name_to_idx=name_to_idx,
                )
    except KeyboardInterrupt:
        print("\nStopped by user.")

        for cutoff_round in range(1, 6):
            saved = load_json(partial_save_path(cutoff_round))
            if saved is not None:
                print(f"{ROUND_NAME[cutoff_round]:<10} best EV saved: {float(saved['best_ev']):.4f}")

        title_saved = load_json(FULL_BRACKET_SAVE_PATH)
        if title_saved is not None:
            print(f"{ROUND_NAME[6]:<10} best EV saved: {float(title_saved['best_ev']):.4f}")


if __name__ == "__main__":
    main()
