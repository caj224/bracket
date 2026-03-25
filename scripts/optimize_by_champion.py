"""
Bracket EV optimization with a fixed champion constraint.

This script mirrors the search flow in optimize.py, but it only considers
brackets whose national champion matches the configured team below.
"""

import copy
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from data import load_bracket_data
from optimize import (
    BRACKET_SEED_ORDER,
    ELITE_MAX_FLIPS,
    ELITE_MIN_FLIPS,
    ELITE_POOL_SIZE,
    MAX_HILL_ITERS,
    P_GREEDY_RESTART,
    P_RANDOM_RESTART,
    PRINT_EVERY,
    REGION_ORDER,
    TOP_K,
    atomic_json_save,
    build_region_bracket,
    build_region_teams,
    favorite_pick,
    generate_one_flip_neighbors,
    greedy_pick,
    load_json,
    print_bracket,
    score_bracket_exact,
    weighted_pick,
)


CHAMPION_NAME = "Duke"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CHAMPION_RESULTS_DIR = RESULTS_DIR / "by_champion"


def slugify_team_name(team_name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in team_name).strip("_")


def save_path_for_champion(champion_name: str) -> str:
    return str(CHAMPION_RESULTS_DIR / f"best_bracket_{slugify_team_name(champion_name)}.json")


def top_k_path_for_champion(champion_name: str) -> str:
    return str(CHAMPION_RESULTS_DIR / f"top_brackets_{slugify_team_name(champion_name)}.json")


def optimize_round_with_picker(
    t1: dict,
    t2: dict,
    prob_grid,
    name_to_idx: Dict[str, int],
    round_num: int,
    picker: str,
) -> dict:
    if picker == "greedy":
        return greedy_pick(t1, t2, prob_grid, name_to_idx, round_num)
    if picker == "chalk":
        return favorite_pick(t1, t2, prob_grid, name_to_idx)
    if picker == "random":
        temp = random.choice([0.7, 0.85, 1.0, 1.15, 1.3])
        return weighted_pick(t1, t2, prob_grid, name_to_idx, temp)
    raise ValueError(f"Unknown picker: {picker}")


def find_team_by_name(teams_64: List[dict], champion_name: str) -> dict:
    champion_folded = champion_name.casefold()
    for team in teams_64:
        if team["name"].casefold() == champion_folded:
            return team

    available = ", ".join(sorted(team["name"] for team in teams_64))
    raise ValueError(f"Unknown champion '{champion_name}'. Available teams: {available}")


def round1_matchup_index(seed: int) -> int:
    return BRACKET_SEED_ORDER.index(seed) // 2


def champion_semi_slot(region: str) -> int:
    if region in {"East", "South"}:
        return 0
    return 1


def build_region_bracket_with_forced_champion(
    region_teams: List[dict],
    champion_team: dict,
    picker: str,
    prob_grid,
    name_to_idx: Dict[str, int],
) -> dict:
    champion_name = champion_team["name"]
    champion_seed = champion_team["seed"]

    r64 = []
    forced_r64_slot = round1_matchup_index(champion_seed)
    for slot in range(8):
        t1 = region_teams[2 * slot]
        t2 = region_teams[2 * slot + 1]
        if slot == forced_r64_slot:
            picked = champion_team
        else:
            picked = optimize_round_with_picker(t1, t2, prob_grid, name_to_idx, 1, picker)
        r64.append(picked)

    forced_r32_slot = forced_r64_slot // 2
    r32 = []
    for slot in range(4):
        t1 = r64[2 * slot]
        t2 = r64[2 * slot + 1]
        if slot == forced_r32_slot:
            picked = champion_team
        else:
            picked = optimize_round_with_picker(t1, t2, prob_grid, name_to_idx, 2, picker)
        r32.append(picked)

    forced_r16_slot = forced_r32_slot // 2
    r16 = []
    for slot in range(2):
        t1 = r32[2 * slot]
        t2 = r32[2 * slot + 1]
        if slot == forced_r16_slot:
            picked = champion_team
        else:
            picked = optimize_round_with_picker(t1, t2, prob_grid, name_to_idx, 3, picker)
        r16.append(picked)

    return {
        "teams": region_teams,
        "round_64_winners": r64,
        "round_32_winners": r32,
        "round_16_winners": r16,
        "region_champion": champion_team,
    }


def build_full_bracket_with_champion(
    all_region_teams: Dict[str, List[dict]],
    champion_team: dict,
    prob_grid,
    name_to_idx: Dict[str, int],
    picker: str = "greedy",
) -> dict:
    champion_region = champion_team["region"]
    regions = {}

    for region_name, teams in all_region_teams.items():
        if region_name == champion_region:
            regions[region_name] = build_region_bracket_with_forced_champion(
                teams,
                champion_team,
                picker,
                prob_grid,
                name_to_idx,
            )
        else:
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

    semi_slot = champion_semi_slot(champion_region)
    if semi_slot == 0:
        semi1 = champion_team
        semi2 = optimize_round_with_picker(west, midwest, prob_grid, name_to_idx, 5, picker)
    else:
        semi1 = optimize_round_with_picker(east, south, prob_grid, name_to_idx, 5, picker)
        semi2 = champion_team

    champ = champion_team

    return {
        "regions": regions,
        "final_four_winners": [semi1, semi2],
        "champion": champ,
    }


def validate_champion_constraint(full_bracket: dict, champion_name: str) -> None:
    if full_bracket["champion"]["name"] != champion_name:
        raise ValueError(
            f"Champion mismatch: expected {champion_name}, got {full_bracket['champion']['name']}"
        )


def best_one_flip_improvement_with_champion(
    full_bracket: dict,
    champion_name: str,
    prob_grid,
    name_to_idx: Dict[str, int],
):
    current_ev = score_bracket_exact(full_bracket, prob_grid, name_to_idx)

    best_ev = current_ev
    best_desc = None
    best_bracket = None

    for desc, candidate in generate_one_flip_neighbors(full_bracket, prob_grid, name_to_idx):
        if candidate["champion"]["name"] != champion_name:
            continue

        ev = score_bracket_exact(candidate, prob_grid, name_to_idx)
        if ev > best_ev:
            best_ev = ev
            best_desc = desc
            best_bracket = candidate

    return best_bracket, best_ev, best_desc


def improve_bracket_hill_climb_with_champion(
    start_bracket: dict,
    champion_name: str,
    prob_grid,
    name_to_idx: Dict[str, int],
    max_iters: int = 50,
    verbose: bool = False,
):
    current = copy.deepcopy(start_bracket)
    validate_champion_constraint(current, champion_name)

    current_ev = score_bracket_exact(current, prob_grid, name_to_idx)
    history = [float(current_ev)]

    if verbose:
        print(f"Starting EV: {current_ev:.4f}")

    for it in range(1, max_iters + 1):
        best_neighbor, best_neighbor_ev, best_desc = best_one_flip_improvement_with_champion(
            current,
            champion_name,
            prob_grid,
            name_to_idx,
        )

        if best_neighbor is None:
            if verbose:
                print(f"No improving constrained 1-flip neighbor at iteration {it}. Final EV: {current_ev:.4f}")
            break

        if verbose:
            improvement = best_neighbor_ev - current_ev
            print(f"Iter {it}: {best_desc} improved EV by {improvement:.4f} -> {best_neighbor_ev:.4f}")

        current = best_neighbor
        current_ev = best_neighbor_ev
        history.append(float(current_ev))

    return current, float(current_ev), history


def maybe_update_top_k_for_champion(
    bracket: dict,
    ev: float,
    source_mode: str,
    history: List[float],
    champion_name: str,
    k: int = TOP_K,
) -> None:
    top_k_path = top_k_path_for_champion(champion_name)
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


def save_best_result_for_champion(
    best_bracket: dict,
    best_ev: float,
    champion_name: str,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "best_ev": float(best_ev),
        "champion_name": champion_name,
        "extra": extra or {},
        "best_bracket": best_bracket,
    }
    atomic_json_save(payload, save_path_for_champion(champion_name))


def choose_elite_restart_for_champion(
    champion_name: str,
    elite_pool_size: int = ELITE_POOL_SIZE,
):
    board = load_json(top_k_path_for_champion(champion_name), default=[])
    if not board:
        return None

    usable = board[:elite_pool_size]
    picked = random.choice(usable)

    source_mode = picked["source_mode"]
    while source_mode.startswith("elite:"):
        source_mode = source_mode[len("elite:"):]

    return source_mode, copy.deepcopy(picked["bracket"])


def mutate_one_random_flip_with_champion(
    full_bracket: dict,
    champion_name: str,
    prob_grid,
    name_to_idx: Dict[str, int],
) -> dict:
    candidates = [
        candidate
        for _, candidate in generate_one_flip_neighbors(full_bracket, prob_grid, name_to_idx)
        if candidate["champion"]["name"] == champion_name
    ]

    if not candidates:
        return copy.deepcopy(full_bracket)

    return copy.deepcopy(random.choice(candidates))


def perturb_bracket_with_champion(
    bracket: dict,
    champion_name: str,
    prob_grid,
    name_to_idx: Dict[str, int],
    min_flips: int = 1,
    max_flips: int = 3,
) -> dict:
    out = copy.deepcopy(bracket)
    n_flips = random.randint(min_flips, max_flips)

    for _ in range(n_flips):
        out = mutate_one_random_flip_with_champion(out, champion_name, prob_grid, name_to_idx)

    return out


def build_restart_bracket_with_champion(
    all_region_teams: Dict[str, List[dict]],
    champion_team: dict,
    prob_grid,
    name_to_idx: Dict[str, int],
):
    u = random.random()

    if u < P_RANDOM_RESTART:
        bracket = build_full_bracket_with_champion(
            all_region_teams,
            champion_team,
            prob_grid,
            name_to_idx,
            picker="random",
        )
        return "random", bracket

    if u < P_RANDOM_RESTART + P_GREEDY_RESTART:
        bracket = build_full_bracket_with_champion(
            all_region_teams,
            champion_team,
            prob_grid,
            name_to_idx,
            picker="greedy",
        )
        return "greedy", bracket

    elite_pick = choose_elite_restart_for_champion(champion_team["name"])
    if elite_pick is not None:
        elite_source_mode, elite_bracket = elite_pick
        return f"elite:{elite_source_mode}", elite_bracket

    bracket = build_full_bracket_with_champion(
        all_region_teams,
        champion_team,
        prob_grid,
        name_to_idx,
        picker="random",
    )
    return "random", bracket


def overnight_search_with_champion(
    all_region_teams: Dict[str, List[dict]],
    champion_team: dict,
    prob_grid,
    name_to_idx: Dict[str, int],
):
    champion_name = champion_team["name"]
    save_path = save_path_for_champion(champion_name)
    top_k_path = top_k_path_for_champion(champion_name)
    saved = load_json(save_path)

    if saved is not None:
        best_ev = float(saved["best_ev"])
        best_bracket = saved["best_bracket"]
        validate_champion_constraint(best_bracket, champion_name)
        print(f"Loaded previous best for champion {champion_name}: {best_ev:.4f}")
    else:
        best_bracket = build_full_bracket_with_champion(
            all_region_teams,
            champion_team,
            prob_grid,
            name_to_idx,
            picker="greedy",
        )
        best_ev = score_bracket_exact(best_bracket, prob_grid, name_to_idx)
        save_best_result_for_champion(best_bracket, best_ev, champion_name, extra={"init": "greedy"})
        maybe_update_top_k_for_champion(best_bracket, best_ev, "greedy_init", [float(best_ev)], champion_name)
        print(f"Initialized best EV for champion {champion_name}: {best_ev:.4f}")

    print(f"Saving best results to: {save_path}")
    print(f"Saving top {TOP_K} constrained brackets to: {top_k_path}")

    starts_completed = 0

    try:
        while True:
            starts_completed += 1

            start_mode, start_bracket = build_restart_bracket_with_champion(
                all_region_teams,
                champion_team,
                prob_grid,
                name_to_idx,
            )

            if start_mode.startswith("elite:"):
                start_bracket = perturb_bracket_with_champion(
                    start_bracket,
                    champion_name,
                    prob_grid,
                    name_to_idx,
                    min_flips=ELITE_MIN_FLIPS,
                    max_flips=ELITE_MAX_FLIPS,
                )

            start_ev = score_bracket_exact(start_bracket, prob_grid, name_to_idx)

            final_bracket, final_ev, history = improve_bracket_hill_climb_with_champion(
                start_bracket,
                champion_name,
                prob_grid,
                name_to_idx,
                max_iters=MAX_HILL_ITERS,
                verbose=False,
            )

            maybe_update_top_k_for_champion(final_bracket, final_ev, start_mode, history, champion_name)

            if final_ev > best_ev:
                best_ev = final_ev
                best_bracket = final_bracket

                save_best_result_for_champion(
                    best_bracket,
                    best_ev,
                    champion_name,
                    extra={
                        "starts_completed": starts_completed,
                        "start_mode": start_mode,
                        "start_ev": float(start_ev),
                        "history": history,
                    },
                )

                print(
                    f"[{starts_completed}] NEW BEST | "
                    f"champion={champion_name:<20} | "
                    f"mode={start_mode:<14} | "
                    f"start_ev={start_ev:.4f} | "
                    f"best_ev={best_ev:.4f}"
                )
            elif starts_completed % PRINT_EVERY == 0:
                print(
                    f"[{starts_completed}] no new best | "
                    f"champion={champion_name:<20} | "
                    f"mode={start_mode:<14} | "
                    f"start_ev={start_ev:.4f} | "
                    f"final_ev={final_ev:.4f} | "
                    f"current_best={best_ev:.4f}"
                )

    except KeyboardInterrupt:
        print("\nStopped by user.")
        print(f"Best EV saved in {save_path}: {best_ev:.4f}")
        return best_bracket, best_ev


def show_saved_best_for_champion(champion_name: str) -> None:
    saved = load_json(save_path_for_champion(champion_name))
    if saved is None:
        print(f"No saved best bracket found for champion {champion_name}.")
        return

    print(f"Saved timestamp: {saved['timestamp']}")
    print(f"Saved best EV:   {saved['best_ev']:.4f}")
    print(f"Champion:        {saved['champion_name']}")
    print_bracket(saved["best_bracket"])


def main() -> None:
    teams_64, grid = load_bracket_data()
    all_region_teams = build_region_teams(teams_64)
    name_to_idx = {team["name"]: i for i, team in enumerate(teams_64)}
    champion_team = find_team_by_name(teams_64, CHAMPION_NAME)

    print(f"Starting constrained search with champion fixed to: {champion_team['name']}")
    print("Press Ctrl+C to stop.\n")

    overnight_search_with_champion(all_region_teams, champion_team, grid, name_to_idx)


if __name__ == "__main__":
    main()
