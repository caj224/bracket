import math
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests


# ── Constants ──────────────────────────────────────────────────────────────────

REGIONS = ["East", "West", "South", "Midwest"]

SEED_BPI_FALLBACK = {
    11: 10.0,
    12: 8.0,
    13: 5.0,
    14: 2.0,
    15: -2.0,
    16: -8.0,
}

SCALING_FACTOR = 0.15

MANUAL_BPI = {
    "Northern Iowa": 7.4,
    "CA Baptist": 1.6,
    "N Dakota St": 1.9,
    "Furman": -0.1,
    "Siena": -1.8,
    "Miami OH": 6.2,
    "Akron": 7.7,
    "Hofstra": 4.6,
    "Wright St": 1.5,
    "Tennessee St": -2.6,
    "Howard": -2.5,
    "Troy": 1.9,
    "Penn": -0.1,
    "Idaho": -0.6,
    "Prairie View": -7.8,
    "Hawai'i": 2.8,
    "Kennesaw St": 2.5,
    "Queens": -1.8,
    "Long Island": -3.3,
}

FIRST_FOUR_MATCHUPS = [
    {
        "team1": "SMU",
        "team2": "Miami OH",
        "seed": 11,
        "region": "Midwest",
    },
    {
        "team1": "Lehigh",
        "team2": "Prairie View",
        "seed": 16,
        "region": "South",
    },
]


# ── Fetching ───────────────────────────────────────────────────────────────────

def fetch_bpi_data(limit: int = 64) -> Dict[str, Any]:
    """Fetch ESPN BPI team data."""
    url = (
        "https://site.web.api.espn.com/apis/fitt/v3/sports/basketball/"
        "mens-college-basketball/powerindex"
    )
    response = requests.get(url, params={"limit": limit}, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_tournament_events(
    main_dates: str = "20260319-20260407",
    first_four_dates: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch tournament scoreboard events and deduplicate by event id."""
    if first_four_dates is None:
        first_four_dates = ["20260318", "20260319"]

    all_events: List[Dict[str, Any]] = []

    response = requests.get(
        "https://site.api.espn.com/apis/site/v2/sports/basketball/"
        "mens-college-basketball/scoreboard",
        params={"groups": 100, "limit": 200, "dates": main_dates},
        timeout=30,
    )
    response.raise_for_status()
    all_events.extend(response.json().get("events", []))

    for date in first_four_dates:
        response = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/"
            "mens-college-basketball/scoreboard",
            params={"groups": 100, "limit": 200, "dates": date},
            timeout=30,
        )
        response.raise_for_status()
        all_events.extend(response.json().get("events", []))

    seen_ids = set()
    unique_events = []
    for event in all_events:
        event_id = event.get("id")
        if event_id and event_id not in seen_ids:
            seen_ids.add(event_id)
            unique_events.append(event)

    return unique_events


# ── Parsing helpers ────────────────────────────────────────────────────────────

def extract_region_from_note(note: str) -> str:
    """Extract bracket region from ESPN note headline."""
    for part in note.split(" - "):
        for region in REGIONS:
            if region in part:
                return region
    return "Unknown"


def parse_bpi_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse ESPN BPI feed into a tournament-team list."""
    teams = []

    for entry in data.get("teams", []):
        team = entry.get("team", {})
        categories = {
            cat["name"]: cat
            for cat in entry.get("categories", [])
            if "name" in cat
        }

        tournament = categories.get("tournament", {})
        tournament_vals = tournament.get("values", [])
        bpi_category = categories.get("bpi", {})
        bpi_vals = bpi_category.get("values", [])

        if not tournament_vals or tournament_vals[0] is None:
            continue
        if not bpi_vals:
            continue

        region = "Unknown"
        totals = tournament.get("totals", [])
        if len(totals) > 2:
            region = totals[2]

        teams.append(
            {
                "id": team.get("id"),
                "name": team.get("shortDisplayName"),
                "seed": int(tournament_vals[0]),
                "bpi": bpi_vals[0],
                "region": region,
                "prob_r64": tournament_vals[8] if len(tournament_vals) > 8 else None,
                "prob_r32": tournament_vals[7] if len(tournament_vals) > 7 else None,
                "prob_r16": tournament_vals[6] if len(tournament_vals) > 6 else None,
                "prob_r8": tournament_vals[5] if len(tournament_vals) > 5 else None,
                "prob_r4": tournament_vals[4] if len(tournament_vals) > 4 else None,
                "prob_final": tournament_vals[3] if len(tournament_vals) > 3 else None,
            }
        )

    return teams


def parse_bracket_games(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse scoreboard events into a simpler bracket-game structure."""
    games = []

    for event in events:
        competition = event["competitions"][0]
        note = competition.get("notes", [{}])[0].get("headline", "")
        competitors = competition.get("competitors", [])

        teams = []
        for competitor in competitors:
            teams.append(
                {
                    "id": competitor["team"]["id"],
                    "name": competitor["team"]["shortDisplayName"],
                    "seed": competitor.get("curatedRank", {}).get("current"),
                }
            )

        games.append(
            {
                "game_id": event["id"],
                "note": note,
                "teams": teams,
            }
        )

    return games


# ── Team assembly ──────────────────────────────────────────────────────────────

def build_full_team_list(
    bpi_teams: List[Dict[str, Any]],
    games: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Combine BPI teams with bracket field from scoreboard data."""
    bpi_lookup = {team["id"]: team for team in bpi_teams}

    seen_ids = set()
    all_teams = []

    for game in games:
        note = game["note"]
        if "1st Round" not in note and "First Four" not in note:
            continue

        region = extract_region_from_note(note)

        for team in game["teams"]:
            team_id = team["id"]
            team_name = team["name"]

            if team_id in seen_ids or team_name == "TBD":
                continue

            seen_ids.add(team_id)

            if team_id in bpi_lookup:
                bpi_entry = bpi_lookup[team_id]
                all_teams.append(
                    {
                        "id": team_id,
                        "name": team_name,
                        "seed": team["seed"],
                        "region": bpi_entry["region"],
                        "bpi": bpi_entry["bpi"],
                        "prob_r64": bpi_entry["prob_r64"],
                        "prob_r32": bpi_entry["prob_r32"],
                        "prob_r16": bpi_entry["prob_r16"],
                        "prob_r8": bpi_entry["prob_r8"],
                        "prob_r4": bpi_entry["prob_r4"],
                        "prob_final": bpi_entry["prob_final"],
                        "bpi_source": "espn",
                    }
                )
            else:
                fallback_bpi = SEED_BPI_FALLBACK.get(team["seed"], 0.0)
                all_teams.append(
                    {
                        "id": team_id,
                        "name": team_name,
                        "seed": team["seed"],
                        "region": region,
                        "bpi": fallback_bpi,
                        "prob_r64": None,
                        "prob_r32": None,
                        "prob_r16": None,
                        "prob_r8": None,
                        "prob_r4": None,
                        "prob_final": None,
                        "bpi_source": "fallback",
                    }
                )

    return all_teams


def apply_manual_bpi_overrides(
    teams: List[Dict[str, Any]],
    manual_bpi: Dict[str, float],
) -> None:
    """Modify team BPI values in place using manual overrides."""
    for team in teams:
        if team["name"] in manual_bpi:
            team["bpi"] = manual_bpi[team["name"]]
            team["bpi_source"] = "manual"


def resolve_first_four(
    teams: List[Dict[str, Any]],
    results: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Resolve First Four teams down to a 64-team bracket.

    Parameters
    ----------
    teams
        Full team list including both sides of First Four play-in games.
    results
        Optional mapping like:
        {
            "SMU": "Miami OH",
            "Lehigh": "Prairie View"
        }
        meaning the named team lost and the mapped team advanced.

        If results is None or a matchup is missing, the winner is simulated using BPI.
    """
    team_lookup = {team["name"]: team for team in teams}

    first_four_names = {
        matchup["team1"] for matchup in FIRST_FOUR_MATCHUPS
    } | {
        matchup["team2"] for matchup in FIRST_FOUR_MATCHUPS
    }

    resolved = [team for team in teams if team["name"] not in first_four_names]

    for matchup in FIRST_FOUR_MATCHUPS:
        team1_name = matchup["team1"]
        team2_name = matchup["team2"]

        team1 = team_lookup.get(team1_name)
        team2 = team_lookup.get(team2_name)

        if team1 is None or team2 is None:
            raise ValueError(
                f"Missing First Four team in input: {team1_name} vs {team2_name}"
            )

        winner = None
        if results is not None:
            if team1_name in results:
                winner_name = results[team1_name]
                winner = team_lookup.get(winner_name)
            elif team2_name in results:
                winner_name = results[team2_name]
                winner = team_lookup.get(winner_name)

        if winner is None:
            prob_team1 = bpi_to_prob(team1["bpi"], team2["bpi"])
            winner = team1 if random.random() < prob_team1 else team2

        resolved.append(winner)

    return resolved


# ── Core probability ───────────────────────────────────────────────────────────

def bpi_to_prob(bpi1: float, bpi2: float) -> float:
    """Convert BPI difference into win probability via logistic transform."""
    diff = bpi1 - bpi2
    return 1 / (1 + math.exp(-diff * SCALING_FACTOR))


def build_prob_grid(teams: List[Dict[str, Any]]) -> np.ndarray:
    """Build NxN matrix where grid[i, j] = P(team i beats team j)."""
    n_teams = len(teams)
    grid = np.zeros((n_teams, n_teams))

    for i in range(n_teams):
        for j in range(n_teams):
            if i == j:
                grid[i, j] = 0.5
            else:
                grid[i, j] = bpi_to_prob(teams[i]["bpi"], teams[j]["bpi"])

    return grid


# ── Optional inspection/export helpers ─────────────────────────────────────────

def teams_to_dataframe(teams: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert team list to a sorted DataFrame."""
    df = pd.DataFrame(
        [
            {
                "id": team["id"],
                "team": team["name"],
                "seed": team["seed"],
                "region": team["region"],
                "bpi": team["bpi"],
                "bpi_source": team["bpi_source"],
                "prob_r64": team["prob_r64"],
                "prob_r32": team["prob_r32"],
                "prob_r16": team["prob_r16"],
                "prob_r8": team["prob_r8"],
                "prob_r4": team["prob_r4"],
                "prob_final": team["prob_final"],
            }
            for team in teams
        ]
    )

    return df.sort_values(["region", "seed"]).reset_index(drop=True)

def load_bracket_data():
    bpi_raw = fetch_bpi_data()
    bpi_teams = parse_bpi_data(bpi_raw)

    events = fetch_tournament_events()
    games = parse_bracket_games(events)

    teams = build_full_team_list(bpi_teams, games)
    apply_manual_bpi_overrides(teams, MANUAL_BPI)

    first_four_results = {
        "SMU": "Miami OH",
        "Lehigh": "Prairie View",
    }

    teams_64 = resolve_first_four(teams, first_four_results)
    prob_grid = build_prob_grid(teams_64)

    return teams_64, prob_grid
# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    bpi_raw = fetch_bpi_data()
    bpi_teams = parse_bpi_data(bpi_raw)

    events = fetch_tournament_events()
    games = parse_bracket_games(events)

    teams = build_full_team_list(bpi_teams, games)
    apply_manual_bpi_overrides(teams, MANUAL_BPI)

    first_four_results = {
        "SMU": "Miami OH",
        "Lehigh": "Prairie View",
    }

    teams_64 = resolve_first_four(teams, first_four_results)
    team_df = teams_to_dataframe(teams_64)
    prob_grid = build_prob_grid(teams_64)

    print(f"Teams: {len(teams_64)}")
    print(f"Grid shape: {prob_grid.shape}")
    print(f"NaN in grid: {np.sum(np.isnan(prob_grid))}")
    print(team_df[["team", "seed", "region", "bpi", "bpi_source"]].to_string(index=False))


if __name__ == "__main__":
    main()