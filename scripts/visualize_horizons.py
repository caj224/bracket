"""
Create an HTML dashboard for the saved horizon-optimized brackets.

The dashboard reads:
    results/partial/best_partial_round_{1..5}.json
    results/full/best_bracket.json for the Title horizon

It then writes a single visual report to:
    results/partial/horizon_dashboard.html
"""

import json
import math
from html import escape
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PARTIAL_RESULTS_DIR = PROJECT_ROOT / "results" / "partial"
FULL_RESULTS_DIR = PROJECT_ROOT / "results" / "full"
OUTPUT_PATH = PARTIAL_RESULTS_DIR / "horizon_dashboard.html"
DOCS_OUTPUT_PATH = PROJECT_ROOT / "docs" / "index.html"

ROUND_LABELS = {
    1: "R64",
    2: "R32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Title",
}

ROUND_KEYS = {
    1: "round_64_winners",
    2: "round_32_winners",
    3: "round_16_winners",
    4: "region_champion",
    5: "final_four_winners",
    6: "champion",
}

REGION_ORDER = ["East", "West", "South", "Midwest"]
SCALING_FACTOR = 0.15
SERIES_COLORS = ["#38bdf8", "#f97316", "#22c55e", "#facc15", "#fb7185", "#a78bfa"]


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def bpi_to_prob(bpi1: float, bpi2: float) -> float:
    diff = bpi1 - bpi2
    return 1 / (1 + math.exp(-diff * SCALING_FACTOR))


def build_prob_grid_from_bracket(bracket: dict):
    teams = []
    for region in REGION_ORDER:
        teams.extend(bracket["regions"][region]["teams"])

    name_to_idx = {team["name"]: i for i, team in enumerate(teams)}
    grid = [[0.5 for _ in teams] for _ in teams]

    for i, team1 in enumerate(teams):
        for j, team2 in enumerate(teams):
            if i == j:
                continue
            grid[i][j] = bpi_to_prob(team1["bpi"], team2["bpi"])

    return teams, grid, name_to_idx


def load_horizon_payloads() -> List[dict]:
    payloads = []
    for cutoff_round in range(1, 6):
        path = PARTIAL_RESULTS_DIR / f"best_partial_round_{cutoff_round}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        payload["path"] = str(path)
        payloads.append(payload)

    full_path = FULL_RESULTS_DIR / "best_bracket.json"
    full_payload = load_json(full_path)
    if full_payload is not None:
        full_payload["best_bracket"]["cutoff_round"] = 6
        full_payload["path"] = str(full_path)
        payloads.append(full_payload)

    if not payloads:
        raise FileNotFoundError("No saved partial horizon files were found in results/partial.")

    return payloads


def team_seed_lookup(payloads: List[dict]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for payload in payloads:
        for region_data in payload["best_bracket"]["regions"].values():
            for team in region_data["teams"]:
                lookup[team["name"]] = team["seed"]
    return lookup


def pick_names(value) -> List[str]:
    if isinstance(value, list):
        return [team["name"] if isinstance(team, dict) else str(team) for team in value]
    if isinstance(value, dict):
        return [value["name"]]
    if value is None:
        return []
    return [str(value)]


def extract_round_picks(bracket: dict, round_num: int) -> List[str]:
    key = ROUND_KEYS[round_num]
    if round_num <= 4:
        out: List[str] = []
        for region in REGION_ORDER:
            region_data = bracket["regions"][region]
            if key not in region_data:
                continue
            out.extend(pick_names(region_data[key]))
        return out

    if key not in bracket:
        return []
    return pick_names(bracket[key])


def extract_signature_picks(bracket: dict) -> Dict[str, List[str]]:
    signature = {}
    for round_num in range(6, 0, -1):
        picks = extract_round_picks(bracket, round_num)
        if picks:
            signature[ROUND_LABELS[round_num]] = picks
    return signature


def diff_common_rounds(prev_bracket: dict, curr_bracket: dict) -> List[Tuple[str, int, List[str]]]:
    changes = []
    max_round = min(prev_bracket["cutoff_round"], curr_bracket["cutoff_round"])

    for round_num in range(1, max_round + 1):
        prev_picks = extract_round_picks(prev_bracket, round_num)
        curr_picks = extract_round_picks(curr_bracket, round_num)
        limit = min(len(prev_picks), len(curr_picks))
        changed = []
        for idx in range(limit):
            if prev_picks[idx] != curr_picks[idx]:
                changed.append(f"{prev_picks[idx]} -> {curr_picks[idx]}")
        if changed:
            changes.append((ROUND_LABELS[round_num], len(changed), changed[:6]))

    return changes


def cumulative_ev_by_round(bracket: dict) -> List[float]:
    cutoff_round = bracket["cutoff_round"]
    teams, prob_grid, name_to_idx = build_prob_grid_from_bracket(bracket)
    team_lookup = {team["name"]: team for team in teams}
    per_round = [0.0] * 6
    round_params = {
        1: (1, 1),
        2: (2, 2),
        3: (4, 3),
        4: (8, 4),
        5: (16, 5),
        6: (32, 6),
    }

    def add_round_ev(round_num: int, amount: float) -> None:
        per_round[round_num - 1] += amount

    def matchup_ev(left_dist: Dict[str, float], right_dist: Dict[str, float], picked_name: str, round_num: int) -> float:
        base, bonus = round_params[round_num]
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

    for region_name, region_data in bracket["regions"].items():
        region_teams = region_data["teams"]
        r64_inputs = [(region_teams[i], region_teams[i + 1]) for i in range(0, 16, 2)]
        r64_winner_dists = []

        if cutoff_round >= 1:
            for slot, picked in enumerate(region_data["round_64_winners"]):
                left_dist, right_dist = winner_dist_two_teams(*r64_inputs[slot])
                add_round_ev(1, matchup_ev(left_dist, right_dist, picked["name"], 1))

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
                add_round_ev(2, matchup_ev(left_dist, right_dist, picked["name"], 2))

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
                add_round_ev(3, matchup_ev(left_dist, right_dist, picked["name"], 3))

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
            add_round_ev(4, matchup_ev(left_dist, right_dist, picked["name"], 4))

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
        ff = bracket["final_four_winners"]

        left_dist = region_champ_dists["East"]
        right_dist = region_champ_dists["South"]
        add_round_ev(5, matchup_ev(left_dist, right_dist, ff[0]["name"], 5))

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
        add_round_ev(5, matchup_ev(left_dist, right_dist, ff[1]["name"], 5))

        semi2_dist = {}
        for left_name, p_left in left_dist.items():
            i = name_to_idx[left_name]
            for right_name, p_right in right_dist.items():
                j = name_to_idx[right_name]
                p_match = p_left * p_right
                semi2_dist[left_name] = semi2_dist.get(left_name, 0.0) + p_match * prob_grid[i][j]
                semi2_dist[right_name] = semi2_dist.get(right_name, 0.0) + p_match * prob_grid[j][i]

    if cutoff_round >= 6:
        champ = bracket["champion"]
        add_round_ev(6, matchup_ev(semi1_dist, semi2_dist, champ["name"], 6))

    running = []
    total = 0.0
    for round_num in range(1, 7):
        if round_num <= cutoff_round:
            total += per_round[round_num - 1]
            running.append(total)
        else:
            running.append(None)

    return running


def build_ev_svg(values: List[float], labels: List[str]) -> str:
    width = 920
    height = 260
    left = 56
    right = 24
    top = 20
    bottom = 40
    inner_w = width - left - right
    inner_h = height - top - bottom

    min_v = min(values)
    max_v = max(values)
    if math.isclose(min_v, max_v):
        min_v -= 1
        max_v += 1

    def x_pos(i: int) -> float:
        if len(values) == 1:
            return left + inner_w / 2
        return left + inner_w * i / (len(values) - 1)

    def y_pos(v: float) -> float:
        frac = (v - min_v) / (max_v - min_v)
        return top + inner_h * (1 - frac)

    points = " ".join(f"{x_pos(i):.1f},{y_pos(v):.1f}" for i, v in enumerate(values))
    area_points = f"{left},{top + inner_h} {points} {left + inner_w},{top + inner_h}"

    grid_lines = []
    for step in range(5):
        y = top + inner_h * step / 4
        val = max_v - (max_v - min_v) * step / 4
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + inner_w}" y2="{y:.1f}" '
            f'stroke="rgba(148,163,184,0.22)" stroke-width="1" />'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#94a3b8" font-size="12">{val:.1f}</text>'
        )

    tick_labels = []
    for i, label in enumerate(labels):
        x = x_pos(i)
        tick_labels.append(
            f'<text x="{x:.1f}" y="{height - 12}" text-anchor="middle" '
            f'fill="#94a3b8" font-size="12">{escape(label)}</text>'
        )

    point_nodes = []
    for i, value in enumerate(values):
        x = x_pos(i)
        y = y_pos(value)
        point_nodes.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#f97316" />'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="rgba(249,115,22,0.18)" />'
            f'<text x="{x:.1f}" y="{y - 14:.1f}" text-anchor="middle" '
            f'fill="#e2e8f0" font-size="12">{value:.2f}</text>'
        )

    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Expected value by horizon">
      <defs>
        <linearGradient id="evArea" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="rgba(14,165,233,0.45)"></stop>
          <stop offset="100%" stop-color="rgba(14,165,233,0.02)"></stop>
        </linearGradient>
      </defs>
      {''.join(grid_lines)}
      <polygon points="{area_points}" fill="url(#evArea)"></polygon>
      <polyline points="{points}" fill="none" stroke="#38bdf8" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></polyline>
      {''.join(point_nodes)}
      {''.join(tick_labels)}
    </svg>
    """


def build_bracket_curve_svg(series_rows: List[dict]) -> str:
    width = 920
    height = 320
    left = 64
    right = 32
    top = 28
    bottom = 48
    inner_w = width - left - right
    inner_h = height - top - bottom
    x_labels = [ROUND_LABELS[i] for i in range(1, 7)]

    numeric_values = [
        value
        for row in series_rows
        for value in row["curve"]
        if value is not None
    ]
    min_v = min(numeric_values)
    max_v = max(numeric_values)
    if math.isclose(min_v, max_v):
        min_v -= 1
        max_v += 1

    def x_pos(i: int) -> float:
        if len(x_labels) == 1:
            return left + inner_w / 2
        return left + inner_w * i / (len(x_labels) - 1)

    def y_pos(v: float) -> float:
        frac = (v - min_v) / (max_v - min_v)
        return top + inner_h * (1 - frac)

    grid_lines = []
    for step in range(5):
        y = top + inner_h * step / 4
        val = max_v - (max_v - min_v) * step / 4
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + inner_w}" y2="{y:.1f}" stroke="rgba(148,163,184,0.22)" stroke-width="1" />'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" fill="#94a3b8" font-size="12">{val:.1f}</text>'
        )

    ticks = []
    for i, label in enumerate(x_labels):
        x = x_pos(i)
        ticks.append(
            f'<line x1="{x:.1f}" y1="{top + inner_h}" x2="{x:.1f}" y2="{top + inner_h + 6}" stroke="rgba(148,163,184,0.34)" stroke-width="1" />'
            f'<text x="{x:.1f}" y="{height - 14}" text-anchor="middle" fill="#94a3b8" font-size="12">{escape(label)}</text>'
        )

    series_nodes = []
    legend = []
    for idx, row in enumerate(series_rows):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        points = [
            (x_pos(i), y_pos(value))
            for i, value in enumerate(row["curve"])
            if value is not None
        ]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        circles = []
        for x, y in points:
            circles.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" />'
            )
        series_nodes.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" />'
            f"{''.join(circles)}"
        )
        legend.append(
            f'<div class="legend-item"><span class="legend-swatch" style="background:{color}"></span>{escape(row["label"])}</div>'
        )

    return f"""
    <div class="legend-row">{''.join(legend)}</div>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Cumulative expected value by round for each optimized bracket">
      {''.join(grid_lines)}
      {''.join(ticks)}
      {''.join(series_nodes)}
    </svg>
    """


def build_curve_table(series_rows: List[dict]) -> str:
    headers = "".join(f"<th>{escape(ROUND_LABELS[i])}</th>" for i in range(1, 7))
    body_rows = []

    for idx, row in enumerate(series_rows):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        values = []
        for value in row["curve"]:
            if value is None:
                values.append('<td class="empty">-</td>')
            else:
                values.append(f"<td>{value:.2f}</td>")

        body_rows.append(
            f"""
            <tr>
              <th scope="row">
                <span class="table-label">
                  <span class="legend-swatch" style="background:{color}"></span>
                  {escape(row["label"])}
                </span>
              </th>
              {''.join(values)}
            </tr>
            """
        )

    return f"""
    <div class="table-wrap">
      <table class="curve-table">
        <thead>
          <tr>
            <th>Bracket</th>
            {headers}
          </tr>
        </thead>
        <tbody>
          {''.join(body_rows)}
        </tbody>
      </table>
    </div>
    """


def deepest_round_by_team(bracket: dict) -> Dict[str, int]:
    depths = {}
    for round_num in range(1, 7):
        for name in extract_round_picks(bracket, round_num):
            depths[name] = round_num

    for region in REGION_ORDER:
        for team in bracket["regions"][region]["teams"]:
            depths.setdefault(team["name"], 0)

    return depths


def build_survival_matrix(payloads: List[dict], seed_lookup: Dict[str, int]) -> str:
    round_order = list(range(1, 7))
    rows = []
    team_rows = []

    for payload in payloads:
        bracket = payload["best_bracket"]
        team_rows.append(
            {
                "label": ROUND_LABELS[bracket["cutoff_round"]],
                "depths": deepest_round_by_team(bracket),
            }
        )

    all_teams = sorted(seed_lookup.keys(), key=lambda name: (seed_lookup[name], name))
    team_summaries = []
    for team in all_teams:
        values = [row["depths"].get(team, 0) for row in team_rows]
        team_summaries.append(
            {
                "name": team,
                "seed": seed_lookup[team],
                "values": values,
                "best": max(values),
                "avg": sum(values) / len(values),
            }
        )

    team_summaries.sort(key=lambda row: (-row["best"], -row["avg"], row["seed"], row["name"]))

    for row in team_summaries:
        cells = []
        for depth in row["values"]:
            label = ROUND_LABELS[depth] if depth else "Out"
            cls = f"depth-{depth}"
            cells.append(f'<td class="{cls}">{escape(label)}</td>')

        rows.append(
            f"""
            <tr>
              <th scope="row">
                <span class="team-label">({row["seed"]}) {escape(row["name"])}</span>
              </th>
              {''.join(cells)}
            </tr>
            """
        )

    headers = "".join(f"<th>{escape(ROUND_LABELS[r])}</th>" for r in round_order)
    return f"""
    <div class="table-wrap">
      <table class="matrix-table">
        <thead>
          <tr>
            <th>Team</th>
            {headers}
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """


def similarity_percent(bracket_a: dict, bracket_b: dict) -> float:
    total = 0
    same = 0
    max_round = min(bracket_a["cutoff_round"], bracket_b["cutoff_round"])
    for round_num in range(1, max_round + 1):
        picks_a = extract_round_picks(bracket_a, round_num)
        picks_b = extract_round_picks(bracket_b, round_num)
        for a, b in zip(picks_a, picks_b):
            total += 1
            if a == b:
                same += 1
    return 100.0 * same / total if total else 0.0


def heatmap_color(pct: float) -> str:
    alpha = 0.14 + 0.56 * (pct / 100.0)
    return f"rgba(56, 189, 248, {alpha:.3f})"


def build_similarity_heatmap(payloads: List[dict]) -> str:
    labels = [ROUND_LABELS[p["best_bracket"]["cutoff_round"]] for p in payloads]
    rows = []
    header = "".join(f"<th>{escape(label)}</th>" for label in labels)

    for i, payload_a in enumerate(payloads):
        cells = []
        for j, payload_b in enumerate(payloads):
            pct = similarity_percent(payload_a["best_bracket"], payload_b["best_bracket"])
            style = f' style="background:{heatmap_color(pct)}"'
            cells.append(f'<td{style}>{pct:.0f}%</td>')
        rows.append(f"<tr><th>{escape(labels[i])}</th>{''.join(cells)}</tr>")

    return f"""
    <div class="table-wrap">
      <table class="heatmap-table">
        <thead>
          <tr>
            <th>Horizon</th>
            {header}
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """


def build_round_chips(picks: List[str], seed_lookup: Dict[str, int]) -> str:
    if not picks:
        return '<span class="muted">Not set yet</span>'

    chips = []
    for name in picks:
        seed = seed_lookup.get(name, "?")
        chips.append(
            f'<span class="chip"><span class="seed">{seed}</span>{escape(name)}</span>'
        )
    return "".join(chips)


def build_calculator_bootstrap(bracket: dict) -> str:
    teams_by_region = {
        region: [
            {
                "name": team["name"],
                "seed": team["seed"],
                "bpi": team["bpi"],
            }
            for team in bracket["regions"][region]["teams"]
        ]
        for region in REGION_ORDER
    }

    optimized_picks = {
        "regions": {},
        "final_four_winners": pick_names(bracket.get("final_four_winners", [])),
        "champion": pick_names(bracket.get("champion")),
    }

    for region in REGION_ORDER:
        region_data = bracket["regions"][region]
        optimized_picks["regions"][region] = {
            "round_64_winners": pick_names(region_data.get("round_64_winners", [])),
            "round_32_winners": pick_names(region_data.get("round_32_winners", [])),
            "round_16_winners": pick_names(region_data.get("round_16_winners", [])),
            "region_champion": pick_names(region_data.get("region_champion")),
        }

    bootstrap = {
        "regions": REGION_ORDER,
        "roundLabels": ROUND_LABELS,
        "teamsByRegion": teams_by_region,
        "optimized": optimized_picks,
        "scalingFactor": SCALING_FACTOR,
        "roundParams": {
            "1": [1, 1],
            "2": [2, 2],
            "3": [4, 3],
            "4": [8, 4],
            "5": [16, 5],
            "6": [32, 6],
        },
    }
    return json.dumps(bootstrap).replace("</", "<\\/")


def render_dashboard(payloads: List[dict]) -> str:
    seed_lookup = team_seed_lookup(payloads)

    horizon_rows = []
    ev_values = []
    ev_labels = []
    curve_rows = []

    for payload in payloads:
        bracket = payload["best_bracket"]
        cutoff_round = bracket["cutoff_round"]
        ev = float(payload["best_ev"])
        ev_values.append(ev)
        ev_labels.append(ROUND_LABELS[cutoff_round])
        curve_rows.append(
            {
                "label": f"{ROUND_LABELS[cutoff_round]} bracket",
                "curve": cumulative_ev_by_round(bracket),
            }
        )

        signature = extract_signature_picks(bracket)
        top_round = next(iter(signature))
        top_picks = signature[top_round]

        secondary_round = None
        secondary_picks: List[str] = []
        for round_name, picks in list(signature.items())[1:]:
            secondary_round = round_name
            secondary_picks = picks
            break

        horizon_rows.append(
            f"""
            <section class="card horizon-card">
              <div class="card-top">
                <div>
                  <div class="eyebrow">Cutoff Round {cutoff_round}</div>
                  <h3>{escape(ROUND_LABELS[cutoff_round])}</h3>
                </div>
                <div class="ev-pill">EV {ev:.2f}</div>
              </div>
              <div class="meta-row">
                <span>Saved from <code>{escape(Path(payload['path']).name)}</code></span>
              </div>
              <div class="pick-block">
                <div class="pick-label">{escape(top_round)} outlook</div>
                <div class="chip-row">{build_round_chips(top_picks, seed_lookup)}</div>
              </div>
              <div class="pick-block">
                <div class="pick-label">{escape(secondary_round) if secondary_round else 'Earlier picks'} </div>
                <div class="chip-row">{build_round_chips(secondary_picks, seed_lookup) if secondary_round else '<span class="muted">Early-round view only</span>'}</div>
              </div>
            </section>
            """
        )

    transition_rows = []
    for idx in range(1, len(payloads)):
        prev_payload = payloads[idx - 1]
        curr_payload = payloads[idx]
        prev_bracket = prev_payload["best_bracket"]
        curr_bracket = curr_payload["best_bracket"]
        changes = diff_common_rounds(prev_bracket, curr_bracket)
        ev_gain = float(curr_payload["best_ev"]) - float(prev_payload["best_ev"])

        if changes:
            lines = []
            for round_name, count, samples in changes:
                samples_html = "".join(
                    f"<li>{escape(sample)}</li>" for sample in samples
                )
                lines.append(
                    f"""
                    <div class="change-group">
                      <div class="change-title">{escape(round_name)} <span>{count} changes</span></div>
                      <ul>{samples_html}</ul>
                    </div>
                    """
                )
        else:
            lines = ['<div class="muted">No overlapping-round changes. The new horizon only adds deeper picks.</div>']

        transition_rows.append(
            f"""
            <section class="card change-card">
              <div class="card-top">
                <h3>{escape(ROUND_LABELS[prev_bracket['cutoff_round']])} -> {escape(ROUND_LABELS[curr_bracket['cutoff_round']])}</h3>
                <div class="gain-pill">+{ev_gain:.2f} EV</div>
              </div>
              {''.join(lines)}
            </section>
            """
        )

    final_payload = payloads[-1]
    final_bracket = final_payload["best_bracket"]
    spotlight_items = []
    for region in REGION_ORDER:
        region_data = final_bracket["regions"][region]
        champ = region_data.get("region_champion")
        if champ:
            spotlight_items.append(
                f'<div class="spotlight-row"><span>{escape(region)}</span><strong>({champ["seed"]}) {escape(champ["name"])}</strong></div>'
            )

    if "final_four_winners" in final_bracket:
        ff = ", ".join(
            f"({team['seed']}) {team['name']}" for team in final_bracket["final_four_winners"]
        )
        spotlight_items.append(
            f'<div class="spotlight-row"><span>Finalists</span><strong>{escape(ff)}</strong></div>'
        )
    if "champion" in final_bracket:
        champ = final_bracket["champion"]
        spotlight_items.append(
            f'<div class="spotlight-row"><span>Champion</span><strong>({champ["seed"]}) {escape(champ["name"])}</strong></div>'
        )

    best_gain = max(ev_values) - min(ev_values)
    calculator_bootstrap = build_calculator_bootstrap(final_bracket)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Horizon Bracket Dashboard</title>
  <style>
    :root {{
      --bg: #08111f;
      --bg2: #0f1d33;
      --panel: rgba(12, 23, 42, 0.84);
      --panel-border: rgba(148, 163, 184, 0.14);
      --text: #e5eefc;
      --muted: #93a4bf;
      --accent: #38bdf8;
      --accent2: #f97316;
      --success: #22c55e;
      --shadow: 0 28px 60px rgba(0, 0, 0, 0.32);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 30%),
        radial-gradient(circle at top right, rgba(249, 115, 22, 0.12), transparent 24%),
        linear-gradient(160deg, var(--bg) 0%, var(--bg2) 100%);
      min-height: 100vh;
    }}
    .shell {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 40px 20px 72px;
    }}
    .tabs {{
      display: flex;
      gap: 12px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }}
    .tab-button {{
      appearance: none;
      border: 1px solid rgba(148, 163, 184, 0.14);
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      font-weight: 700;
      color: #c9d8ed;
      background: rgba(15, 23, 42, 0.68);
      cursor: pointer;
      transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
    }}
    .tab-button:hover {{
      transform: translateY(-1px);
      border-color: rgba(56, 189, 248, 0.34);
    }}
    .tab-button.active {{
      color: #f8fbff;
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.24), rgba(249, 115, 22, 0.2));
      border-color: rgba(56, 189, 248, 0.42);
    }}
    .tab-panel[hidden] {{
      display: none;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.5fr 0.9fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}
    .hero-copy {{
      padding: 28px;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--accent);
      font-size: 0.75rem;
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: clamp(2.1rem, 5vw, 4rem);
      line-height: 0.95;
    }}
    p {{
      color: var(--muted);
      line-height: 1.55;
      margin: 0;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 22px;
    }}
    .stat {{
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(15, 23, 42, 0.58);
      border: 1px solid rgba(148, 163, 184, 0.1);
    }}
    .stat .label {{
      color: var(--muted);
      font-size: 0.78rem;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .stat .value {{
      font-size: 1.55rem;
      font-weight: 700;
    }}
    .spotlight {{
      padding: 28px;
      position: relative;
      overflow: hidden;
    }}
    .spotlight:before {{
      content: "";
      position: absolute;
      inset: auto -40px -40px auto;
      width: 160px;
      height: 160px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(249, 115, 22, 0.24), transparent 68%);
    }}
    .spotlight-row {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 0;
      border-bottom: 1px solid rgba(148, 163, 184, 0.08);
      position: relative;
      z-index: 1;
    }}
    .spotlight-row span {{
      color: var(--muted);
    }}
    .section-title {{
      margin: 0 0 12px;
      font-size: 1.2rem;
    }}
    .chart-card {{
      padding: 20px 20px 8px;
      margin-bottom: 24px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
      margin-bottom: 24px;
    }}
    .horizon-card, .change-card {{
      padding: 20px;
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 14px;
    }}
    h3 {{
      margin: 0;
      font-size: 1.25rem;
    }}
    .ev-pill, .gain-pill {{
      white-space: nowrap;
      font-weight: 700;
      padding: 10px 12px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.14);
      color: #bfe8ff;
      border: 1px solid rgba(56, 189, 248, 0.2);
    }}
    .gain-pill {{
      background: rgba(34, 197, 94, 0.14);
      color: #c6f6d5;
      border-color: rgba(34, 197, 94, 0.2);
    }}
    .meta-row {{
      color: var(--muted);
      font-size: 0.92rem;
      margin-bottom: 16px;
    }}
    .chart-note {{
      margin: 0 0 14px;
      color: var(--muted);
      max-width: 760px;
    }}
    .legend-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin: 2px 0 12px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: #c8d7eb;
      font-size: 0.92rem;
    }}
    .legend-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      box-shadow: 0 0 0 3px rgba(255,255,255,0.05);
    }}
    .table-wrap {{
      overflow-x: auto;
      margin-top: 12px;
      border-radius: 18px;
      border: 1px solid rgba(148, 163, 184, 0.12);
      background: rgba(15, 23, 42, 0.42);
    }}
    .curve-table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
    }}
    .curve-table th,
    .curve-table td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid rgba(148, 163, 184, 0.08);
      font-size: 0.94rem;
    }}
    .curve-table thead th {{
      color: #d9e7f7;
      background: rgba(255, 255, 255, 0.03);
      font-weight: 700;
    }}
    .curve-table tbody th {{
      color: #d9e7f7;
      font-weight: 600;
    }}
    .curve-table td {{
      color: var(--muted);
    }}
    .curve-table td.empty {{
      color: rgba(148, 163, 184, 0.55);
    }}
    .table-label {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
    }}
    .matrix-table,
    .heatmap-table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
    }}
    .matrix-table th,
    .matrix-table td,
    .heatmap-table th,
    .heatmap-table td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid rgba(148, 163, 184, 0.08);
      font-size: 0.92rem;
    }}
    .matrix-table thead th,
    .heatmap-table thead th {{
      color: #d9e7f7;
      background: rgba(255, 255, 255, 0.03);
      font-weight: 700;
      position: sticky;
      top: 0;
    }}
    .matrix-table tbody th,
    .heatmap-table tbody th {{
      color: #d9e7f7;
      font-weight: 600;
      background: rgba(10, 18, 34, 0.82);
      position: sticky;
      left: 0;
    }}
    .matrix-table td,
    .heatmap-table td {{
      color: #d6e4f5;
      text-align: center;
    }}
    .team-label {{
      white-space: nowrap;
    }}
    .depth-0 {{ color: rgba(148, 163, 184, 0.62); }}
    .depth-1 {{ background: rgba(148, 163, 184, 0.08); }}
    .depth-2 {{ background: rgba(56, 189, 248, 0.12); }}
    .depth-3 {{ background: rgba(34, 197, 94, 0.12); }}
    .depth-4 {{ background: rgba(250, 204, 21, 0.12); }}
    .depth-5 {{ background: rgba(249, 115, 22, 0.14); }}
    .depth-6 {{ background: rgba(167, 139, 250, 0.18); font-weight: 700; }}
    .pick-block + .pick-block {{
      margin-top: 16px;
    }}
    .pick-label {{
      color: #c5d5ea;
      font-size: 0.9rem;
      margin-bottom: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(148, 163, 184, 0.12);
      font-size: 0.92rem;
    }}
    .seed {{
      display: inline-flex;
      width: 22px;
      height: 22px;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      background: rgba(249, 115, 22, 0.18);
      color: #fed7aa;
      font-size: 0.78rem;
      font-weight: 700;
    }}
    .change-group + .change-group {{
      margin-top: 16px;
    }}
    .change-title {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .change-title span {{
      color: var(--accent);
      font-weight: 600;
    }}
    .calculator-shell {{
      display: grid;
      gap: 20px;
    }}
    .calculator-card {{
      padding: 24px;
    }}
    .calculator-actions {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 18px;
    }}
    .action-button {{
      appearance: none;
      border: 1px solid rgba(148, 163, 184, 0.16);
      border-radius: 14px;
      padding: 11px 14px;
      font: inherit;
      font-weight: 700;
      color: #e7f1fd;
      background: rgba(15, 23, 42, 0.7);
      cursor: pointer;
    }}
    .action-button.primary {{
      background: rgba(56, 189, 248, 0.18);
      border-color: rgba(56, 189, 248, 0.28);
      color: #d7f2ff;
    }}
    .calculator-summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 22px;
    }}
    .summary-stat {{
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(15, 23, 42, 0.58);
      border: 1px solid rgba(148, 163, 184, 0.1);
    }}
    .summary-stat .label {{
      color: var(--muted);
      font-size: 0.78rem;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .summary-stat .value {{
      font-size: 1.35rem;
      font-weight: 700;
    }}
    .round-summary {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .round-stat {{
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(15, 23, 42, 0.42);
      border: 1px solid rgba(148, 163, 184, 0.08);
    }}
    .round-stat .label {{
      color: var(--muted);
      font-size: 0.74rem;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .round-stat .value {{
      font-size: 1rem;
      font-weight: 700;
      color: #dce9f7;
    }}
    .calculator-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .region-card {{
      padding: 20px;
    }}
    .bracket-region {{
      overflow: hidden;
    }}
    .region-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      margin-bottom: 16px;
    }}
    .bracket-board {{
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 14px;
      align-items: start;
      overflow-x: auto;
      padding-bottom: 6px;
    }}
    .bracket-column {{
      min-width: 180px;
    }}
    .bracket-column.round-64 .matchup-stack {{
      padding-top: 0;
    }}
    .bracket-column.round-32 .matchup-stack {{
      padding-top: 28px;
    }}
    .bracket-column.round-16 .matchup-stack {{
      padding-top: 82px;
    }}
    .bracket-column.round-elite-8 .matchup-stack {{
      padding-top: 192px;
    }}
    .matchup-stack {{
      display: grid;
      gap: 12px;
    }}
    .bracket-column.round-32 .matchup-stack,
    .bracket-column.round-16 .matchup-stack,
    .bracket-column.round-elite-8 .matchup-stack {{
      gap: 30px;
    }}
    .bracket-column.round-16 .matchup-stack {{
      gap: 84px;
    }}
    .bracket-column.round-elite-8 .matchup-stack {{
      gap: 196px;
    }}
    .matchup-row {{
      padding: 12px;
      border-radius: 16px;
      background: rgba(15, 23, 42, 0.52);
      border: 1px solid rgba(148, 163, 184, 0.1);
      position: relative;
    }}
    .matchup-row label {{
      display: block;
      font-size: 0.86rem;
      color: #c8d8eb;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .matchup-row select {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(148, 163, 184, 0.16);
      background: rgba(8, 17, 31, 0.9);
      color: #e5eefc;
      font: inherit;
    }}
    .matchup-row.connector:after {{
      content: "";
      position: absolute;
      top: 50%;
      right: -14px;
      width: 14px;
      height: 1px;
      background: rgba(148, 163, 184, 0.22);
    }}
    .finals-grid {{
      display: grid;
      grid-template-columns: 1fr minmax(220px, 280px) 1fr;
      gap: 18px;
      align-items: center;
    }}
    .finals-column {{
      display: grid;
      gap: 18px;
    }}
    .final-card {{
      padding: 20px;
    }}
    .title-card {{
      padding: 24px;
      text-align: center;
      background:
        radial-gradient(circle at top, rgba(56, 189, 248, 0.12), transparent 58%),
        rgba(12, 23, 42, 0.84);
    }}
    .helper-text {{
      color: var(--muted);
      margin-top: 10px;
    }}
    .status-ok {{
      color: #bbf7d0;
    }}
    .status-warn {{
      color: #fde68a;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .muted {{
      color: var(--muted);
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      color: #cde7ff;
    }}
    @media (max-width: 980px) {{
      .hero {{
        grid-template-columns: 1fr;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
      .stats {{
        grid-template-columns: 1fr;
      }}
      .calculator-summary,
      .calculator-grid,
      .finals-grid,
      .round-summary {{
        grid-template-columns: 1fr;
      }}
      .bracket-board {{
        grid-template-columns: repeat(4, minmax(160px, 1fr));
      }}
      .bracket-column.round-32 .matchup-stack,
      .bracket-column.round-16 .matchup-stack,
      .bracket-column.round-elite-8 .matchup-stack {{
        padding-top: 0;
        gap: 12px;
      }}
      .matchup-row.connector:after {{
        display: none;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <div class="tabs" role="tablist" aria-label="Dashboard tabs">
      <button class="tab-button active" type="button" data-tab-target="dashboard-tab" aria-selected="true">Dashboard</button>
      <button class="tab-button" type="button" data-tab-target="calculator-tab" aria-selected="false">Custom EV</button>
    </div>

    <section id="dashboard-tab" class="tab-panel">
      <section class="hero">
        <div class="card hero-copy">
          <div class="eyebrow">Bracket Horizon Dashboard</div>
          <h1>How the optimized bracket evolves as the horizon gets deeper.</h1>
          <p>
            This report compares the best saved bracket for each cutoff round and highlights
            how expected value rises, which teams survive deeper into the tree, and where
            the optimizer changes its mind as it starts caring about later rounds.
          </p>
          <div class="stats">
            <div class="stat">
              <div class="label">Horizons Loaded</div>
              <div class="value">{len(payloads)}</div>
            </div>
            <div class="stat">
              <div class="label">EV Range</div>
              <div class="value">{best_gain:.2f}</div>
            </div>
            <div class="stat">
              <div class="label">Best Horizon</div>
              <div class="value">{escape(ROUND_LABELS[payloads[-1]["best_bracket"]["cutoff_round"]])}</div>
            </div>
          </div>
        </div>
        <aside class="card spotlight">
          <div class="eyebrow">Deepest Horizon Snapshot</div>
          <h3 class="section-title">Round 6 bracket spine</h3>
          {''.join(spotlight_items)}
        </aside>
      </section>

      <section class="card chart-card">
        <div class="eyebrow">Trend</div>
        <h3 class="section-title">Expected Value by Horizon</h3>
        {build_ev_svg(ev_values, ev_labels)}
      </section>

      <section class="card chart-card">
        <div class="eyebrow">Comparison</div>
        <h3 class="section-title">Each Optimized Bracket Against the Others</h3>
        <p class="chart-note">
          Each line shows the cumulative expected value of one horizon-optimized bracket.
          Lines stop at their own cutoff, so the R64 bracket ends after Round 1, the R32 bracket ends after Round 2, and so on.
        </p>
        {build_bracket_curve_svg(curve_rows)}
        {build_curve_table(curve_rows)}
      </section>

      <section>
        <div class="eyebrow">Transitions</div>
        <h3 class="section-title">What Changed from One Horizon to the Next</h3>
        <div class="grid">
          {''.join(transition_rows)}
        </div>
      </section>

      <section class="card chart-card">
        <div class="eyebrow">Teams</div>
        <h3 class="section-title">Team Survival Matrix Across Horizons</h3>
        <p class="chart-note">
          Each cell shows the deepest round that team reaches in that horizon's optimized bracket.
          Teams are sorted by how deep they ever go across the six horizons.
        </p>
        {build_survival_matrix(payloads, seed_lookup)}
      </section>

      <section class="card chart-card">
        <div class="eyebrow">Similarity</div>
        <h3 class="section-title">How Similar the Horizon Brackets Are</h3>
        <p class="chart-note">
          Each percentage compares two horizons across all rounds they both specify.
          Lighter cells mean the optimizer is making almost the same choices in both views.
        </p>
        {build_similarity_heatmap(payloads)}
      </section>

      <section>
        <div class="eyebrow">Profiles</div>
        <h3 class="section-title">Best Bracket at Each Cutoff</h3>
        <div class="grid">
          {''.join(horizon_rows)}
        </div>
      </section>
    </section>

    <section id="calculator-tab" class="tab-panel" hidden>
      <div class="calculator-shell">
        <section class="card calculator-card">
          <div class="eyebrow">Interactive</div>
          <h2 class="section-title">Build Your Own Bracket and Score Its EV</h2>
          <p>
            Enter winners round by round. The calculator uses the same BPI-derived matchup model
            as the dashboard and computes the full expected value for your custom bracket.
          </p>
          <div class="calculator-actions">
            <button id="load-optimized" class="action-button primary" type="button">Load best Title bracket</button>
            <button id="reset-calculator" class="action-button" type="button">Clear picks</button>
          </div>
          <div class="calculator-summary">
            <div class="summary-stat">
              <div class="label">Custom EV</div>
              <div class="value" id="custom-ev">-</div>
            </div>
            <div class="summary-stat">
              <div class="label">Completed Picks</div>
              <div class="value" id="custom-complete">0 / 63</div>
            </div>
            <div class="summary-stat">
              <div class="label">Champion</div>
              <div class="value" id="custom-champion">-</div>
            </div>
            <div class="summary-stat">
              <div class="label">Status</div>
              <div class="value status-warn" id="custom-status">Waiting for picks</div>
            </div>
          </div>
          <div class="round-summary">
            <div class="round-stat">
              <div class="label">R64 EV</div>
              <div class="value" id="round-ev-1">-</div>
            </div>
            <div class="round-stat">
              <div class="label">R32 EV</div>
              <div class="value" id="round-ev-2">-</div>
            </div>
            <div class="round-stat">
              <div class="label">Sweet 16 EV</div>
              <div class="value" id="round-ev-3">-</div>
            </div>
            <div class="round-stat">
              <div class="label">Elite 8 EV</div>
              <div class="value" id="round-ev-4">-</div>
            </div>
            <div class="round-stat">
              <div class="label">Final Four EV</div>
              <div class="value" id="round-ev-5">-</div>
            </div>
            <div class="round-stat">
              <div class="label">Title EV</div>
              <div class="value" id="round-ev-6">-</div>
            </div>
          </div>
          <p class="helper-text">Round choices unlock as upstream winners are selected. EV updates automatically once the bracket is complete.</p>
        </section>

        <div id="calculator-root"></div>
      </div>
    </section>
  </main>
  <script id="calculator-data" type="application/json">{calculator_bootstrap}</script>
  <script>
    const tabButtons = document.querySelectorAll(".tab-button");
    const tabPanels = document.querySelectorAll(".tab-panel");
    tabButtons.forEach((button) => {{
      button.addEventListener("click", () => {{
        const target = button.dataset.tabTarget;
        tabButtons.forEach((candidate) => {{
          const active = candidate === button;
          candidate.classList.toggle("active", active);
          candidate.setAttribute("aria-selected", active ? "true" : "false");
        }});
        tabPanels.forEach((panel) => {{
          panel.hidden = panel.id !== target;
        }});
      }});
    }});

    const calculatorData = JSON.parse(document.getElementById("calculator-data").textContent);
    const regionOrder = calculatorData.regions;
    const teamsByRegion = calculatorData.teamsByRegion;
    const optimized = calculatorData.optimized;
    const scalingFactor = calculatorData.scalingFactor;
    const roundParams = calculatorData.roundParams;
    const roundKeyOrder = ["round_64_winners", "round_32_winners", "round_16_winners", "region_champion"];
    const roundTitleMap = {{
      round_64_winners: "R64",
      round_32_winners: "R32",
      round_16_winners: "Sweet 16",
      region_champion: "Elite 8",
    }};
    const state = {{
      regions: {{}},
      final_four_winners: ["", ""],
      champion: "",
    }};
    const teamLookup = {{}};

    regionOrder.forEach((region) => {{
      state.regions[region] = {{
        round_64_winners: Array(8).fill(""),
        round_32_winners: Array(4).fill(""),
        round_16_winners: Array(2).fill(""),
        region_champion: [""],
      }};
      teamsByRegion[region].forEach((team) => {{
        teamLookup[team.name] = team;
      }});
    }});

    const root = document.getElementById("calculator-root");
    const customEv = document.getElementById("custom-ev");
    const customComplete = document.getElementById("custom-complete");
    const customChampion = document.getElementById("custom-champion");
    const customStatus = document.getElementById("custom-status");
    const roundEvNodes = [
      document.getElementById("round-ev-1"),
      document.getElementById("round-ev-2"),
      document.getElementById("round-ev-3"),
      document.getElementById("round-ev-4"),
      document.getElementById("round-ev-5"),
      document.getElementById("round-ev-6"),
    ];

    function prob(teamAName, teamBName) {{
      if (teamAName === teamBName) {{
        return 0.5;
      }}
      const teamA = teamLookup[teamAName];
      const teamB = teamLookup[teamBName];
      return 1 / (1 + Math.exp(-(teamA.bpi - teamB.bpi) * scalingFactor));
    }}

    function createOptionMarkup(options, selectedValue) {{
      const placeholder = '<option value="">Choose winner</option>';
      const choices = options.map((name) => {{
        const selected = name === selectedValue ? " selected" : "";
        const team = teamLookup[name];
        return `<option value="${{name}}"${{selected}}>(${{team.seed}}) ${{name}}</option>`;
      }});
      return placeholder + choices.join("");
    }}

    function getRegionRoundSources(region, roundKey) {{
      if (roundKey === "round_64_winners") {{
        const teams = teamsByRegion[region].map((team) => team.name);
        const pairs = [];
        for (let i = 0; i < teams.length; i += 2) {{
          pairs.push([teams[i], teams[i + 1]]);
        }}
        return pairs;
      }}
      if (roundKey === "round_32_winners") {{
        const picks = state.regions[region].round_64_winners;
        return [[picks[0], picks[1]], [picks[2], picks[3]], [picks[4], picks[5]], [picks[6], picks[7]]];
      }}
      if (roundKey === "round_16_winners") {{
        const picks = state.regions[region].round_32_winners;
        return [[picks[0], picks[1]], [picks[2], picks[3]]];
      }}
      const picks = state.regions[region].round_16_winners;
      return [[picks[0], picks[1]]];
    }}

    function syncInvalidPick(region, roundKey, slot, validOptions) {{
      if (!validOptions.includes(state.regions[region][roundKey][slot])) {{
        state.regions[region][roundKey][slot] = "";
      }}
    }}

    function syncRegionDependencies(region) {{
      roundKeyOrder.forEach((roundKey) => {{
        const sources = getRegionRoundSources(region, roundKey);
        sources.forEach((options, slot) => {{
          syncInvalidPick(region, roundKey, slot, options.filter(Boolean));
        }});
      }});
    }}

    function syncFinalDependencies() {{
      const eastSouth = [state.regions.East.region_champion[0], state.regions.South.region_champion[0]].filter(Boolean);
      const westMidwest = [state.regions.West.region_champion[0], state.regions.Midwest.region_champion[0]].filter(Boolean);
      if (!eastSouth.includes(state.final_four_winners[0])) {{
        state.final_four_winners[0] = "";
      }}
      if (!westMidwest.includes(state.final_four_winners[1])) {{
        state.final_four_winners[1] = "";
      }}
      const finalOptions = state.final_four_winners.filter(Boolean);
      if (!finalOptions.includes(state.champion)) {{
        state.champion = "";
      }}
    }}

    function regionCardMarkup(region) {{
      const roundClasses = {{
        round_64_winners: "round-64",
        round_32_winners: "round-32",
        round_16_winners: "round-16",
        region_champion: "round-elite-8",
      }};
      const columns = roundKeyOrder.map((roundKey, roundIndex) => {{
        const sources = getRegionRoundSources(region, roundKey);
        const rows = sources.map((options, slot) => {{
          const validOptions = options.filter(Boolean);
          const selectedValue = state.regions[region][roundKey][slot];
          const disabled = validOptions.length < 2 ? " disabled" : "";
          const subtitle = validOptions.length
            ? validOptions.map((name) => `(${{teamLookup[name].seed}}) ${{name}}`).join(" vs ")
            : "Waiting for prior round";
          const connectorClass = roundIndex < roundKeyOrder.length - 1 ? " connector" : "";
          return `
            <div class="matchup-row${{connectorClass}}">
              <label>${{roundTitleMap[roundKey]}} matchup ${{slot + 1}}</label>
              <div class="helper-text">${{subtitle}}</div>
              <select data-region="${{region}}" data-round-key="${{roundKey}}" data-slot="${{slot}}"${{disabled}}>
                ${{createOptionMarkup(validOptions, selectedValue)}}
              </select>
            </div>
          `;
        }}).join("");
        return `
          <div class="bracket-column ${{roundClasses[roundKey]}}">
            <div class="pick-label">${{roundTitleMap[roundKey]}}</div>
            <div class="matchup-stack">${{rows}}</div>
          </div>
        `;
      }}).join("");
      return `
        <section class="card region-card bracket-region">
          <div class="region-header">
            <h3>${{region}}</h3>
            <div class="ev-pill">${{teamsByRegion[region].length}} teams</div>
          </div>
          <div class="bracket-board">${{columns}}</div>
        </section>
      `;
    }}

    function finalsMarkup() {{
      const semi1Options = [state.regions.East.region_champion[0], state.regions.South.region_champion[0]].filter(Boolean);
      const semi2Options = [state.regions.West.region_champion[0], state.regions.Midwest.region_champion[0]].filter(Boolean);
      const finalOptions = state.final_four_winners.filter(Boolean);
      const semis = [
        {{ label: "East vs South", options: semi1Options, slot: 0 }},
        {{ label: "West vs Midwest", options: semi2Options, slot: 1 }},
      ].map((item) => {{
        const disabled = item.options.length < 2 ? " disabled" : "";
        return `
          <section class="card final-card">
            <div class="pick-label">${{item.label}}</div>
            <div class="matchup-row">
              <div class="helper-text">${{item.options.length ? item.options.map((name) => `(${{teamLookup[name].seed}}) ${{name}}`).join(" vs ") : "Waiting for region champions"}}</div>
              <select data-final-slot="${{item.slot}}"${{disabled}}>
                ${{createOptionMarkup(item.options, state.final_four_winners[item.slot])}}
              </select>
            </div>
          </section>
        `;
      }});
      const titleDisabled = finalOptions.length < 2 ? " disabled" : "";
      const titleSubtitle = finalOptions.length
        ? finalOptions.map((name) => `(${{teamLookup[name].seed}}) ${{name}}`).join(" vs ")
        : "Waiting for finalists";
      return `
        <section class="finals-grid">
          <div class="finals-column">${{semis[0]}}</div>
          <section class="card title-card">
            <div class="pick-label">National Champion</div>
            <div class="matchup-row">
              <div class="helper-text">${{titleSubtitle}}</div>
              <select id="champion-select"${{titleDisabled}}>
                ${{createOptionMarkup(finalOptions, state.champion)}}
              </select>
            </div>
            <div class="helper-text">Final Four winners feed the title game automatically.</div>
          </section>
          <div class="finals-column">${{semis[1]}}</div>
        </section>
      `;
    }}

    function renderCalculator() {{
      regionOrder.forEach((region) => syncRegionDependencies(region));
      syncFinalDependencies();
      root.innerHTML = `
        <div class="calculator-grid">
          ${{regionOrder.map((region) => regionCardMarkup(region)).join("")}}
        </div>
        ${{finalsMarkup()}}
      `;
      root.querySelectorAll("select[data-region]").forEach((select) => {{
        select.addEventListener("change", (event) => {{
          const el = event.currentTarget;
          state.regions[el.dataset.region][el.dataset.roundKey][Number(el.dataset.slot)] = el.value;
          renderCalculator();
          updateSummary();
        }});
      }});
      root.querySelectorAll("select[data-final-slot]").forEach((select) => {{
        select.addEventListener("change", (event) => {{
          state.final_four_winners[Number(event.currentTarget.dataset.finalSlot)] = event.currentTarget.value;
          renderCalculator();
          updateSummary();
        }});
      }});
      const championSelect = document.getElementById("champion-select");
      if (championSelect) {{
        championSelect.addEventListener("change", (event) => {{
          state.champion = event.currentTarget.value;
          updateSummary();
        }});
      }}
    }}

    function buildCustomBracket() {{
      const bracket = {{ cutoff_round: 6, regions: {{}}, final_four_winners: [] }};
      for (const region of regionOrder) {{
        const picks = state.regions[region];
        if (picks.round_64_winners.some((v) => !v) || picks.round_32_winners.some((v) => !v) || picks.round_16_winners.some((v) => !v) || !picks.region_champion[0]) {{
          return null;
        }}
        bracket.regions[region] = {{
          teams: teamsByRegion[region],
          round_64_winners: picks.round_64_winners.map((name) => teamLookup[name]),
          round_32_winners: picks.round_32_winners.map((name) => teamLookup[name]),
          round_16_winners: picks.round_16_winners.map((name) => teamLookup[name]),
          region_champion: teamLookup[picks.region_champion[0]],
        }};
      }}
      if (state.final_four_winners.some((v) => !v) || !state.champion) {{
        return null;
      }}
      bracket.final_four_winners = state.final_four_winners.map((name) => teamLookup[name]);
      bracket.champion = teamLookup[state.champion];
      return bracket;
    }}

    function matchupEv(leftDist, rightDist, pickedName, roundNum) {{
      const [base, bonus] = roundParams[String(roundNum)];
      let ev = 0;
      for (const [leftName, leftProb] of Object.entries(leftDist)) {{
        const leftTeam = teamLookup[leftName];
        for (const [rightName, rightProb] of Object.entries(rightDist)) {{
          const rightTeam = teamLookup[rightName];
          const pMatch = leftProb * rightProb;
          if (pickedName === leftName) {{
            ev += pMatch * prob(leftName, rightName) * (base + bonus * Math.max(0, leftTeam.seed - rightTeam.seed));
          }} else if (pickedName === rightName) {{
            ev += pMatch * prob(rightName, leftName) * (base + bonus * Math.max(0, rightTeam.seed - leftTeam.seed));
          }}
        }}
      }}
      return ev;
    }}

    function advanceDistribution(leftDist, rightDist) {{
      const out = {{}};
      for (const [leftName, leftProb] of Object.entries(leftDist)) {{
        for (const [rightName, rightProb] of Object.entries(rightDist)) {{
          const pMatch = leftProb * rightProb;
          out[leftName] = (out[leftName] || 0) + pMatch * prob(leftName, rightName);
          out[rightName] = (out[rightName] || 0) + pMatch * prob(rightName, leftName);
        }}
      }}
      return out;
    }}

    function scoreBracketEv(bracket) {{
      let totalEv = 0;
      const roundTotals = [0, 0, 0, 0, 0, 0];
      const regionChampDists = {{}};
      for (const region of regionOrder) {{
        const regionData = bracket.regions[region];
        const regionTeams = regionData.teams;
        const r64Inputs = [];
        for (let i = 0; i < regionTeams.length; i += 2) {{
          r64Inputs.push([regionTeams[i], regionTeams[i + 1]]);
        }}
        const r64WinnerDists = [];
        regionData.round_64_winners.forEach((picked, slot) => {{
          const [teamA, teamB] = r64Inputs[slot];
          const ev = matchupEv({{ [teamA.name]: 1 }}, {{ [teamB.name]: 1 }}, picked.name, 1);
          totalEv += ev;
          roundTotals[0] += ev;
          r64WinnerDists.push({{ [teamA.name]: prob(teamA.name, teamB.name), [teamB.name]: prob(teamB.name, teamA.name) }});
        }});
        const r32WinnerDists = [];
        regionData.round_32_winners.forEach((picked, slot) => {{
          const leftDist = r64WinnerDists[2 * slot];
          const rightDist = r64WinnerDists[2 * slot + 1];
          const ev = matchupEv(leftDist, rightDist, picked.name, 2);
          totalEv += ev;
          roundTotals[1] += ev;
          r32WinnerDists.push(advanceDistribution(leftDist, rightDist));
        }});
        const r16WinnerDists = [];
        regionData.round_16_winners.forEach((picked, slot) => {{
          const leftDist = r32WinnerDists[2 * slot];
          const rightDist = r32WinnerDists[2 * slot + 1];
          const ev = matchupEv(leftDist, rightDist, picked.name, 3);
          totalEv += ev;
          roundTotals[2] += ev;
          r16WinnerDists.push(advanceDistribution(leftDist, rightDist));
        }});
        const eliteEv = matchupEv(r16WinnerDists[0], r16WinnerDists[1], regionData.region_champion.name, 4);
        totalEv += eliteEv;
        roundTotals[3] += eliteEv;
        regionChampDists[region] = advanceDistribution(r16WinnerDists[0], r16WinnerDists[1]);
      }}
      const ffEv1 = matchupEv(regionChampDists.East, regionChampDists.South, bracket.final_four_winners[0].name, 5);
      totalEv += ffEv1;
      roundTotals[4] += ffEv1;
      const semi1 = advanceDistribution(regionChampDists.East, regionChampDists.South);
      const ffEv2 = matchupEv(regionChampDists.West, regionChampDists.Midwest, bracket.final_four_winners[1].name, 5);
      totalEv += ffEv2;
      roundTotals[4] += ffEv2;
      const semi2 = advanceDistribution(regionChampDists.West, regionChampDists.Midwest);
      const titleEv = matchupEv(semi1, semi2, bracket.champion.name, 6);
      totalEv += titleEv;
      roundTotals[5] += titleEv;
      return {{ total: totalEv, rounds: roundTotals }};
    }}

    function countCompletedPicks() {{
      let complete = 0;
      for (const region of regionOrder) {{
        for (const roundKey of roundKeyOrder) {{
          complete += state.regions[region][roundKey].filter(Boolean).length;
        }}
      }}
      complete += state.final_four_winners.filter(Boolean).length;
      if (state.champion) {{
        complete += 1;
      }}
      return complete;
    }}

    function updateSummary() {{
      const completed = countCompletedPicks();
      customComplete.textContent = `${{completed}} / 63`;
      customChampion.textContent = state.champion ? `(${{teamLookup[state.champion].seed}}) ${{state.champion}}` : "-";
      const bracket = buildCustomBracket();
      if (!bracket) {{
        customEv.textContent = "-";
        customStatus.textContent = completed === 0 ? "Waiting for picks" : "Incomplete bracket";
        customStatus.className = "value status-warn";
        roundEvNodes.forEach((node) => {{
          node.textContent = "-";
        }});
        return;
      }}
      const scored = scoreBracketEv(bracket);
      customEv.textContent = scored.total.toFixed(2);
      scored.rounds.forEach((value, idx) => {{
        roundEvNodes[idx].textContent = value.toFixed(2);
      }});
      customStatus.textContent = "Full bracket scored";
      customStatus.className = "value status-ok";
    }}

    function loadOptimizedBracket() {{
      regionOrder.forEach((region) => {{
        const saved = optimized.regions[region];
        roundKeyOrder.forEach((roundKey) => {{
          state.regions[region][roundKey] = [...saved[roundKey]];
        }});
      }});
      state.final_four_winners = [...optimized.final_four_winners];
      state.champion = optimized.champion[0] || "";
      renderCalculator();
      updateSummary();
    }}

    function resetCalculator() {{
      regionOrder.forEach((region) => {{
        roundKeyOrder.forEach((roundKey) => {{
          state.regions[region][roundKey] = state.regions[region][roundKey].map(() => "");
        }});
      }});
      state.final_four_winners = ["", ""];
      state.champion = "";
      renderCalculator();
      updateSummary();
    }}

    document.getElementById("load-optimized").addEventListener("click", loadOptimizedBracket);
    document.getElementById("reset-calculator").addEventListener("click", resetCalculator);
    renderCalculator();
    updateSummary();
  </script>
</body>
</html>
"""


def main() -> None:
    payloads = load_horizon_payloads()
    html = render_dashboard(payloads)
    DOCS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html)
    DOCS_OUTPUT_PATH.write_text(html)
    print(f"Wrote dashboard to {OUTPUT_PATH}")
    print(f"Wrote publishable site to {DOCS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
