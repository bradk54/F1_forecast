"""Regenerate References/team_lineage.md from the constructor lineage table.

Same arrangement as ``write_data_dictionary``: ``src/data/teams.py`` is the
single source of truth and this renders it, so the reference cannot drift from
the code that the pipeline actually joins on.

Markdown rather than CSV because ``*.csv`` is gitignored here, with two named
exceptions for the model logs.  A lookup table nobody can read in a diff is
not much of a lookup table; ``teams.entries_frame()`` is the joinable form.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import teams  # noqa: E402

OUTPUT = REPO_ROOT / "References" / "team_lineage.md"

#: The window the table guarantees to cover completely.
FIRST_SEASON = 2006
LAST_SEASON = 2026

GRADE_BLURB = {
    teams.REBRAND: "New name, unchanged entity, factory and staff.",
    teams.TAKEOVER: "New owner, continuing operation.",
    teams.RECONSTITUTED: (
        "Continued through an insolvency as a new legal entity. "
        "**The championship restarts the constructor's record here.**"
    ),
    teams.NEW: "No predecessor.",
}


def render() -> str:
    lines: list[str] = []
    w = lines.append

    w("# Formula One constructor lineage, 2006-2026\n")
    w("Which team on today's grid is which team of twenty years ago.\n")
    w("Generated from `src/data/teams.py`. Regenerate with:\n")
    w("```bash\npython -m scripts.write_team_lineage\n```\n")

    entries = teams.entries_frame()
    lineages = teams.lineage_frame()
    born = sorted(
        (
            e for e in teams.ENTRIES
            if e.succession == teams.NEW and e.first_season >= FIRST_SEASON
        ),
        key=lambda e: (e.first_season, e.name),
    )
    renames = teams.renames_between(FIRST_SEASON, LAST_SEASON)

    w("\n## Why this file exists\n")
    w("Constructors are bought and renamed far more often than they are founded.")
    w(
        f"Across the {LAST_SEASON - FIRST_SEASON + 1} seasons from {FIRST_SEASON} "
        f"to {LAST_SEASON} there were **{len(renames)} changes of constructor "
        f"name** but only **{len(born)} genuinely new teams**:"
    )
    w(", ".join(f"{e.name} ({e.first_season})" for e in born) + ".\n")
    w("Every team feature in `src/features/build_features.py` is a rolling history")
    w("keyed on `TeamId` -- `team_dnf_rate_10`, `team_points_rate_career`,")
    w("`team_races_to_date`. A rename resets all of them to zero. The 2021 season")
    w("opens with Aston Martin and Alpine apparently having never entered a race;")
    w("`is_new_pairing` fires for twenty untouched driver-team combinations. The")
    w("reset lands at the start of a season, where the rolling window is most")
    w("valuable, and nothing reports it, because a team with no history looks")
    w("exactly like a team that is genuinely new.\n")

    w("\n## How to use it\n")
    w("```python")
    w("from src.data import teams")
    w("")
    w("teams.resolve(\"sauber\", 2024).name        # 'Kick Sauber'")
    w("teams.lineage_of(\"Force India\", 2016)     # 'aston_martin'")
    w("frame = teams.attach_lineage(frame)       # joins on TeamId + Year")
    w("```\n")
    w("`attach_lineage` adds two grouping keys, and choosing between them is a")
    w("modelling decision this table does not make for you:\n")
    w("| key | groups by | Aston Martin's history starts |")
    w("| --- | --- | --- |")
    w(
        "| `lineage_id` | continuous **operation** -- the factory and the "
        "people | 1991, as Jordan |"
    )
    w(
        "| `(lineage_id, lineage_era)` | continuous **record**, split at every "
        "insolvency | 2018, as Racing Point |"
    )
    w("\nThe second is what the championship itself recognises, and is the")
    w("conservative choice.\n")

    w("\n## Succession grades\n")
    w("| grade | meaning | count |")
    w("| --- | --- | --- |")
    for grade in (teams.REBRAND, teams.TAKEOVER, teams.RECONSTITUTED, teams.NEW):
        n = int((entries["succession"] == grade).sum())
        w(f"| `{grade}` | {GRADE_BLURB[grade]} | {n} |")

    w("\n## Lineages\n")
    w("Named after the constructor currently racing them. `base` is the town the")
    w("operation actually lives in, which is the more durable identity.\n")
    w("| lineage | base | names | active | seasons |")
    w("| --- | --- | --- | --- | --- |")
    for row in lineages.itertuples(index=False):
        last = "present" if row.active else row.last_season
        w(
            f"| `{row.lineage_id}` | {row.lineage_base} | {row.names} | "
            f"{'yes' if row.active else 'no'} | {row.first_season}-{last} |"
        )

    w("\n## Every constructor, by lineage\n")
    for lineage_id, meta in sorted(
        teams.LINEAGES.items(), key=lambda kv: kv[1]["name"]
    ):
        chain = teams._lineage_entries(lineage_id)
        w(f"\n### {meta['name']} — {meta['base']}\n")
        w("| seasons | constructor | key | from | succession | era |")
        w("| --- | --- | --- | --- | --- | --- |")
        for entry in chain:
            prior = teams._predecessor(entry)
            w(
                f"| {entry.span} | {entry.name} | `{entry.key}` | "
                f"{prior.name if prior else '—'} | `{entry.succession}` | "
                f"{teams.era_of(entry)} |"
            )
        w("")
        for entry in chain:
            w(f"- **{entry.name}** ({entry.span}) — {entry.note}")

    w("\n## Name changes, newest first\n")
    w("Each row is a point at which a `TeamId`-keyed rolling history silently")
    w("restarts.\n")
    w("| season | from | to | lineage | succession | record carries |")
    w("| --- | --- | --- | --- | --- | --- |")
    for row in renames.itertuples(index=False):
        w(
            f"| {row.season} | {row.from_name} | {row.to_name} | "
            f"`{row.lineage_id}` | `{row.succession}` | "
            f"{'yes' if row.history_carries else '**no**'} |"
        )

    w("\n## The grid, season by season\n")
    w("| season | n | constructors |")
    w("| --- | --- | --- |")
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        active = teams.active_in(season)
        w(f"| {season} | {len(active)} | {', '.join(e.name for e in active)} |")

    w("\n## Two traps\n")
    w("**\"Lotus\" in 2011 and 2012 are different teams.** In 2011 the grid carried")
    w("both Team Lotus (Hingham, later Caterham) and Lotus Renault GP (Enstone,")
    w("later Alpine); in 2012 the Enstone team took the plain name the Hingham team")
    w("had just given up. `resolve(\"Lotus\")` without a season raises rather than")
    w("guessing.\n")
    w("**An identifier can outlive the name it stood for.** The results backend has")
    w("kept a constructor id across a rebrand, so a 2024 row can arrive tagged")
    w("`sauber`. `resolve` carries such a name *forward* along its lineage, and")
    w("fills a gap inside it (`sauber` in 2008 is BMW Sauber). It never carries a")
    w("name backwards: the 2015 Enstone car was a Lotus, not an Alpine.\n")

    w("\n## Sources\n")
    w("Wikipedia's constructor histories, themselves sourced to the FIA entry lists")
    w("and contemporaneous reporting: *List of Formula One constructors*, *Team")
    w("Enstone*, *Sauber Motorsport*, *Racing Point F1 Team*, *Brawn GP*, *Scuderia")
    w("Toro Rosso*, *Racing Bulls*, *Marussia F1*, *Caterham F1*, *HRT Formula 1")
    w("Team*, *Cadillac in Formula One*, *Audi in Formula One*. Each entry's note")
    w("above records the specific fact and the year it happened.\n")

    return "\n".join(lines) + "\n"


def main() -> int:
    problems = teams.validate()
    if problems:
        print("teams table is inconsistent; not writing:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(), encoding="utf-8")
    print(
        f"wrote {OUTPUT.relative_to(REPO_ROOT)} "
        f"({len(teams.ENTRIES)} constructors, {len(teams.LINEAGES)} lineages)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
