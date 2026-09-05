"""Regenerate References/data_dictionary.md from the feature registry.

The registry is the single source of truth for what a feature is and when it
becomes knowable; this script renders it so the documentation cannot drift from
the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import registry  # noqa: E402
from src.features.labels import CAUSE_ORDER  # noqa: E402

OUTPUT = REPO_ROOT / "References" / "data_dictionary.md"


def render() -> str:
    lines: list[str] = []
    w = lines.append

    w("# Data dictionary: retirement (DNF) prediction dataset\n")
    w("Generated from `src/features/registry.py`. Regenerate with:\n")
    w("```bash\npython -m scripts.write_data_dictionary\n```\n")

    w("\n## The target\n")
    w("| column | definition |")
    w("| --- | --- |")
    w("| `dnf` | **Primary target.** 1 when the car stopped before the end of the race. |")
    w("| `dnf_strict` | 1 when the car retired *and* was left unclassified. |")
    w("| `dnf_classified` | 1 when the car retired but had covered enough distance to keep a position. |")
    w("| `dnf_cause` | " + " / ".join(f"`{c}`" for c in CAUSE_ORDER) + " |")
    w("| `finished_on_track` | 1 when the car took the chequered flag under its own power. |")
    w("| `started` | 1 when the driver took the start. Rows with 0 are excluded from the dataset. |")
    w("| `classified` | 1 when `ClassifiedPosition` holds an integer. |")

    w("\nTwo decisions worth knowing about:\n")
    w("1. **A driver who retires past 90% distance is still a DNF.** They keep an official")
    w("   position, so `ClassifiedPosition` alone would score it a finish and systematically")
    w("   under-count retirements. `dnf_classified` flags exactly these rows.")
    w("2. **A disqualification after finishing is not a DNF.** A DSQ is a scrutineering")
    w("   outcome applied to a car that usually completed the race. `dnf_cause` still records")
    w("   it, so the rows stay findable.\n")

    w("\n## Feature stages\n")
    w("Each feature is tagged with when it becomes knowable. A model may use its own stage")
    w("and every earlier one.\n")
    w("| stage | available | use for |")
    w("| --- | --- | --- |")
    w("| `pre_weekend` | Monday before the race | season simulation, early forecasting |")
    w("| `post_quali` | Saturday evening | the strongest honest forecast |")
    w("| `race_day` | **after the race** | retrospective analysis only — never a forecast |\n")
    w("> Observed weather is `race_day`. Including it in a model and reporting the result as")
    w("> forecast accuracy would be wrong; it is kept so you can measure how much of")
    w("> retirement risk weather explains.\n")

    frame = registry.registry_frame()
    for stage in registry.STAGE_ORDER:
        block = frame.loc[frame["stage"] == stage]
        w(f"\n## `{stage}` features ({len(block)})\n")
        w("| feature | kind | description |")
        w("| --- | --- | --- |")
        for _, row in block.iterrows():
            w(f"| `{row['feature']}` | {row['kind']} | {row['description']} |")

    w("\n## Leakage guarantee\n")
    w("Every history feature is built by shifting within its entity before aggregating, so")
    w("no feature can see the race it describes. This is enforced, not asserted:")
    w("`src.features.build_features.detect_target_leakage` flips one event's outcomes,")
    w("rebuilds the whole feature table, and reports any feature that moved at or before")
    w("that event. The dataset build runs it and refuses to write if anything is found.\n")
    w("The detector is itself tested against deliberately planted leaks — a `cumsum` with no")
    w("shift, and a same-race team aggregate — so a silent pass means something.\n")

    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
