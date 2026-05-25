#!/usr/bin/env python3
"""Validate that README.md's Record History table is consistent with the
underlying submission result.json files.

Run from the repo root:

    python3 scripts/validate_record_history.py

Exit code is 0 if the table is consistent, 1 if any check fails.

Checks performed:

1. The Record History table parses cleanly (7 columns, all rows
   well-formed).
2. No rows exist outside the table (no orphan submission rows after
   footnotes — a regression caused by the prior ``append_record``
   placing rows past the end of the file).
3. For each submission row whose dir link points to ``submissions/<name>/``,
   the linked ``result.json`` exists and its energy / accuracy match the
   row to within reasonable tolerance.
4. No submission appears multiple times as PASS on the same date.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
README = HERE / "README.md"
SUBMISSIONS = HERE / "submissions"

ENERGY_TOL_REL = 0.01
ACC_TOL = 1e-4


def main() -> int:
    text = README.read_text()
    failures: list[str] = []

    table_rows, orphan_rows = _extract_table(text)

    if not table_rows:
        failures.append("Could not find the Record History table.")
        return _report(failures)

    if orphan_rows:
        failures.append(
            f"Found {len(orphan_rows)} orphan submission row(s) outside "
            f"the Record History table:"
        )
        for line_no, row in orphan_rows:
            failures.append(f"  line {line_no}: {row.strip()}")

    # Each submission slot may have multiple rows (one per re-run on a
    # setup change). result.json is overwritten and only reflects the
    # most recent run — so only the latest row per slot should match
    # result.json. Earlier rows are historical and skipped.
    parsed_rows: list[tuple[int, tuple[str, str, str, str, str, str, str]]] = []
    pass_by_config: dict[str, list[tuple[int, str, str]]] = {}
    for line_no, row in table_rows:
        parsed = _parse_row(row)
        if parsed is None:
            failures.append(f"line {line_no}: row failed to parse: {row.strip()}")
            continue
        parsed_rows.append((line_no, parsed))
        date, energy_cell, acc_cell, gpu, config, dir_link, contributor = parsed
        if acc_cell != "DQ":
            pass_by_config.setdefault(config, []).append((line_no, date, acc_cell))

    # Group by slot dir, validate only the last (highest line_no) row
    # against the slot's current result.json.
    latest_by_slot: dict[str, tuple[int, tuple]] = {}
    for line_no, parsed in parsed_rows:
        _, _, _, _, _, dir_link, _ = parsed
        m = re.match(r"\[dir\]\(submissions/([^)]+)\)", dir_link)
        if not m:
            continue
        slot = m.group(1).rstrip("/")
        latest_by_slot[slot] = (line_no, parsed)

    for slot, (line_no, parsed) in latest_by_slot.items():
        date, energy_cell, acc_cell, gpu, config, dir_link, contributor = parsed
        result_path = SUBMISSIONS / slot / "result.json"
        if not result_path.exists():
            failures.append(
                f"line {line_no}: {slot}: result.json missing at {result_path}"
            )
            continue
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError as exc:
            failures.append(f"line {line_no}: {slot}: result.json unreadable ({exc})")
            continue
        _check_row_against_result(
            line_no, slot, energy_cell, acc_cell, result, failures
        )

    for config, rows in pass_by_config.items():
        dates = [r[1] for r in rows]
        if len(set(dates)) < len(dates):
            same_date = sorted(rows, key=lambda r: r[0])
            failures.append(f"{config}: multiple PASS rows on the same date:")
            for line_no, date, acc in same_date:
                failures.append(f"  line {line_no}: {date} acc={acc}")

    return _report(failures)


def _extract_table(text: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return (table_rows, orphan_rows).

    table_rows: data rows inside the Record History markdown table.
    orphan_rows: lines starting with ``|`` AFTER the table block closed,
    i.e. submission-looking rows that landed past the table separator
    (typically appended after footnotes by a buggy append_record).
    """
    lines = text.splitlines()
    in_record_history = False
    in_table = False
    table_rows: list[tuple[int, str]] = []
    orphans: list[tuple[int, str]] = []
    past_table = False

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("## Record History"):
            in_record_history = True
            continue
        if in_record_history and not in_table:
            if line.startswith("|") and "Energy" in line and "Val" in line:
                in_table = True
                continue
        if in_table:
            if line.startswith("|---") or line.startswith("|--"):
                continue
            if line.startswith("|"):
                table_rows.append((i, line))
                continue
            in_table = False
            past_table = True
            continue
        if past_table:
            if line.startswith("|") and "[dir](submissions/" in line:
                orphans.append((i, line))

    return table_rows, orphans


def _parse_row(row: str) -> tuple[str, str, str, str, str, str, str] | None:
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    if len(cells) < 6:
        return None
    if len(cells) == 6:
        date, energy, acc, config, dir_link, contributor = cells
        gpu = ""
    elif len(cells) >= 7:
        date, energy, acc, gpu, config, dir_link, contributor = cells[:7]
    else:
        return None
    return date, energy, acc, gpu, config, dir_link, contributor


def _check_row_against_result(
    line_no: int,
    slot: str,
    energy_cell: str,
    acc_cell: str,
    result: dict,
    failures: list,
) -> None:
    is_dq = (
        result.get("disqualified", False)
        or result.get("val_char_accuracy") is None
        or result.get("val_char_accuracy", 0.0) < 0.70
    )

    if acc_cell == "DQ":
        if not is_dq:
            failures.append(
                f"line {line_no}: {slot}: row says DQ but result.json is PASS"
            )
    else:
        if is_dq:
            failures.append(
                f"line {line_no}: {slot}: row claims PASS but result.json is DQ"
            )
        else:
            try:
                row_acc = float(acc_cell)
            except ValueError:
                failures.append(
                    f"line {line_no}: {slot}: cannot parse acc cell {acc_cell!r}"
                )
                return
            result_acc = result.get("val_char_accuracy", 0.0)
            if abs(row_acc - result_acc) > ACC_TOL:
                failures.append(
                    f"line {line_no}: {slot}: acc row={row_acc:.4f} vs "
                    f"result.json={result_acc:.4f}"
                )

    try:
        row_energy = float(energy_cell.replace(",", "").strip())
    except ValueError:
        failures.append(
            f"line {line_no}: {slot}: cannot parse energy cell {energy_cell!r}"
        )
        return
    expected_energy = result.get("total_energy_J")
    if expected_energy is None:
        expected_energy = result.get("training_energy_J", 0.0)
    if expected_energy is None or expected_energy == 0:
        return
    rel = abs(row_energy - expected_energy) / max(1.0, expected_energy)
    if rel > ENERGY_TOL_REL:
        failures.append(
            f"line {line_no}: {slot}: energy row={row_energy:,.0f} vs "
            f"result.json={expected_energy:,.0f} (rel diff {rel:.2%})"
        )


def _report(failures: list[str]) -> int:
    if not failures:
        print("README Record History: OK")
        return 0
    print(f"README Record History: {len(failures)} issue(s) found:")
    for f in failures:
        print(f"  {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
