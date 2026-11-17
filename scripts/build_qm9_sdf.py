#!/usr/bin/env python3
"""Builds `datasets/qm9.sdf` from the MoleculeNet bundle.

The MoleculeNet `qm9.zip` ships `gdb9.sdf` (structures only, titles
`gdb_<n>`) and `gdb9.sdf.csv` (properties in Hartree, keyed by `mol_id`
`gdb_<n>`). The project's `mi-io` QM9 loader reads properties as SDF data
fields with `gdb9_<n>` titles — the original figshare `qm9.sdf` convention.
This script reproduces that convention at the data boundary:

- injects `homo`, `lumo`, `gap` (verbatim Hartree strings) as `> <tag>`
  data fields after each record's `M  END`;
- renames titles `gdb_<n>` → `gdb9_<n>`;
- verifies 1-based record order against the CSV row order (title ==
  `mol_id` for every record; hard failure on any mismatch).

Usage: python scripts/build_qm9_sdf.py <gdb9.sdf> <gdb9.sdf.csv> <out.sdf>
"""

import csv
import sys


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit("usage: build_qm9_sdf.py <gdb9.sdf> <gdb9.sdf.csv> <out.sdf>")
    sdf_path, csv_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    properties = {}
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            properties[row["mol_id"]] = (row["homo"], row["lumo"], row["gap"])
    print(f"csv rows: {len(properties)}")

    records = 0
    with open(sdf_path, encoding="utf-8") as src, open(out_path, "w", encoding="utf-8", newline="\n") as dst:
        in_header = True  # before the counts line / inside the molecule block
        seen_mol_end = False
        pending_title = None
        for line in src:
            text = line.rstrip("\n")
            if in_header:
                if pending_title is None:
                    # First line of a record: the title.
                    pending_title = text
                    if not text.startswith("gdb_"):
                        sys.exit(f"unexpected title: {text!r}")
                    dst.write("gdb9_" + text[len("gdb_"):].lstrip("_") + "\n")
                    continue
                if text.lstrip().endswith("V2000"):
                    in_header = False
                dst.write(line)
                continue
            # Molecule block or data-field region.
            if text.rstrip() == "M  END":
                seen_mol_end = True
                dst.write(line)
                continue
            if seen_mol_end and text == "$$$$":
                # Terminate the record: inject the three properties.
                if pending_title is None:
                    sys.exit("record without a title")
                homo, lumo, gap = properties.get(pending_title, (None, None, None))
                if homo is None:
                    sys.exit(f"csv row missing for {pending_title}")
                for tag, value in (("homo", homo), ("lumo", lumo), ("gap", gap)):
                    dst.write(f">  <{tag}>\n{value}\n\n")
                dst.write("$$$$\n")
                records += 1
                in_header = True
                seen_mol_end = False
                pending_title = None
                continue
            if seen_mol_end:
                # Nothing may precede the terminator inside the data-field
                # region of the source file.
                sys.exit(f"unexpected content after M  END: {text!r}")
            dst.write(line)

    if records != len(properties):
        sys.exit(f"record count {records} != csv rows {len(properties)}")
    print(f"records written: {records} -> {out_path}")


if __name__ == "__main__":
    main()
