#!/usr/bin/env python3
"""Generate experiment-scoped [8,4096] RMSNorm PTO variants."""

import argparse
import re
from pathlib import Path


VARIANTS = {
    "A": ("ordinary", "off", "off"),
    "B": ("prefer-vmi", "off", "off"),
    "C": ("prefer-vmi", "loop", "off"),
    "D": ("prefer-vmi", "full", "generic"),
    "DL": ("prefer-vmi", "full", "legacy"),
}


def identifier(text):
    value = re.sub(r"[^a-zA-Z0-9_]", "_", text)
    if not value or value[0].isdigit():
        value = "e_" + value
    return value.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-tag", required=True)
    args = parser.parse_args()
    source = Path(args.source).read_text()
    old_symbol = "rms_norm_1vl_ascendc_aligned"
    if source.count(old_symbol) != 1:
        raise ValueError(f"expected one {old_symbol}, found {source.count(old_symbol)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = identifier(args.experiment_tag)
    manifest = [
        "variant\tkernel_symbol\tcandidate_policy\tfusion_mode\t"
        "state_promotion_mode\tpto"
    ]
    for variant, (policy, fusion_mode, state_mode) in VARIANTS.items():
        symbol = f"rms_norm_1vl_{variant.lower()}_{tag}"
        path = output_dir / f"{variant}.pto"
        path.write_text(source.replace(old_symbol, symbol))
        manifest.append(
            f"{variant}\t{symbol}\t{policy}\t{fusion_mode}\t{state_mode}\t{path.name}"
        )
    (output_dir / "variants.tsv").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
