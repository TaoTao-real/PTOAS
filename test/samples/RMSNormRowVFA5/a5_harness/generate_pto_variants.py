#!/usr/bin/env python3
"""Generate static-row PTO variants for the RMSNorm row-VF A5 matrix."""

import argparse
import re
from pathlib import Path


SUPPORTED_ROWS = (1, 8, 32, 64)
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


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise ValueError("expected one {!r}, found {}".format(old, count))
    return text.replace(old, new)


def render(source, rows, symbol):
    text = source
    text = replace_once(
        text,
        "!bf16_rows = !pto.tile_buf<vec, 64x64xbf16>",
        "!bf16_rows = !pto.tile_buf<vec, {}x64xbf16>".format(rows),
    )
    text = replace_once(
        text,
        "!f32_rows = !pto.tile_buf<vec, 64x64xf32>",
        "!f32_rows = !pto.tile_buf<vec, {}x64xf32>".format(rows),
    )
    text = replace_once(
        text,
        "!f32_row_state = !pto.tile_buf<vec, 64x8xf32, valid=64x1>",
        "!f32_row_state = !pto.tile_buf<vec, {}x8xf32, valid={}x1>".format(
            rows, rows
        ),
    )
    text = replace_once(text, "func.func @rmsnorm_row_vf_n64(", "func.func @{}(".format(symbol))
    text = replace_once(
        text,
        "    %c1 = arith.constant 1 : index\n",
        "    %c1 = arith.constant 1 : index\n"
        "    %c_rows = arith.constant {} : index\n".format(rows),
    )
    text = text.replace("64x64xbf16", "{}x64xbf16".format(rows))
    text = text.replace("64x64xf32", "{}x64xf32".format(rows))
    if text.count("sizes = [%c64, %c64]") != 2:
        raise ValueError("expected x/y partition sizes in canonical source")
    text = text.replace("sizes = [%c64, %c64]", "sizes = [%c_rows, %c64]")
    text = text.replace("[64,64] -> [64,1] -> [64,64]", "[{0},64] -> [{0},1] -> [{0},64]".format(rows))
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-tag", required=True)
    parser.add_argument("--rows", type=int, action="append", choices=SUPPORTED_ROWS)
    args = parser.parse_args()

    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = source_path.read_text()
    rows_set = tuple(args.rows or SUPPORTED_ROWS)
    tag = identifier(args.experiment_tag)

    manifest = [
        "rows\tvariant\tkernel_symbol\tcandidate_policy\tfusion_mode\tstate_mode\tpto"
    ]
    for rows in rows_set:
        for variant, (policy, mode, state_mode) in VARIANTS.items():
            symbol = "rmsnorm_row_vf_n{}_{}_{}".format(rows, variant.lower(), tag)
            path = output_dir / "n{}-{}.pto".format(rows, variant)
            path.write_text(render(source, rows, symbol))
            manifest.append(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                    rows, variant, symbol, policy, mode, state_mode, path.name
                )
            )
    (output_dir / "variants.tsv").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
