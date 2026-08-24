#!/usr/bin/env python3
"""Generate Base32/Base64/Base128 PTO variants for the A5 matrix."""

import argparse
import re
from pathlib import Path


WIDTHS = (32, 64, 128)
VARIANTS = {
    "A": ("ordinary", "off"),
    "B": ("prefer-vmi", "off"),
    "C": ("prefer-vmi", "loop"),
    "D": ("prefer-vmi", "full"),
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


def render(source, width, symbol):
    text = replace_once(
        source,
        "!f32_tile = !pto.tile_buf<vec, 16x64xf32>",
        "!f32_tile = !pto.tile_buf<vec, 16x{}xf32>".format(width),
    )
    text = replace_once(
        text,
        "!f32_col = !pto.tile_buf<vec, 1x64xf32>",
        "!f32_col = !pto.tile_buf<vec, 1x{}xf32>".format(width),
    )
    text = replace_once(
        text,
        "func.func @softmax_dnd_vf_b4_m16_n64(",
        "func.func @{}(".format(symbol),
    )
    text = replace_once(
        text,
        "%c_inner = arith.constant 64 : index",
        "%c_inner = arith.constant {} : index".format(width),
    )
    text = text.replace("64x64xf32", "64x{}xf32".format(width))
    text = text.replace("16x64xf32", "16x{}xf32".format(width))
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-tag", required=True)
    parser.add_argument("--width", type=int, action="append", choices=WIDTHS)
    args = parser.parse_args()

    source = Path(args.source).read_text()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = identifier(args.experiment_tag)
    widths = tuple(args.width or WIDTHS)
    manifest = [
        "width\tvariant\tkernel_symbol\tcandidate_policy\tfusion_mode\tpto"
    ]
    for width in widths:
        for variant, (policy, mode) in VARIANTS.items():
            symbol = "softmax_dnd_b4_m16_n{}_{}_{}".format(
                width, variant.lower(), tag
            )
            path = output_dir / "n{}-{}.pto".format(width, variant)
            path.write_text(render(source, width, symbol))
            manifest.append(
                "{}\t{}\t{}\t{}\t{}\t{}".format(
                    width, variant, symbol, policy, mode, path.name
                )
            )
    (output_dir / "variants.tsv").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
