#!/usr/bin/env python3

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ir", type=Path)
    args = parser.parse_args()

    text = args.ir.read_text()
    scope_markers = ("pto.vecscope {", "pto.strict_vecscope(")
    scope_count = 0
    cursor = 0
    while cursor < len(text):
        positions = [text.find(marker, cursor) for marker in scope_markers]
        positions = [position for position in positions if position >= 0]
        if not positions:
            break
        start = min(positions)
        body_start = text.find("{", start)
        if body_start < 0:
            raise SystemExit("vecscope without a body")

        depth = 0
        body_end = -1
        for index in range(body_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    body_end = index
                    break
        if body_end < 0:
            raise SystemExit("unterminated vecscope body")

        body = text[body_start : body_end + 1]
        if "pto.set_flag" in body or "pto.wait_flag" in body:
            raise SystemExit("pipe synchronization found inside vecscope")
        scope_count += 1
        cursor = body_end + 1

    if scope_count == 0:
        raise SystemExit("no vecscope found")
    print(f"vecscope_sync_check=PASS scopes={scope_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
