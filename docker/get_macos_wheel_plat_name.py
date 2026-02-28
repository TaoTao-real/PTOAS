#!/usr/bin/env python3
"""Return a single-arch macOS wheel platform tag."""

import argparse
import sysconfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_arch", choices=["x86_64", "arm64"])
    args = parser.parse_args()

    plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    if plat.endswith("universal2"):
        plat = plat[: -len("universal2")] + args.target_arch
    print(plat)


if __name__ == "__main__":
    main()
