"""python -m orderflow screenshot.png [--compact] > out.json"""
import sys

from .assemble import to_json


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1:
        print("usage: python -m orderflow SCREENSHOT.png [--compact]", file=sys.stderr)
        return 2
    print(to_json(args[0], pretty="--compact" not in argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
