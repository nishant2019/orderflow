"""python -m orderflow screenshot.png [--layout auto|single|split] [--analyze] [--compact] > out.json"""
import json
import sys

from .analytics import analyze
from .assemble import parse_screenshot


def main(argv: list[str]) -> int:
    layout = "auto"
    if "--layout" in argv:
        i = argv.index("--layout")
        layout = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1:
        print("usage: python -m orderflow SCREENSHOT.png [--layout auto|single|split] [--analyze] [--compact]", file=sys.stderr)
        return 2
    doc = parse_screenshot(args[0], layout=layout)
    result = {"parsed": doc, "analysis": analyze(doc)} if "--analyze" in argv else doc
    print(json.dumps(result, indent=None if "--compact" in argv else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
