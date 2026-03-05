import argparse
from app.viewer import run_app


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="", help="Optional STL path. If omitted, file dialog opens.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_app(args.inp)