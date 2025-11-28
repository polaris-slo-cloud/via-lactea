#!/usr/bin/env python3
import sys
import argparse
import pandas as pd


def read_df(path: str, no_header: bool) -> pd.DataFrame:
    """
    Read CSV with columns: run,strategy,stitch_id (or sid).
    If there is no header row, use --no-header.
    """
    if path == "-":
        data = sys.stdin
    else:
        data = path

    if no_header:
        # assume order: run,strategy,stitch_id
        df = pd.read_csv(data, header=None, names=["run", "strategy", "stitch_id"])
    else:
        df = pd.read_csv(data)

    # normalise column name to 'sid'
    if "sid" in df.columns:
        df = df.rename(columns={"sid": "sid"})
    elif "stitch_id" in df.columns:
        df = df.rename(columns={"stitch_id": "sid"})
    else:
        sys.exit(
            f"Input must have either 'sid' or 'stitch_id' column, got {list(df.columns)}"
        )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Count occurrences of (strategy, sid/stitch_id) pairs."
    )
    parser.add_argument(
        "input",
        help="Input CSV file or '-' for stdin"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV file (default: stdout)"
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Use this if input has no header line"
    )
    args = parser.parse_args()

    df = read_df(args.input, args.no_header)

    expected_cols = {"run", "strategy", "sid"}
    if not expected_cols.issubset(df.columns):
        sys.exit(f"Input must have columns {expected_cols}, got {list(df.columns)}")

    counts = (
        df.groupby(["strategy", "sid"])
          .size()
          .reset_index(name="count")
          .sort_values(["strategy", "sid"])
    )

    if args.output:
        counts.to_csv(args.output, index=False)
    else:
        counts.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
