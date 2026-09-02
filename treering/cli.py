"""Command-line interface for the tree-ring detrending pipeline.

Usage
-----
::

    python -m treering input.rwl output.csv [--overwrite] [--skip-failed]

Arguments
---------
input_rwl : str
    Path to a Tucson-format ``.rwl`` file.
output_csv : str
    Path for the output CSV file.

Options
-------
--overwrite
    Overwrite the output CSV if it already exists.
--skip-failed
    Skip series that fail curve fitting instead of aborting.
--verbose / -v
    Enable debug-level logging.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from treering.export import export_csv, ExportError
from treering.model import FittingError
from treering.parser import RWLParseError
from treering.pipeline import process_rwl, PipelineError
from treering.rwi import RWIError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="treering",
        description=(
            "Tree-ring width detrending pipeline.\n\n"
            "Reads a Tucson-format .rwl file, performs biological detrending\n"
            "using a negative exponential growth model, calculates the\n"
            "Ring Width Index (RWI), and exports results to CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_rwl",
        type=str,
        help="Path to input .rwl file (Tucson format)",
    )
    parser.add_argument(
        "output_csv",
        type=str,
        help="Path for output CSV file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output file",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        default=False,
        help="Skip series that fail curve fitting instead of aborting",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug-level logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    args_list = sys.argv[1:] if argv is None else list(argv)
    if args_list and args_list[0] == "spei":
        from treering.spei import main as spei_main
        return spei_main(args_list[1:])
    if args_list and args_list[0] == "forecast":
        from treering.forecast import main as forecast_main
        return forecast_main()
    if args_list and args_list[0] == "holdout":
        from treering.holdout import main as holdout_main
        return holdout_main()

    parser = _build_parser()
    args = parser.parse_args(args_list)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        # Process
        result_df = process_rwl(
            args.input_rwl,
            skip_failed_series=args.skip_failed,
        )

        # Export
        output_path = export_csv(
            result_df,
            args.output_csv,
            overwrite=args.overwrite,
        )

        print(f"Output written to {output_path}")
        return 0

    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except RWLParseError as exc:
        print(f"Parse error: {exc}", file=sys.stderr)
        return 1
    except FittingError as exc:
        print(f"Fitting error: {exc}", file=sys.stderr)
        return 1
    except RWIError as exc:
        print(f"RWI error: {exc}", file=sys.stderr)
        return 1
    except PipelineError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1
    except ExportError as exc:
        print(f"Export error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
