"""Command-line entry point for the packaged notebook workflow."""

import argparse

from .config import DEFAULT_DATA_DIR, DEFAULT_MAT_PATH, DEFAULT_WORKING_DIR


def build_parser():
    parser = argparse.ArgumentParser(description="Run the packaged AD Detection ResNet notebook workflow.")
    parser.add_argument(
        "--mode",
        choices=["single", "iterations", "all"],
        default="iterations",
        help="Training section to run.",
    )
    parser.add_argument("--mat-path", default=str(DEFAULT_MAT_PATH), help="Path to EEG_full_4D_1Hz.mat.")
    parser.add_argument("--working-dir", default=str(DEFAULT_WORKING_DIR), help="Directory used for generated outputs.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing lobe subject folders.")
    parser.add_argument("--skip-export", action="store_true", help="Use already-exported time-series files.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterative training runs.")
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=None,
        help="Override the notebook sleep interval after each training run.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from .pipeline import run_pipeline

    run_pipeline(
        mode=args.mode,
        mat_path=args.mat_path,
        working_dir=args.working_dir,
        data_dir=args.data_dir,
        skip_export=args.skip_export,
        iterations=args.iterations,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()
