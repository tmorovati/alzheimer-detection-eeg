"""End-to-end orchestration for the packaged notebook workflow."""

from .config import DEFAULT_DATA_DIR, DEFAULT_MAT_PATH, DEFAULT_WORKING_DIR
from .data_export import load_eeg_mat, save_time_series
from .dataset import collect_subject_folders_and_labels, load_and_preprocess_data
from .model import build_compiled_resnet
from .training import run_iterative_training, run_single_training


def prepare_data(mat_path=DEFAULT_MAT_PATH, working_dir=DEFAULT_WORKING_DIR, data_dir=DEFAULT_DATA_DIR, skip_export=False):
    data = None
    if not skip_export:
        data = load_eeg_mat(mat_path)
        epoch_num = data["epoch_num"]
        eeg_data = data["EEG"]
        save_time_series(eeg_data, epoch_num, working_dir=working_dir)

    subject_folders, labels = collect_subject_folders_and_labels(data_dir=data_dir)
    prepared = load_and_preprocess_data(subject_folders, labels)

    return data, prepared


def run_pipeline(
    mode="iterations",
    mat_path=DEFAULT_MAT_PATH,
    working_dir=DEFAULT_WORKING_DIR,
    data_dir=DEFAULT_DATA_DIR,
    skip_export=False,
    iterations=5,
    sleep_seconds=None,
):
    _, prepared = prepare_data(
        mat_path=mat_path,
        working_dir=working_dir,
        data_dir=data_dir,
        skip_export=skip_export,
    )

    (
        train_series,
        train_labels_t,
        val_series,
        val_labels_t,
        test_series,
        test_labels_t,
        test_subjects,
        test_image_counts_per_subject,
    ) = prepared

    model = build_compiled_resnet()

    results = {}

    if mode in {"single", "all"}:
        results["single"] = run_single_training(
            model,
            train_series,
            train_labels_t,
            val_series,
            val_labels_t,
            test_series,
            test_labels_t,
            test_subjects,
            test_image_counts_per_subject,
            working_dir=working_dir,
            sleep_seconds=30 if sleep_seconds is None else sleep_seconds,
        )

    if mode in {"iterations", "all"}:
        results["iterations"] = run_iterative_training(
            model,
            train_series,
            train_labels_t,
            val_series,
            val_labels_t,
            test_series,
            test_labels_t,
            test_subjects,
            test_image_counts_per_subject,
            working_dir=working_dir,
            iterations=iterations,
            sleep_seconds=90 if sleep_seconds is None else sleep_seconds,
        )

    return results
