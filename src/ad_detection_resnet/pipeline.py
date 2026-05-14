"""End-to-end orchestration for the packaged notebook workflow."""

from .config import DEFAULT_DATA_DIR, DEFAULT_EEG_KEY, DEFAULT_EPOCH_KEY, DEFAULT_MAT_PATH, DEFAULT_WORKING_DIR
from .data_export import get_eeg_data, get_epoch_num, load_eeg_mat, save_time_series
from .dataset import collect_subject_folders_and_labels, iter_cross_validation_folds, load_and_preprocess_data
from .model import build_compiled_model
from .training import run_cross_validation, run_iterative_training, run_single_training


def prepare_subjects(
    mat_path=DEFAULT_MAT_PATH,
    working_dir=DEFAULT_WORKING_DIR,
    data_dir=DEFAULT_DATA_DIR,
    skip_export=False,
    eeg_key=DEFAULT_EEG_KEY,
    epoch_key=DEFAULT_EPOCH_KEY,
):
    data = None
    if not skip_export:
        data = load_eeg_mat(mat_path)
        epoch_num = get_epoch_num(data, epoch_key=epoch_key)
        eeg_data = get_eeg_data(data, eeg_key=eeg_key)
        save_time_series(eeg_data, epoch_num, working_dir=working_dir)

    subject_folders, labels = collect_subject_folders_and_labels(data_dir=data_dir)
    return data, subject_folders, labels


def prepare_data(
    mat_path=DEFAULT_MAT_PATH,
    working_dir=DEFAULT_WORKING_DIR,
    data_dir=DEFAULT_DATA_DIR,
    skip_export=False,
    eeg_key=DEFAULT_EEG_KEY,
    epoch_key=DEFAULT_EPOCH_KEY,
):
    data, subject_folders, labels = prepare_subjects(
        mat_path=mat_path,
        working_dir=working_dir,
        data_dir=data_dir,
        skip_export=skip_export,
        eeg_key=eeg_key,
        epoch_key=epoch_key,
    )
    prepared = load_and_preprocess_data(subject_folders, labels)

    return data, prepared


def run_pipeline(
    mode="crossval",
    mat_path=DEFAULT_MAT_PATH,
    working_dir=DEFAULT_WORKING_DIR,
    data_dir=DEFAULT_DATA_DIR,
    skip_export=False,
    iterations=5,
    sleep_seconds=None,
    model_name="eegnet",
    eeg_key=DEFAULT_EEG_KEY,
    epoch_key=DEFAULT_EPOCH_KEY,
):
    _, subject_folders, labels = prepare_subjects(
        mat_path=mat_path,
        working_dir=working_dir,
        data_dir=data_dir,
        skip_export=skip_export,
        eeg_key=eeg_key,
        epoch_key=epoch_key,
    )

    results = {}

    if mode in {"single", "iterations", "all"}:
        prepared = load_and_preprocess_data(subject_folders, labels)

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

        model = build_compiled_model(model_name=model_name)

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
            model_name=model_name,
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
            model_name=model_name,
        )

    if mode in {"crossval", "all"}:
        def model_builder():
            return build_compiled_model(model_name=model_name)

        results["crossval"] = run_cross_validation(
            model_builder,
            iter_cross_validation_folds(subject_folders, labels),
            working_dir=working_dir,
            sleep_seconds=90 if sleep_seconds is None else sleep_seconds,
            model_name=model_name,
        )

    return results
