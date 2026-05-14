"""Dataset folder discovery and loading from saved time-series arrays."""

import os

import numpy as np

from .config import DEFAULT_DATA_DIR

MENTOR_FOLD_VAL_SUBJECTS = {
    1: {
        "ad": [13, 14, 18, 20, 22, 24, 32, 34],
        "hc": [39, 49, 53, 59, 60],
    },
    2: {
        "ad": [2, 8, 9, 11, 19, 28, 35],
        "hc": [37, 38, 41, 48, 56, 57],
    },
    3: {
        "ad": [5, 7, 10, 16, 17, 30, 33],
        "hc": [43, 44, 47, 50, 63, 65],
    },
    4: {
        "ad": [3, 21, 26, 27, 29, 31, 36],
        "hc": [42, 54, 55, 61, 62, 64],
    },
    5: {
        "ad": [1, 4, 6, 12, 15, 23, 25],
        "hc": [40, 45, 46, 51, 52, 58],
    },
}


def extract_numeric_part(folder_name):
    parts = folder_name.split("_")
    for part in parts:
        if part.startswith("subject"):
            return int(part.replace("subject", ""))
    return -1


def collect_subject_folders_and_labels(data_dir=DEFAULT_DATA_DIR):
    folders = sorted(
        [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))],
        key=extract_numeric_part,
    )

    subject_folders, labels = [], []
    skipped_subjects = []

    for folder_name in folders:
        subject_num = extract_numeric_part(folder_name)
        subject_folder = os.path.join(data_dir, folder_name)

        if subject_num <= 36:
            subject_folders.append(subject_folder)
            labels.append(0)
        elif 37 <= subject_num <= 65:
            subject_folders.append(subject_folder)
            labels.append(1)
        else:
            skipped_subjects.append(subject_num)

    subject_folders = np.array(subject_folders)
    labels = np.array(labels)

    print("Subject Folders:", subject_folders)
    print("Labels:", labels)
    if skipped_subjects:
        print("Skipped unlabeled subjects:", skipped_subjects)

    return subject_folders, labels


def load_series_and_labels(subjects, labels):
    series, labels_t = [], []
    for subject_folder, label in zip(subjects, labels):
        for filename in sorted(os.listdir(subject_folder)):
            series_path = os.path.join(subject_folder, os.fsdecode(filename))
            try:
                ser = np.load(series_path)

                series.append(ser)
                labels_t.append(label)
            except Exception as e:
                print(f"Error loading time-series: {series_path}, {e}")

    return np.array(series), np.array(labels_t)


def _subjects_by_number(subject_folders, labels):
    subjects = {}
    for subject_folder, label in zip(subject_folders, labels):
        subject_num = extract_numeric_part(os.path.basename(subject_folder))
        subjects[subject_num] = (subject_folder, label)
    return subjects


def _select_subjects(subjects, subject_numbers):
    missing = sorted(set(subject_numbers) - set(subjects))
    if missing:
        raise ValueError(f"Subject folders are missing for subjects: {missing}")

    selected_subjects = [subjects[subject_num][0] for subject_num in subject_numbers]
    selected_labels = [subjects[subject_num][1] for subject_num in subject_numbers]
    return selected_subjects, selected_labels


def _image_counts_per_subject(subjects):
    return [len(os.listdir(folder)) for folder in subjects]


def _fold_subject_numbers(fold_no):
    fold = MENTOR_FOLD_VAL_SUBJECTS[fold_no]
    return sorted(fold["ad"] + fold["hc"])


def load_subject_split(subject_folders, labels, train_subject_nums, val_subject_nums, test_subject_nums=None):
    subjects = _subjects_by_number(subject_folders, labels)

    train_subjects, train_labels = _select_subjects(subjects, train_subject_nums)
    val_subjects, val_labels = _select_subjects(subjects, val_subject_nums)

    train_images, train_labels_t = load_series_and_labels(train_subjects, train_labels)
    val_images, val_labels_t = load_series_and_labels(val_subjects, val_labels)
    val_image_counts_per_subject = _image_counts_per_subject(val_subjects)

    if test_subject_nums is not None:
        test_subjects, test_labels = _select_subjects(subjects, test_subject_nums)
        test_images, test_labels_t = load_series_and_labels(test_subjects, test_labels)
        test_image_counts_per_subject = _image_counts_per_subject(test_subjects)
        return (
            train_images,
            train_labels_t,
            val_images,
            val_labels_t,
            test_images,
            test_labels_t,
            test_subjects,
            test_image_counts_per_subject,
        )

    return (
        train_images,
        train_labels_t,
        val_images,
        val_labels_t,
        val_subjects,
        val_image_counts_per_subject,
    )


def load_cross_validation_fold(subject_folders, labels, fold_no):
    """Load one leakage-resistant subject-level cross-validation split."""
    if fold_no not in MENTOR_FOLD_VAL_SUBJECTS:
        raise ValueError(f"Unknown fold {fold_no}. Expected one of {sorted(MENTOR_FOLD_VAL_SUBJECTS)}.")

    subjects = _subjects_by_number(subject_folders, labels)
    all_subject_nums = set(subjects)
    validation_fold_no = (fold_no % len(MENTOR_FOLD_VAL_SUBJECTS)) + 1
    test_subject_nums = _fold_subject_numbers(fold_no)
    val_subject_nums = _fold_subject_numbers(validation_fold_no)
    train_subject_nums = sorted(all_subject_nums - set(test_subject_nums) - set(val_subject_nums))

    print(f"\nFold {fold_no}:")
    print(f"  Train subjects:      {len(train_subject_nums)}")
    print(f"  Internal val fold:   {validation_fold_no}")
    print(f"  Internal val subjects: {len(val_subject_nums)}")
    print(f"  Held-out test subjects: {len(test_subject_nums)}")
    print(f"  TEST - AD: {MENTOR_FOLD_VAL_SUBJECTS[fold_no]['ad']}")
    print(f"  TEST - HC: {MENTOR_FOLD_VAL_SUBJECTS[fold_no]['hc']}")

    prepared = load_subject_split(subject_folders, labels, train_subject_nums, val_subject_nums, test_subject_nums)
    print(f"  Train images: {len(prepared[0])}")
    print(f"  Internal val images: {len(prepared[2])}")
    print(f"  Held-out test images: {len(prepared[4])}")

    return prepared


def iter_cross_validation_folds(subject_folders, labels, fold_numbers=None):
    fold_numbers = sorted(MENTOR_FOLD_VAL_SUBJECTS) if fold_numbers is None else fold_numbers
    for fold_no in fold_numbers:
        yield fold_no, load_cross_validation_fold(subject_folders, labels, fold_no)


def load_and_preprocess_data(subject_folders, labels):
    """Load the original fixed holdout split preserved from the notebook."""
    train_subjects_dict = {}
    val_subjects_dict = {}
    test_subjects_dict = {}

    for subject_folder, label in zip(subject_folders, labels):
        subject_num = extract_numeric_part(os.path.basename(subject_folder))
        if 33 <= subject_num <= 41:
            val_subjects_dict[subject_folder] = label
        elif 27 <= subject_num <= 32 or 42 <= subject_num <= 48:
            test_subjects_dict[subject_folder] = label
        else:
            train_subjects_dict[subject_folder] = label

    train_subjects = list(train_subjects_dict.keys())
    train_labels = [train_subjects_dict[subj] for subj in train_subjects]
    val_subjects = list(val_subjects_dict.keys())
    val_labels = [val_subjects_dict[subj] for subj in val_subjects]
    test_subjects = list(test_subjects_dict.keys())
    test_labels = [test_subjects_dict[subj] for subj in test_subjects]

    train_images, train_labels_t = load_series_and_labels(train_subjects, train_labels)
    val_images, val_labels_t = load_series_and_labels(val_subjects, val_labels)
    test_images, test_labels_t = load_series_and_labels(test_subjects, test_labels)
    test_image_counts_per_subject = [len(os.listdir(folder)) for folder in test_subjects]

    return (
        train_images,
        train_labels_t,
        val_images,
        val_labels_t,
        test_images,
        test_labels_t,
        test_subjects,
        test_image_counts_per_subject,
    )
