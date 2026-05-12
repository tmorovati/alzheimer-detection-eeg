"""Dataset folder discovery and loading from saved time-series arrays."""

import os

import numpy as np

from .config import DEFAULT_DATA_DIR


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

    for folder_name in folders:
        subject_num = extract_numeric_part(folder_name)
        subject_folder = os.path.join(data_dir, folder_name)
        subject_folders.append(subject_folder)

        if subject_num <= 36:
            labels.append(0)
        elif 37 <= subject_num <= 65:
            labels.append(1)

    subject_folders = np.array(subject_folders)
    labels = np.array(labels)

    print("Subject Folders:", subject_folders)
    print("Labels:", labels)

    return subject_folders, labels


def load_series_and_labels(subjects, labels):
    series, labels_t = [], []
    for subject_folder, label in zip(subjects, labels):
        for filename in os.listdir(subject_folder):
            series_path = os.path.join(subject_folder, filename.decode())
            try:
                ser = np.load(series_path)

                series.append(ser)
                labels_t.append(label)
            except Exception as e:
                print(f"Error loading time-series: {series_path}, {e}")

    return np.array(series), np.array(labels_t)


def load_and_preprocess_data(subject_folders, labels):
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
