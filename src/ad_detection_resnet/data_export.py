"""Load the EEG `.mat` file and save the notebook time-series arrays."""

import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .config import DEFAULT_EEG_KEY, DEFAULT_EPOCH_KEY, DEFAULT_MAT_PATH, DEFAULT_WORKING_DIR, LOBES


def load_eeg_mat(mat_path=DEFAULT_MAT_PATH):
    """Load the EEG MATLAB file with the same behavior as the notebook."""
    data = loadmat(mat_path)
    print("done")
    return data


def get_mat_value(data, key, value_name):
    """Read a named MATLAB variable and show available variables if missing."""
    try:
        return data[key]
    except KeyError as exc:
        available_keys = sorted(k for k in data if not k.startswith("__"))
        raise KeyError(
            f"Could not find {value_name} key {key!r} in the .mat file. "
            f"Available keys are: {available_keys}"
        ) from exc


def get_eeg_data(data, eeg_key=DEFAULT_EEG_KEY):
    return get_mat_value(data, eeg_key, "EEG data")


def get_epoch_num(data, epoch_key=DEFAULT_EPOCH_KEY):
    return get_mat_value(data, epoch_key, "epoch number")


def iter_subject_arrays(eeg_data):
    """Yield one subject array from either a 4D array or MATLAB cell/object array."""
    if isinstance(eeg_data, np.ndarray) and eeg_data.dtype == object:
        for subject_data in eeg_data.ravel():
            yield np.asarray(subject_data)
        return

    for subject_idx in range(eeg_data.shape[0]):
        yield eeg_data[subject_idx, :, :, :]


def save_time_series(eeg_data, epoch_num, working_dir=DEFAULT_WORKING_DIR, lobes=LOBES):
    """Save all subject/channel/segment arrays into the notebook folder layout."""
    working_dir = Path(working_dir)

    for subject_idx, subject_data in enumerate(iter_subject_arrays(eeg_data)):
        if subject_data.ndim != 3:
            raise ValueError(
                "Each subject EEG array must have 3 dimensions: "
                f"time x channels x segments. Got shape {subject_data.shape} "
                f"for subject {subject_idx + 1}."
            )

        save_dir = os.path.join(working_dir, "Alzheimer", "Time_series")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        segment_count = min(10, subject_data.shape[2])
        if segment_count < 10:
            print(f"Subject {subject_idx + 1} has only {segment_count} segments; saving available segments.")

        for channel_idx in range(subject_data.shape[1]):
            channel_data = subject_data[:, channel_idx, :]

            for seg_idx in range(segment_count):
                segment_data = channel_data[:, seg_idx]

                save_path = os.path.join(
                    save_dir,
                    "All lobes",
                    f"subject{subject_idx + 1}",
                )
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path2 = os.path.join(
                    save_path,
                    f"subject{subject_idx + 1}_channel{channel_idx + 1}_segment{seg_idx + 1}",
                )
                np.save(save_path2, segment_data)

                for lobe_name, lobe_channels in lobes.items():
                    if channel_idx + 1 in lobe_channels:
                        save_dir2 = os.path.join(
                            working_dir,
                            "Alzheimer",
                            "Time_series",
                            lobe_name,
                            f"subject{subject_idx + 1}",
                        )
                        if not os.path.exists(save_dir2):
                            os.makedirs(save_dir2)
                        save_path = os.path.join(
                            save_dir2,
                            f"subject{subject_idx + 1}_channel{channel_idx + 1}_segment{seg_idx + 1}",
                        )

                        np.save(save_path, segment_data)

    print("saved successfully.")
