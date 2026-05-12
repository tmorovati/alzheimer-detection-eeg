"""Load the EEG `.mat` file and save the notebook time-series arrays."""

import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .config import DEFAULT_MAT_PATH, DEFAULT_WORKING_DIR, LOBES


def load_eeg_mat(mat_path=DEFAULT_MAT_PATH):
    """Load the EEG MATLAB file with the same behavior as the notebook."""
    data = loadmat(mat_path)
    print("done")
    return data


def save_time_series(eeg_data, epoch_num, working_dir=DEFAULT_WORKING_DIR, lobes=LOBES):
    """Save all subject/channel/segment arrays into the notebook folder layout."""
    working_dir = Path(working_dir)

    for subject_idx in range(eeg_data.shape[0]):
        subject_data = eeg_data[subject_idx, :, :, :]

        save_dir = os.path.join(working_dir, "Alzheimer", "Time_series")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for channel_idx in range(subject_data.shape[1]):
            channel_data = subject_data[:, channel_idx, :]

            for seg_idx in range(10):
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
