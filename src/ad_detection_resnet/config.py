"""Configuration values copied from the original notebook."""

from pathlib import Path

LOBES = {
    "Frontal": [1, 2, 3, 4, 11, 12, 17],
    "Temporal": [13, 14, 15, 16],
    "Parietal": [7, 8, 19],
    "Central": [5, 6, 18],
    "Occipital": [9, 10],
}

DEFAULT_MAT_PATH = Path("/kaggle/input/eeg-dataset/EEG_full_4D_1Hz.mat")
DEFAULT_WORKING_DIR = Path("/kaggle/working")
DEFAULT_TIME_SERIES_DIR = DEFAULT_WORKING_DIR / "Alzheimer" / "Time_series"
DEFAULT_DATA_DIR = DEFAULT_TIME_SERIES_DIR / "Parietal"

DEFAULT_INPUT_SHAPE = (2500, 1)
DEFAULT_NUM_CLASSES = 1
DEFAULT_EEG_KEY = "EEG"
DEFAULT_EPOCH_KEY = "epoch_num"
