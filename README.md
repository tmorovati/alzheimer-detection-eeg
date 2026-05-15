# AD Detection EEG

This repository contains a packaged version of `ad-detection-ResNet.ipynb`.
The original notebook is left untouched. The Python package keeps the same
execution flow and default paths, while splitting the notebook into readable
modules.

## Package Layout

```text
src/ad_detection_resnet/
  __main__.py       Command-line entry point
  config.py         Constants copied from the notebook
  data_export.py    `.mat` loading and time-series saving
  dataset.py        Subject folder, label, and split loading
  model.py          EEGNet and InceptionTime model code
  pipeline.py       End-to-end orchestration
  training.py       Single-run and iterative training loops
```

## Install

From the project root:

```bash
python -m pip install -e .
```

The package declares the same major libraries used by the notebook, including
TensorFlow, Keras, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, and Pillow.

## Run

The default paths match the Kaggle notebook:

```bash
ad-detection-resnet --mode crossval --model eegnet
```

Useful options:

```bash
ad-detection-resnet --mode crossval --model inceptiontime
ad-detection-resnet --mode single
ad-detection-resnet --mode all
ad-detection-resnet --mode iterations
ad-detection-resnet --skip-export
ad-detection-resnet --validation-protocol paper
ad-detection-resnet --validation-protocol strict
ad-detection-resnet --mat-path /path/to/EEG_full_4D_1Hz.mat
ad-detection-resnet --eeg-key EEG --epoch-key epoch_num
ad-detection-resnet --eeg-key EEG_Class --epoch-key epoch_num
ad-detection-resnet --working-dir /kaggle/working
ad-detection-resnet --data-dir /kaggle/working/Alzheimer/Time_series/Parietal
```

If your `.mat` file uses different variable names, inspect them first:

```python
from scipy.io import loadmat

data = loadmat("/path/to/EEG_full_4D_1Hz.mat")
[(key, getattr(value, "shape", None)) for key, value in data.items() if not key.startswith("__")]
```

Then pass the matching names with `--eeg-key` and `--epoch-key`.
The exporter supports both a direct 4D EEG array and a MATLAB cell/object array
such as `EEG_Class`, where each subject is stored as its own time x channels x
segments array.

## Validation

The default `crossval` mode uses the co-author-aligned subject-level stratified
5-fold protocol. In each fold, 52 subjects are used for training and 13 subjects
are used as the validation/evaluation fold. This matches the fold lists used by
the other approaches in the paper.

An optional stricter protocol is also available:

```bash
ad-detection-resnet --validation-protocol strict
```

In strict mode, one fold is held out for final testing and the next fold is used
only for internal validation, early stopping, learning-rate scheduling, and
checkpoint selection.

The AD-vs-HC setup uses subjects 1-65:

```text
subjects 1-36   AD
subjects 37-65  HC
subjects 66-88  excluded as unlabeled/FTD for this binary task
```

The exporter uses all available non-overlapping 10-second epochs for each
subject. For direct zero-padded 4D arrays, `epoch_num` is used to avoid padded
segments. For `EEG_Class` MATLAB cell arrays, each subject array is exported up
to its available segment count.

The original notebook split is still available through `single` and
`iterations`: subjects 33-41 are used for validation, subjects 27-32 and 42-48
are used for test, and the remaining subjects are used for training.

## Lobe Folders

The lobe channel groups follow the paper's named regions:

```text
Frontal:   F7, F3, Fz, F4, F8
Temporal:  T3, T4, T5, T6
Parietal:  P3, Pz, P4
Central:   C3, Cz, C4
Occipital: O1, O2
```

To compare lobes, export once and then rerun with `--skip-export` while changing
only `--data-dir`, using the same `--mode` and `--model` settings for every
lobe.

## Model Choice

Use `--model eegnet` or `--model inceptiontime` to switch the architecture for
any training mode. EEGNet is the default.

## Notes

The original notebook-style training paths are preserved for comparison, while
the package default now follows the mentor-provided cross-validation split.
