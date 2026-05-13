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
ad-detection-resnet --mat-path /path/to/EEG_full_4D_1Hz.mat
ad-detection-resnet --working-dir /kaggle/working
ad-detection-resnet --data-dir /kaggle/working/Alzheimer/Time_series/Parietal
```

## Validation

The default `crossval` mode uses the mentor-provided 5-fold subject-level
stratified split. Each fold trains on 52 subjects and validates on 13 held-out
subjects, with a fresh model built for every fold. The original notebook split
is still available through `single` and `iterations`: subjects 33-41 are used
for validation, subjects 27-32 and 42-48 are used for test, and the remaining
subjects are used for training.

## Model Choice

Use `--model eegnet` or `--model inceptiontime` to switch the architecture for
any training mode. EEGNet is the default.

## Notes

The original notebook-style training paths are preserved for comparison, while
the package default now follows the mentor-provided cross-validation split.
