# AD Detection ResNet

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
  model.py          ResNet and squeeze-excitation model code
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
ad-detection-resnet --mode iterations
```

Useful options:

```bash
ad-detection-resnet --mode single
ad-detection-resnet --mode all
ad-detection-resnet --skip-export
ad-detection-resnet --mat-path /path/to/EEG_full_4D_1Hz.mat
ad-detection-resnet --working-dir /kaggle/working
ad-detection-resnet --data-dir /kaggle/working/Alzheimer/Time_series/Parietal
```

## Notes

This conversion intentionally does not fix or reinterpret the notebook logic.
Code that may depend on notebook/Kaggle behavior is preserved in the package so
that the packaged version remains faithful to the original workflow.
