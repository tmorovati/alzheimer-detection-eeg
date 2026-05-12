"""Training and evaluation loops copied from the notebook."""

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .config import DEFAULT_WORKING_DIR


def run_single_training(
    model,
    train_series,
    train_labels_t,
    val_series,
    val_labels_t,
    test_series,
    test_labels_t,
    test_subjects,
    test_image_counts_per_subject,
    working_dir=DEFAULT_WORKING_DIR,
    sleep_seconds=30,
):
    fold_no = 1
    scores = []
    keras = tf.keras
    hist = []
    model = model
    del keras

    filepath = str(Path(working_dir) / f"best_model_folds_{fold_no}.keras")
    if os.path.exists(filepath):
        print("File exists")
    else:
        print("File does not exist")

    checkpoint = ModelCheckpoint(
        filepath,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
        initial_value_threshold=None,
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.1,
        patience=6,
        min_lr=1e-6,
    )

    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        verbose=1,
        mode="max",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=25,
    )

    hist_temp = model.fit(
        train_series,
        train_labels_t,
        validation_data=(val_series, val_labels_t),
        epochs=70,
        batch_size=8,
        callbacks=[checkpoint, lr_scheduler, early_stopping],
        verbose=1,
    )

    hist.append(hist_temp)

    model.load_weights(filepath)
    score = model.evaluate(test_series, test_labels_t, verbose=1)
    scores.append(score[1])
    print(
        f"\nScore for fold {fold_no}: {model.metrics_names[1]} of {score[1]}; "
        f"{model.metrics_names[0]} of {score[0]}"
    )

    test_predictions = model.predict(test_series)
    test_predictions = (test_predictions > 0.5).astype(int)

    accuracy = accuracy_score(test_labels_t, test_predictions)
    precision = precision_score(test_labels_t, test_predictions)
    recall = recall_score(test_labels_t, test_predictions)
    f1 = f1_score(test_labels_t, test_predictions)

    print("\nPredictions for Each Subject's Images:")
    start_idx = 0
    for i, (subject_folder, num_images) in enumerate(zip(test_subjects, test_image_counts_per_subject)):
        end_idx = start_idx + num_images
        subject_images = test_series[start_idx:end_idx]
        subject_predictions = (model.predict(subject_images) > 0.5).astype(int)
        ad_count = sum(subject_predictions == 0)
        hc_count = sum(subject_predictions == 1)
        print(f"Subject {i + 1} ({subject_folder}):")
        print(f"AD count: {ad_count}, HC count: {hc_count}")
        start_idx = end_idx

    history = hist_temp.history
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["accuracy"], label="Training Accuracy", marker="o")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy", marker="o")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["loss"], label="Training Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Validation Loss", marker="o")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

    time.sleep(sleep_seconds)

    return {
        "scores": scores,
        "history": hist,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_iterative_training(
    model,
    train_series,
    train_labels_t,
    val_series,
    val_labels_t,
    test_series,
    test_labels_t,
    test_subjects,
    test_image_counts_per_subject,
    working_dir=DEFAULT_WORKING_DIR,
    iterations=5,
    sleep_seconds=90,
):
    scores = []
    keras = tf.keras
    hist = []
    model = model
    del keras

    acc_epoch_list = []
    precision_epoch_list = []
    f1_epoch_list = []
    recall_epoch_list = []
    acc_sbj_list = []
    precision_sbj_list = []
    f1_sbj_list = []
    recall_sbj_list = []

    for iteration in range(iterations):
        filepath = str(Path(working_dir) / f"best_model_folds_{iteration}.keras")
        if os.path.exists(filepath):
            print("File exists")
        else:
            print("File does not exist")

        checkpoint = ModelCheckpoint(
            filepath,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max",
            initial_value_threshold=None,
        )

        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.1,
            patience=6,
            min_lr=1e-6,
        )

        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            verbose=1,
            mode="max",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=20,
        )

        hist_temp = model.fit(
            train_series,
            train_labels_t,
            validation_data=(val_series, val_labels_t),
            epochs=60,
            batch_size=8,
            callbacks=[checkpoint, lr_scheduler, early_stopping],
            verbose=1,
        )

        hist.append(hist_temp)

        model.load_weights(filepath)
        score = model.evaluate(test_series, test_labels_t, verbose=1)
        scores.append(score[1])

        test_predictions = model.predict(test_series)
        test_predictions = (test_predictions > 0.5).astype(int)

        accuracy = accuracy_score(test_labels_t, test_predictions)
        precision = precision_score(test_labels_t, test_predictions)
        recall = recall_score(test_labels_t, test_predictions)
        f1 = f1_score(test_labels_t, test_predictions)
        print(
            f"Accuracy for epoch: {accuracy:.4f}, Precision for epoch: {precision:.4f}, "
            f"Recall for epoch: {recall:.4f}, F1 for epoch: {f1:.4f}"
        )

        acc_epoch_list.append(accuracy)
        precision_epoch_list.append(precision)
        recall_epoch_list.append(recall)
        f1_epoch_list.append(f1)

        print("\nPredictions for Each Subject's Images:")
        start_idx = 0
        predicted_subjects_list = []
        groundtruth_subjects_list = []
        for i, (subject_folder, num_images) in enumerate(zip(test_subjects, test_image_counts_per_subject)):
            end_idx = start_idx + num_images
            subject_images = test_series[start_idx:end_idx]
            subject_predictions = (model.predict(subject_images) > 0.5).astype(int)
            ad_count = sum(subject_predictions == 0)
            hc_count = sum(subject_predictions == 1)
            print(f"Subject {i + 1} ({subject_folder}):")
            print(f"AD count: {ad_count}, HC count: {hc_count}")
            start_idx = end_idx
            num = subject_folder[int(subject_folder.find("subject")) + 7 :]
            if int(num) < 37:
                groundtruth_subjects_list.append(0)
            else:
                groundtruth_subjects_list.append(1)

            start_idx = end_idx

            if ad_count > hc_count:
                predicted_subjects_list.append(0)
            else:
                predicted_subjects_list.append(1)

        accuracy_sbj = accuracy_score(groundtruth_subjects_list, predicted_subjects_list)
        precision_sbj = precision_score(groundtruth_subjects_list, predicted_subjects_list)
        recall_sbj = recall_score(groundtruth_subjects_list, predicted_subjects_list)
        f1_sbj = f1_score(groundtruth_subjects_list, predicted_subjects_list)

        print(
            f"Accuracy for subject: {accuracy_sbj:.4f}, Precision for subject: {precision_sbj:.4f}, "
            f"Recall for subject: {recall_sbj:.4f}, F1 for subject: {f1_sbj:.4f}"
        )

        history = hist_temp.history
        epochs = range(1, len(history["loss"]) + 1)

        acc_sbj_list.append(accuracy_sbj)
        precision_sbj_list.append(precision_sbj)
        recall_sbj_list.append(recall_sbj)
        f1_sbj_list.append(f1_sbj)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["accuracy"], label="Training Accuracy", marker="o")
        plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy", marker="o")
        plt.title(f"Training and Validation Accuracy for iteration:{iteration}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history["loss"], label="Training Loss", marker="o")
        plt.plot(epochs, history["val_loss"], label="Validation Loss", marker="o")
        plt.title(f"Training and Validation Loss for iteration:{iteration}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.show()

        time.sleep(sleep_seconds)

    print(f"\nmean for epoch, accuracy {np.mean(acc_epoch_list)} and std is {np.std(acc_epoch_list)}")
    print(f"\nmean for epoch, precision {np.mean(precision_epoch_list)} and std is {np.std(precision_epoch_list)}")
    print(f"\nmean for epoch, recall {np.mean(recall_epoch_list)} and std is {np.std(recall_epoch_list)}")
    print(f"\nmean for epoch, f1-score {np.mean(f1_epoch_list)} and std is {np.std(f1_epoch_list)}")

    print(f"\nmean for subject, accuracy {np.mean(acc_sbj_list)} and std is {np.std(acc_sbj_list)}")
    print(f"\nmean for subject, precision {np.mean(precision_sbj_list)} and std is {np.std(precision_sbj_list)}")
    print(f"\nmean for subject, recall {np.mean(recall_sbj_list)} and std is {np.std(recall_sbj_list)}")
    print(f"\nmean for subject, f1-score {np.mean(f1_sbj_list)} and std is {np.std(f1_sbj_list)}")

    return {
        "scores": scores,
        "history": hist,
        "acc_epoch_list": acc_epoch_list,
        "precision_epoch_list": precision_epoch_list,
        "recall_epoch_list": recall_epoch_list,
        "f1_epoch_list": f1_epoch_list,
        "acc_sbj_list": acc_sbj_list,
        "precision_sbj_list": precision_sbj_list,
        "recall_sbj_list": recall_sbj_list,
        "f1_sbj_list": f1_sbj_list,
    }
