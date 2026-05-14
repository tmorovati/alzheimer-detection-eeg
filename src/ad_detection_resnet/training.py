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
from .dataset import extract_numeric_part


def _image_level_metrics(labels, predictions):
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def _subject_level_metrics(model, series, subjects, image_counts_per_subject):
    print("\nPredictions for Each Subject's Images:")
    start_idx = 0
    predicted_subjects = []
    groundtruth_subjects = []
    subject_probabilities = []

    for subject_folder, num_images in zip(subjects, image_counts_per_subject):
        end_idx = start_idx + num_images
        subject_images = series[start_idx:end_idx]
        subject_probabilities_t = model.predict(subject_images, verbose=0).reshape(-1)
        subject_predictions = (subject_probabilities_t > 0.5).astype(int)
        ad_count = int(np.sum(subject_predictions == 0))
        hc_count = int(np.sum(subject_predictions == 1))
        mean_hc_probability = float(np.mean(subject_probabilities_t))
        subject_num = extract_numeric_part(os.path.basename(subject_folder))

        print(f"Subject {subject_num} ({subject_folder}):")
        print(f"AD count: {ad_count}, HC count: {hc_count}, mean HC probability: {mean_hc_probability:.4f}")

        groundtruth_subjects.append(0 if subject_num < 37 else 1)
        predicted_subjects.append(1 if mean_hc_probability >= 0.5 else 0)
        subject_probabilities.append(mean_hc_probability)
        start_idx = end_idx

    metrics = _image_level_metrics(groundtruth_subjects, predicted_subjects)
    return metrics, groundtruth_subjects, predicted_subjects, subject_probabilities


def _plot_history(history, title_suffix):
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["accuracy"], label="Training Accuracy", marker="o")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy", marker="o")
    plt.title(f"Training and Validation Accuracy {title_suffix}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["loss"], label="Training Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Validation Loss", marker="o")
    plt.title(f"Training and Validation Loss {title_suffix}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def _print_metric_summary(prefix, metric_lists):
    for metric_name, values in metric_lists.items():
        print(f"\nmean for {prefix}, {metric_name} {np.mean(values)} and std is {np.std(values)}")


def _safe_model_name(model_name):
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in str(model_name).lower()).strip("_")
    return safe_name or "model"


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
    model_name="model",
):
    fold_no = 1
    scores = []
    keras = tf.keras
    hist = []
    model = model
    del keras

    checkpoint_name = _safe_model_name(model_name)
    filepath = str(Path(working_dir) / f"best_{checkpoint_name}_single_fold_{fold_no}.keras")
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


def run_cross_validation(
    model_builder,
    fold_data,
    working_dir=DEFAULT_WORKING_DIR,
    sleep_seconds=90,
    model_name="model",
):
    scores = []
    hist = []

    image_metric_lists = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1-score": [],
    }
    subject_metric_lists = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1-score": [],
    }
    fold_results = []

    for fold_no, prepared_fold in fold_data:
        (
            train_series,
            train_labels_t,
            val_series,
            val_labels_t,
            test_series,
            test_labels_t,
            test_subjects,
            test_image_counts_per_subject,
        ) = prepared_fold

        tf.keras.backend.clear_session()
        model = model_builder()

        checkpoint_name = _safe_model_name(model_name)
        filepath = str(Path(working_dir) / f"best_{checkpoint_name}_fold_{fold_no}.keras")
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

        model = tf.keras.models.load_model(filepath)
        score = model.evaluate(test_series, test_labels_t, verbose=1)
        scores.append(score[1])

        test_predictions = (model.predict(test_series) > 0.5).astype(int)
        image_metrics = _image_level_metrics(test_labels_t, test_predictions)
        print(
            f"Fold {fold_no} held-out image metrics: Accuracy {image_metrics['accuracy']:.4f}, "
            f"Precision {image_metrics['precision']:.4f}, Recall {image_metrics['recall']:.4f}, "
            f"F1 {image_metrics['f1']:.4f}"
        )

        image_metric_lists["accuracy"].append(image_metrics["accuracy"])
        image_metric_lists["precision"].append(image_metrics["precision"])
        image_metric_lists["recall"].append(image_metrics["recall"])
        image_metric_lists["f1-score"].append(image_metrics["f1"])

        subject_metrics, groundtruth_subjects, predicted_subjects, subject_probabilities = _subject_level_metrics(
            model,
            test_series,
            test_subjects,
            test_image_counts_per_subject,
        )
        print(
            f"Fold {fold_no} held-out subject metrics: Accuracy {subject_metrics['accuracy']:.4f}, "
            f"Precision {subject_metrics['precision']:.4f}, Recall {subject_metrics['recall']:.4f}, "
            f"F1 {subject_metrics['f1']:.4f}"
        )

        subject_metric_lists["accuracy"].append(subject_metrics["accuracy"])
        subject_metric_lists["precision"].append(subject_metrics["precision"])
        subject_metric_lists["recall"].append(subject_metrics["recall"])
        subject_metric_lists["f1-score"].append(subject_metrics["f1"])

        fold_results.append(
            {
                "fold": fold_no,
                "score": score,
                "image_metrics": image_metrics,
                "subject_metrics": subject_metrics,
                "groundtruth_subjects": groundtruth_subjects,
                "predicted_subjects": predicted_subjects,
                "subject_probabilities": subject_probabilities,
            }
        )

        _plot_history(hist_temp.history, f"for fold {fold_no}")

        if sleep_seconds:
            time.sleep(sleep_seconds)

    _print_metric_summary("image", image_metric_lists)
    _print_metric_summary("subject", subject_metric_lists)

    return {
        "scores": scores,
        "history": hist,
        "fold_results": fold_results,
        "image_metrics": image_metric_lists,
        "subject_metrics": subject_metric_lists,
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
    model_name="model",
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
        checkpoint_name = _safe_model_name(model_name)
        filepath = str(Path(working_dir) / f"best_{checkpoint_name}_iteration_{iteration}.keras")
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
