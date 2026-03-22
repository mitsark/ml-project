import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.metrics import accuracy_score, f1_score

from data_pipeline import create_complete_pipeline
from trabsa_model import TRABSA
from train_and_evaluate import (
    BaselineModel,
    calculate_class_weights,
    train_epoch,
    validate,
    generate_predictions_with_confidence,
)


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    csv_path = os.getenv("DATA_CSV", "Combined Data.csv")
    print(f"Using dataset: {csv_path}")

    print("=" * 80)
    print("MINI TRAINING CHECK (Baseline + 1-Epoch TRABSA)")
    print("=" * 80)

    # Keep this small enough for a quick sanity run.
    pipeline = create_complete_pipeline(
        csv_path=csv_path,
        batch_size=8,
        sample_size=512,
        tokenization_mode="streamed",
        max_length=128,
        chunk_size=256,
    )

    train_loader = pipeline["train_loader"]
    val_loader = pipeline["val_loader"]
    test_loader = pipeline["test_loader"]
    df = pipeline["df"]
    class_names = pipeline["class_names"]
    train_idx, val_idx, test_idx = pipeline["indices"]

    print("\n" + "-" * 80)
    print("BASELINE CHECK")
    print("-" * 80)

    baseline = BaselineModel(max_features=2000, ngram_range=(1, 2))
    baseline.train(
        X_train=df.iloc[train_idx]["text_clean"].values,
        y_train=df.iloc[train_idx]["label"].values,
        X_val=df.iloc[val_idx]["text_clean"].values,
        y_val=df.iloc[val_idx]["label"].values,
    )
    baseline_results = baseline.evaluate(
        X_test=df.iloc[test_idx]["text_clean"].values,
        y_test=df.iloc[test_idx]["label"].values,
        class_names=class_names,
    )

    print("\n" + "-" * 80)
    print("TRABSA 1-EPOCH CHECK")
    print("-" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TRABSA(
        num_classes=len(class_names),
        freeze_roberta_layers=11,
        hidden_dim=128,
        dropout=0.3,
        num_lstm_layers=1,
        num_attention_heads=8,
    ).to(device)

    class_weights = calculate_class_weights(df["label"].values, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    test_preds, test_labels, _, _ = generate_predictions_with_confidence(model, test_loader, device)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1_w = f1_score(test_labels, test_preds, average="weighted", zero_division=0)

    print("\n" + "=" * 80)
    print("SANITY SUMMARY")
    print("=" * 80)
    print(f"Baseline test accuracy: {baseline_results['accuracy']:.4f}")
    print(f"Baseline test F1 (weighted): {baseline_results['f1_weighted']:.4f}")
    print(f"TRABSA train loss (1 epoch): {train_loss:.4f}")
    print(f"TRABSA train acc (1 epoch): {train_acc:.4f}")
    print(f"TRABSA val loss (after 1 epoch): {val_loss:.4f}")
    print(f"TRABSA val acc (after 1 epoch): {val_acc:.4f}")
    print(f"TRABSA test acc (after 1 epoch): {test_acc:.4f}")
    print(f"TRABSA test F1 (weighted, after 1 epoch): {test_f1_w:.4f}")


if __name__ == "__main__":
    main()
