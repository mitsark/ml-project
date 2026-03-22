import time
import numpy as np
import torch
import torch.nn as nn

from data_pipeline import create_complete_pipeline
from trabsa_model import TRABSA
from train_and_evaluate import (
    calculate_class_weights,
    train_epoch,
    validate,
)


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 80)
    print("TRABSA 3-EPOCH TIMING CHECK")
    print("=" * 80)

    pipeline = create_complete_pipeline(
        csv_path="Combined Data.csv",
        batch_size=8,
        sample_size=512,
        tokenization_mode="streamed",
        max_length=128,
        chunk_size=256,
    )

    train_loader = pipeline["train_loader"]
    val_loader = pipeline["val_loader"]
    df = pipeline["df"]
    class_names = pipeline["class_names"]

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

    epoch_times = []
    print("\n" + "-" * 80)
    print("EPOCH METRICS")
    print("-" * 80)

    for epoch in range(1, 4):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        print(
            f"Epoch {epoch}/3 | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"time={elapsed:.2f}s"
        )

    avg_epoch = sum(epoch_times) / len(epoch_times)
    est_10_epochs = avg_epoch * 10

    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    print(f"Per-epoch times (s): {[round(x, 2) for x in epoch_times]}")
    print(f"Average epoch time (s): {avg_epoch:.2f}")
    print(f"Estimated 10-epoch training time (s): {est_10_epochs:.2f}")
    print(f"Estimated 10-epoch training time (min): {est_10_epochs / 60:.2f}")


if __name__ == "__main__":
    main()
