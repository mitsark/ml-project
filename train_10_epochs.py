import time
import numpy as np
import torch
import torch.nn as nn
import os
from datetime import datetime

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

    csv_path = os.getenv("DATA_CSV", "Combined Data.csv")
    print(f"Using dataset: {csv_path}")

    print("=" * 80)
    print("TRABSA 10-EPOCH TRAINING")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    pipeline = create_complete_pipeline(
        csv_path=csv_path,
        batch_size=8,
        sample_size=None,  # Use full dataset
        tokenization_mode="streamed",
        max_length=128,
        chunk_size=256,
    )

    train_loader = pipeline["train_loader"]
    val_loader = pipeline["val_loader"]
    test_loader = pipeline["test_loader"]
    df = pipeline["df"]
    class_names = pipeline["class_names"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    epoch_times = []
    print("\n" + "-" * 80)
    print("TRAINING PROGRESS")
    print("-" * 80)

    best_val_loss = float('inf')
    best_epoch = 0
    patience = 3
    patience_counter = 0

    for epoch in range(1, 11):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:2d}/10 | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | "
            f"time={elapsed:.2f}s"
        )

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_trabsa_model.pth")
            print(f"  → Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  → Early stopping triggered (no improvement for {patience} epochs)")
                break

    # Final test evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)

    # Load best model
    if os.path.exists("best_trabsa_model.pth"):
        model.load_state_dict(torch.load("best_trabsa_model.pth"))
        print(f"Loaded best model from epoch {best_epoch}")

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    print(f"Total epochs trained: {len(epoch_times)}")
    print(f"Per-epoch times (s): {[round(x, 2) for x in epoch_times]}")
    print(f"Average epoch time (s): {sum(epoch_times) / len(epoch_times):.2f}")
    print(f"Total training time (s): {sum(epoch_times):.2f}")
    print(f"Total training time (min): {sum(epoch_times) / 60:.2f}")
    print(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
