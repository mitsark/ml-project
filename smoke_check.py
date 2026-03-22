import traceback
import numpy as np
import torch

from data_pipeline import create_complete_pipeline, CURRENT_CLASS_NAMES, CURRENT_STATUS_TO_LABEL
from trabsa_model import TRABSA


def run_check(name, fn, results):
    try:
        fn()
        results.append((name, "PASS", ""))
    except Exception as exc:
        results.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))


def main():
    csv_path = "Combined Data.csv"
    results = []

    state = {
        "pipeline": None,
        "batch": None,
        "model": None,
        "logits": None,
    }

    def check_data_pipeline_build():
        pipeline = create_complete_pipeline(
            csv_path=csv_path,
            batch_size=8,
            sample_size=256,
            tokenization_mode="streamed",
            max_length=128,
            chunk_size=256,
        )
        state["pipeline"] = pipeline

        assert len(pipeline["class_names"]) == 7, "Expected 7 classes"
        assert set(pipeline["status_to_label"].keys()) == set(CURRENT_STATUS_TO_LABEL.keys())
        assert pipeline["df"]["label"].min() >= 0
        assert pipeline["df"]["label"].max() <= 6

    def check_batch_shapes():
        pipeline = state["pipeline"]
        assert pipeline is not None, "Pipeline not initialized"

        batch = next(iter(pipeline["train_loader"]))
        state["batch"] = batch

        assert batch["input_ids"].ndim == 2
        assert batch["attention_mask"].ndim == 2
        assert batch["label"].ndim == 1
        assert batch["input_ids"].shape[0] == batch["label"].shape[0]
        assert int(batch["label"].min()) >= 0
        assert int(batch["label"].max()) <= 6

    def check_model_forward():
        batch = state["batch"]
        assert batch is not None, "Batch not initialized"

        device = torch.device("cpu")
        model = TRABSA(num_classes=7).to(device)
        state["model"] = model

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            logits, token_weights = model(input_ids, attention_mask)

        state["logits"] = logits

        assert logits.shape[0] == input_ids.shape[0]
        assert logits.shape[1] == 7
        assert torch.isfinite(logits).all().item(), "Found non-finite logits"

        probs = torch.softmax(logits, dim=1)
        sums = probs.sum(dim=1).cpu().numpy()
        assert np.allclose(sums, np.ones_like(sums), atol=1e-5), "Softmax rows do not sum to 1"

        assert token_weights.ndim in (1, 2), "Unexpected token weight shape"

    def check_backward_step():
        model = state["model"]
        batch = state["batch"]
        assert model is not None and batch is not None

        device = torch.device("cpu")
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        assert torch.isfinite(loss).item(), "Loss is non-finite"

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    run_check("1) Data pipeline build", check_data_pipeline_build, results)
    run_check("2) Batch shape/label sanity", check_batch_shapes, results)
    run_check("3) Model forward/logits sanity", check_model_forward, results)
    run_check("4) Single backward step", check_backward_step, results)

    print("\n=== SMOKE CHECK RESULTS ===")
    for name, status, msg in results:
        if status == "PASS":
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name}: {msg}")

    failed = [r for r in results if r[1] == "FAIL"]
    if failed:
        raise SystemExit(1)

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nUnhandled error during smoke checks:")
        traceback.print_exc()
        raise
