import argparse
import os
import re
from pathlib import Path


def parse_training_log(log_path: Path):
    summary = {
        "exists": log_path.exists(),
        "used_cuda": False,
        "epochs_trained": None,
        "best_epoch": None,
        "val_acc_last": None,
        "test_accuracy": None,
    }

    if not log_path.exists():
        return summary

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    summary["used_cuda"] = "Using device: cuda" in text

    epoch_matches = re.findall(r"Epoch\s+(\d+)/10\s+\|.*?val_acc=([0-9.]+)", text)
    if epoch_matches:
        summary["epochs_trained"] = int(epoch_matches[-1][0])
        summary["val_acc_last"] = float(epoch_matches[-1][1])

    best_match = re.search(r"Best epoch:\s*(\d+)", text)
    if best_match:
        summary["best_epoch"] = int(best_match.group(1))

    test_match = re.search(r"Test Accuracy:\s*([0-9.]+)", text)
    if test_match:
        summary["test_accuracy"] = float(test_match.group(1))

    return summary


def check_files(project_root: Path, submission_dir: Path | None):
    required_for_review = [
        "data_pipeline.py",
        "trabsa_model.py",
        "train_and_evaluate.py",
        "train_10_epochs.py",
        "requirements.txt",
        "README.md",
        "data_exploration.png",
        "baseline_confusion_matrix.png",
    ]

    strongly_recommended = [
        "trabsa_confusion_matrix.png",
        "training_history.png",
        "model_comparison.png",
        "MID_REVIEW_SUMMARY.txt",
        "best_trabsa_model.pth",
        "train_10_epochs_output.log",
    ]

    print("=" * 78)
    print("MID-REVIEW READINESS CHECK")
    print("=" * 78)
    print(f"Project root: {project_root}")

    missing_required = []
    for rel in required_for_review:
        path = project_root / rel
        ok = path.exists()
        print(f"[{'OK' if ok else 'MISSING'}] required: {rel}")
        if not ok:
            missing_required.append(rel)

    print("-" * 78)
    missing_recommended = []
    for rel in strongly_recommended:
        path = project_root / rel
        ok = path.exists()
        print(f"[{'OK' if ok else 'MISSING'}] recommended: {rel}")
        if not ok:
            missing_recommended.append(rel)

    print("-" * 78)
    log_info = parse_training_log(project_root / "train_10_epochs_output.log")
    print("Training evidence:")
    if not log_info["exists"]:
        print("[MISSING] train_10_epochs_output.log not found")
    else:
        print(f"[{'OK' if log_info['used_cuda'] else 'WARN'}] CUDA used: {log_info['used_cuda']}")
        print(f"[INFO] epochs trained: {log_info['epochs_trained']}")
        print(f"[INFO] best epoch: {log_info['best_epoch']}")
        print(f"[INFO] last val_acc: {log_info['val_acc_last']}")
        print(f"[INFO] test_accuracy: {log_info['test_accuracy']}")

    if submission_dir:
        print("-" * 78)
        print(f"Submission folder check: {submission_dir}")
        if not submission_dir.exists():
            print("[MISSING] submission folder does not exist")
        else:
            sub_files = sorted(str(p.relative_to(submission_dir)) for p in submission_dir.rglob("*") if p.is_file())
            for rel in sub_files:
                print(f"[FILE] {rel}")

    print("=" * 78)
    if missing_required:
        print("STATUS: NOT READY (missing required items)")
        print("Missing required files:")
        for item in missing_required:
            print(f"- {item}")
    else:
        print("STATUS: MINIMUM READY")

    if missing_recommended:
        print("\nTo strengthen your presentation, add:")
        for item in missing_recommended:
            print(f"- {item}")

    print("\nUseful generation commands:")
    print("- python smoke_check.py")
    print("- python train_10_epochs.py 2>&1 | tee train_10_epochs_output.log")
    print("- python main_execution.py")


def main():
    parser = argparse.ArgumentParser(description="Check mid-review submission readiness")
    parser.add_argument("--project-root", default=".", help="Project root containing code/results")
    parser.add_argument("--submission-dir", default=None, help="Optional submission folder to inspect")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    submission_dir = Path(args.submission_dir).resolve() if args.submission_dir else None
    check_files(project_root, submission_dir)


if __name__ == "__main__":
    main()
