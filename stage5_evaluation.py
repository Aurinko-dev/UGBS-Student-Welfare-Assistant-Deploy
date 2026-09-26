"""Stage 5 evaluation runner.

Runs the project's existing evaluation pipeline on the main 87-message test
set and the separate crisis-v3 adversarial set. The existing eval_classifier.py
contains the same live triage path and baseline comparison; this wrapper keeps
all Stage 5 outputs together and makes the dataset scope explicit.

Important: if trained model weights are not present, eval_classifier.py uses the
rule-based fallback. In that situation the output must NOT be described as a
neural-network evaluation. Train the model first with train_classifier.py.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def run(cmd, log_path):
    print("\n$", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stage5_evaluation_output")
    ap.add_argument("--main-testset", default="eval_testset.csv")
    ap.add_argument("--crisis-testset", default="eval_testset_crisis_v3.csv")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rc1 = run([sys.executable, "eval_classifier.py", "--testset", args.main_testset,
               "--out", str(out / "main_87")], out / "main_run.log")
    rc2 = run([sys.executable, "eval_classifier.py", "--testset", args.crisis_testset,
               "--out", str(out / "crisis_v3_14")], out / "crisis_v3_run.log")

    status = "PASS" if rc1 == 0 and rc2 == 0 else "FAIL"
    (out / "STAGE5_STATUS.txt").write_text(
        f"{status}\nMain evaluation return code: {rc1}\nCrisis-v3 return code: {rc2}\n",
        encoding="utf-8")
    print(f"\nStage 5 evaluation runner: {status}")
    print("Main dataset: 87 messages (eval_testset.csv)")
    print("Crisis-v3 dataset: 14 messages (eval_testset_crisis_v3.csv)")
    print("Check the logs to confirm whether trained neural weights were available.")
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
