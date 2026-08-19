"""Re-measure the 5 excavation failures (plus 2 healthy controls) after the notch fix."""
import importlib.util, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

SRC = Path("/Users/joduq24/Desktop/decomb/studies/2026-08-19-arm-comparison/fixed_config.py")
spec = importlib.util.spec_from_file_location("study", SRC)
study = importlib.util.module_from_spec(spec); sys.modules["study"] = study
spec.loader.exec_module(study)

FAILURES = ["sub-0003_task-thermalactive_run-1_eeg",
            "sub-0006_task-thermalactive_run-1_eeg",
            "sub-0006_task-thermalactive_run-2_eeg",
            "sub-0013_task-thermalactive_run-1_eeg",
            "sub-0014_task-thermalactive_run-6_eeg"]
CONTROLS = ["sub-0013_task-thermalactive_run-3_eeg",
            "sub-0015_task-thermalactive_run-1_eeg"]
ARMS = ("notching_declared", "comb_subtracted", "combined")
t0 = time.perf_counter()

def main():
    names = FAILURES + CONTROLS
    rows = []
    out = Path(sys.argv[1])
    with ProcessPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(study._worker, (n, ARMS)) for n in names]
        for i, f in enumerate(as_completed(futs), 1):
            name, measured, err = f.result()
            if err:
                print(f"[{time.perf_counter()-t0:6.1f}s] [{i}/{len(names)}] {name} FAILED: "
                      f"{err.splitlines()[-1]}", flush=True); continue
            for r in measured: r["group"] = "failure" if name in FAILURES else "control"
            rows.extend(measured)
            pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
            by = {r["arm"]: r for r in measured}
            print(f"[{time.perf_counter()-t0:6.1f}s] [{i}/{len(names)}] {name}: " + "  ".join(
                f"{a}: comb {by[a]['comb_db']:+.2f} gamma {by[a]['gamma_kept']:.3f} "
                f"peaks {by[a]['peaks_above_2dB']}" for a in ARMS if a in by), flush=True)
    print(f"[{time.perf_counter()-t0:6.1f}s] done -> {out}", flush=True)

if __name__ == "__main__":
    main()
