from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


base_dir = Path(__file__).resolve().parent.parent
csv_dir = base_dir / "CSV_bins"

cases = [
    ("36wd_mean_ws", "10 deg wind direction bins x mean wind speed"),
    ("72wd_mean_ws", "5 deg wind direction bins x mean wind speed"),
    ("180wd_mean_ws_2deg", "2 deg wind direction bins x mean wind speed"),
    ("360wd_mean_ws_1deg", "1 deg wind direction bins x mean wind speed"),
]

case_labels = dict(cases)
case_order = [case_name for case_name, _ in cases]
case_sort_key = {case_name: i for i, case_name in enumerate(case_order)}


def load_latest_history():
    history_paths = sorted(
        csv_dir.rglob("bins_runtime_history_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    if not history_paths:
        return None

    return pd.read_csv(history_paths[0])


def load_case_runs(case_name):
    seed_paths = sorted(csv_dir.glob(f"{case_name}_seed*.csv"))

    if seed_paths:
        runs = []
        for path in seed_paths:
            df = pd.read_csv(path)
            if "seed" not in df.columns:
                seed_text = path.stem.rsplit("_seed", 1)[-1]
                df["seed"] = int(seed_text)
            runs.append(df)
        return runs

    fallback_path = csv_dir / f"{case_name}.csv"
    if fallback_path.exists():
        return [pd.read_csv(fallback_path)]

    df_history = load_latest_history()
    if df_history is not None and "run_name" in df_history.columns:
        df_case = df_history[df_history["run_name"] == case_name].copy()
        if not df_case.empty:
            if "seed" not in df_case.columns:
                return [df_case]
            return [
                df_seed.copy()
                for _, df_seed in df_case.groupby("seed", sort=True)
            ]

    print(f"Skipping missing files/history for: {case_name}")
    return []


def average_by_iteration(runs):
    df_all = pd.concat(runs, ignore_index=True)

    return (
        df_all.groupby("iteration", as_index=False)
        .agg(
            elapsed_mean=("elapsed_sec", "mean"),
            elapsed_std=("elapsed_sec", "std"),
            aep_mean=("AEP [GWh]", "mean"),
            aep_std=("AEP [GWh]", "std"),
            n_seeds=("seed", "nunique") if "seed" in df_all.columns else ("elapsed_sec", "size")
        )
        .fillna({"elapsed_std": 0.0, "aep_std": 0.0})
    )


def load_latest_summary():
    summary_paths = sorted(
        csv_dir.rglob("bins_runtime_summary_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    if not summary_paths:
        print("Skipping scatter plot: no bins_runtime_summary_*.csv found")
        return None

    return pd.read_csv(summary_paths[0])


runs = []

for case_name, case_label in cases:
    case_runs = load_case_runs(case_name)
    if not case_runs:
        continue

    df_avg = average_by_iteration(case_runs)
    n_seeds = int(df_avg["n_seeds"].max())
    runs.append((case_name, case_label, df_avg, n_seeds))

# -------------------------
# AEP over time
# -------------------------
plt.figure(figsize=(10, 6))

for _, label, df, n_seeds in runs:
    legend_label = f"{label} (mean of {n_seeds} seeds)" if n_seeds > 1 else label
    plt.plot(df["elapsed_mean"], df["aep_mean"], label=legend_label)

plt.xlabel("Time (seconds)")
plt.ylabel("AEP (GWh)")
plt.title("Mean AEP over time")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------
# Runtime vs final AEP
# -------------------------
df_summary = load_latest_summary()

if df_summary is not None:
    df_final = df_summary[
        (df_summary["method"] == "SS--2S") &
        (df_summary["bin_case"].isin(case_labels))
    ].copy()
    df_final["case_order"] = df_final["bin_case"].map(case_sort_key)
    df_final = df_final.sort_values("case_order")

    y_col = "reference_aep_mean" if "reference_aep_mean" in df_final.columns else "aep_mean"
    y_label = (
        "Mean reference-AEP (GWh)"
        if y_col == "reference_aep_mean"
        else "Mean final AEP on case bins (GWh)"
    )
    title = (
        "Mean runtime vs reference-AEP"
        if y_col == "reference_aep_mean"
        else "Mean runtime vs final AEP on case bins"
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(df_final["runtime_mean_sec"], df_final[y_col], s=80)

    for _, row in df_final.iterrows():
        plt.annotate(
            case_labels[row["bin_case"]],
            (row["runtime_mean_sec"], row[y_col]),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=9
        )

    plt.xlabel("Mean runtime (seconds)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# -------------------------
# Time over iterations
# -------------------------
plt.figure(figsize=(10, 6))

for _, label, df, n_seeds in runs:
    legend_label = f"{label} (mean of {n_seeds} seeds)" if n_seeds > 1 else label
    plt.plot(df["iteration"], df["elapsed_mean"], label=legend_label)

plt.xlabel("Iterations")
plt.ylabel("Time (seconds)")
plt.title("Mean time over iterations")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------
# Time over iterations - mean wind speed
# -------------------------
plt.figure(figsize=(10, 6))

for case_name, label, df, n_seeds in runs:
    legend_label = f"{label} (mean of {n_seeds} seeds)" if n_seeds > 1 else label
    plt.plot(df["iteration"], df["elapsed_mean"], label=legend_label)

plt.xlabel("Iterations")
plt.ylabel("Time (seconds)")
plt.title("Mean time over iterations - mean wind speed")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
