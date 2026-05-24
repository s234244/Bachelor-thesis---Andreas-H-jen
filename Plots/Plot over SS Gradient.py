from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


base_dir = Path(__file__).resolve().parent.parent
csv_dir = base_dir / "CSV_Gradient"


def latest_csv(pattern):
    paths = sorted(
        csv_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    if not paths:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    return paths[0]


def summarize(df):
    df_gradient = df[df["method"] == "SS-Gradient"].copy()

    return (
        df_gradient.groupby(["wake_model", "gradient_iterations"], as_index=False)
        .agg(
            aep_mean=("AEP [GWh]", "mean"),
            improvement_horns_rev_mean=("Improvement over Horns Rev [GWh]", "mean"),
            improvement_smartstart_mean=("Improvement over SmartStart [GWh]", "mean"),
            runtime_mean=("runtime_sec", "mean"),
            recorded_iterations_mean=("recorded_gradient_iterations", "mean"),
            n_seeds=("seed", "nunique")
        )
        .sort_values("gradient_iterations")
    )


def summarize_history(df):
    return (
        df.groupby(["wake_model", "gradient_iterations_setting", "iteration"], as_index=False)
        .agg(
            aep_mean=("AEP [GWh]", "mean"),
            elapsed_mean=("elapsed_sec", "mean"),
            n_seeds=("seed", "nunique")
        )
        .sort_values(["gradient_iterations_setting", "iteration"])
    )


results_path = latest_csv("ss_gradient_iterations_all_*.csv")

df_all = pd.read_csv(results_path)
df_summary = df_all[df_all["row_type"] == "summary"].copy()
df_history = df_all[df_all["row_type"] == "history"].copy()

summary = summarize(df_summary)
history = summarize_history(df_history)

print(f"Using: {results_path}")

# -------------------------
# SS-Gradient iterations summary
# -------------------------
plt.figure(figsize=(9, 5))

for wake_model, df_sub in summary.groupby("wake_model"):
    n_seeds = int(df_sub["n_seeds"].max())
    plt.plot(
        df_sub["gradient_iterations"],
        df_sub["aep_mean"],
        marker="o",
        linewidth=2,
        label=f"{wake_model} (mean of {n_seeds} seeds)"
    )

plt.title("SS-Gradient AEP vs gradient iterations")
plt.xlabel("Gradient max iterations")
plt.ylabel("AEP [GWh]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))

for wake_model, df_sub in summary.groupby("wake_model"):
    n_seeds = int(df_sub["n_seeds"].max())
    plt.plot(
        df_sub["gradient_iterations"],
        df_sub["improvement_smartstart_mean"],
        marker="o",
        linewidth=2,
        label=f"{wake_model} (mean of {n_seeds} seeds)"
    )

plt.title("SS-Gradient improvement over SmartStart vs iterations")
plt.xlabel("Gradient max iterations")
plt.ylabel("Improvement over SmartStart [GWh]")
plt.axhline(0, color="black", linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))

for wake_model, df_sub in summary.groupby("wake_model"):
    n_seeds = int(df_sub["n_seeds"].max())
    plt.plot(
        df_sub["gradient_iterations"],
        df_sub["improvement_horns_rev_mean"],
        marker="o",
        linewidth=2,
        label=f"{wake_model} (mean of {n_seeds} seeds)"
    )

plt.title("SS-Gradient improvement over Horns Rev vs iterations")
plt.xlabel("Gradient max iterations")
plt.ylabel("Improvement over Horns Rev [GWh]")
plt.axhline(0, color="black", linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))

for wake_model, df_sub in summary.groupby("wake_model"):
    n_seeds = int(df_sub["n_seeds"].max())
    plt.plot(
        df_sub["gradient_iterations"],
        df_sub["runtime_mean"],
        marker="o",
        linewidth=2,
        label=f"{wake_model} (mean of {n_seeds} seeds)"
    )

plt.title("SS-Gradient runtime vs gradient iterations")
plt.xlabel("Gradient max iterations")
plt.ylabel("Runtime [s]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))

for wake_model, df_sub in summary.groupby("wake_model"):
    n_seeds = int(df_sub["n_seeds"].max())
    plt.plot(
        df_sub["gradient_iterations"],
        df_sub["recorded_iterations_mean"],
        marker="o",
        linewidth=2,
        label=f"{wake_model} (mean of {n_seeds} seeds)"
    )

plt.title("Recorded SLSQP iterations vs max iterations")
plt.xlabel("Gradient max iterations")
plt.ylabel("Recorded gradient iterations")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# History from CSV
# -------------------------
plt.figure(figsize=(10, 6))

for gradient_iterations, df_sub in history.groupby("gradient_iterations_setting"):
    n_seeds = int(df_sub["n_seeds"].max())
    df_sub = df_sub[df_sub["n_seeds"] == n_seeds]
    plt.plot(
        df_sub["iteration"],
        df_sub["aep_mean"],
        linewidth=2,
        label=f"{gradient_iterations} max iterations (mean of {n_seeds} seeds)"
    )

plt.title("SS-Gradient AEP history")
plt.xlabel("Recorded gradient iteration")
plt.ylabel("AEP [GWh]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

for gradient_iterations, df_sub in history.groupby("gradient_iterations_setting"):
    n_seeds = int(df_sub["n_seeds"].max())
    df_sub = df_sub[df_sub["n_seeds"] == n_seeds]
    plt.plot(
        df_sub["iteration"],
        df_sub["elapsed_mean"],
        linewidth=2,
        label=f"{gradient_iterations} max iterations (mean of {n_seeds} seeds)"
    )

plt.title("SS-Gradient time over recorded iterations")
plt.xlabel("Recorded gradient iteration")
plt.ylabel("Elapsed time [s]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
