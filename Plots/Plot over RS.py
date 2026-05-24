from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


base_dir = Path(__file__).resolve().parent.parent
csv_dir = base_dir / "CSV_RS"


def latest_csv(pattern):
    paths = sorted(
        csv_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    if not paths:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    return paths[0]


def summarize_summary(df, group_col):
    df_random = df[df["method"] == "RandomSearch"].copy()

    return (
        df_random.groupby(["wake_model", group_col], as_index=False)
        .agg(
            aep_mean=("AEP [GWh]", "mean"),
            aep_std=("AEP [GWh]", "std"),
            improvement_mean=("Improvement over Horns Rev [GWh]", "mean"),
            improvement_std=("Improvement over Horns Rev [GWh]", "std"),
            runtime_mean=("runtime_sec", "mean"),
            runtime_std=("runtime_sec", "std"),
            n_seeds=("seed", "nunique")
        )
        .fillna({
            "aep_std": 0.0,
            "improvement_std": 0.0,
            "runtime_std": 0.0
        })
        .sort_values(["wake_model", group_col])
    )


def add_step_case_labels(df):
    df = df.copy()

    def make_label(row):
        if row["method"] == "RandomSearch staged":
            schedule = row.get("step_schedule", "staged")
            schedule_text = str(schedule).replace("coarse_to_fine_", "")
            parts = [part.replace("D", "") for part in schedule_text.split("_")]
            return " -> ".join(f"{part}D" for part in parts)

        return f"{int(row['max_step_D'])}D"

    def make_order(row):
        if row["method"] == "RandomSearch staged":
            return 1000

        return float(row["max_step_D"])

    df["step_case"] = df.apply(make_label, axis=1)
    df["step_case_order"] = df.apply(make_order, axis=1)
    return df


def summarize_step_cases(df):
    df_cases = add_step_case_labels(
        df[df["method"].isin(["RandomSearch", "RandomSearch staged"])].copy()
    )

    return (
        df_cases.groupby(["wake_model", "step_case", "step_case_order"], as_index=False)
        .agg(
            aep_mean=("AEP [GWh]", "mean"),
            aep_std=("AEP [GWh]", "std"),
            improvement_mean=("Improvement over Horns Rev [GWh]", "mean"),
            improvement_std=("Improvement over Horns Rev [GWh]", "std"),
            runtime_mean=("runtime_sec", "mean"),
            runtime_std=("runtime_sec", "std"),
            n_seeds=("seed", "nunique")
        )
        .fillna({
            "aep_std": 0.0,
            "improvement_std": 0.0,
            "runtime_std": 0.0
        })
        .sort_values(["wake_model", "step_case_order"])
    )


def summarize_history(df, group_col):
    return (
        df.groupby(["wake_model", group_col, "iteration"], as_index=False)
        .agg(
            aep_mean=("AEP [GWh]", "mean"),
            aep_std=("AEP [GWh]", "std"),
            elapsed_mean=("elapsed_sec", "mean"),
            elapsed_std=("elapsed_sec", "std"),
            n_seeds=("seed", "nunique")
        )
        .fillna({"aep_std": 0.0, "elapsed_std": 0.0})
        .sort_values(["wake_model", group_col, "iteration"])
    )


def build_iteration_progress(df):
    progress_rows = []

    for (wake_model, iterations_setting, seed), df_seed in df.groupby(
        ["wake_model", "iterations_setting", "seed"]
    ):
        df_seed = df_seed.sort_values("iteration")
        recorded_max = df_seed["iteration"].max()

        if recorded_max <= 0:
            continue

        target_iterations = np.arange(0, int(iterations_setting) + 1)
        scaled_iteration = (
            df_seed["iteration"].to_numpy(dtype=float)
            / recorded_max
            * float(iterations_setting)
        )
        aep_values = df_seed["AEP [GWh]"].to_numpy(dtype=float)
        interpolated_aep = np.interp(target_iterations, scaled_iteration, aep_values)

        progress_rows.append(pd.DataFrame({
            "wake_model": wake_model,
            "iterations_setting": iterations_setting,
            "seed": seed,
            "rs_iteration": target_iterations,
            "AEP [GWh]": interpolated_aep
        }))

    df_progress = pd.concat(progress_rows, ignore_index=True)

    return (
        df_progress.groupby(
            ["wake_model", "iterations_setting", "rs_iteration"],
            as_index=False
        )
        .agg(
            aep_mean=("AEP [GWh]", "mean"),
            aep_std=("AEP [GWh]", "std"),
            n_seeds=("seed", "nunique")
        )
        .fillna({"aep_std": 0.0})
    )


def build_step_time_progress(df):
    progress_rows = []

    for key, df_seed in df.groupby(["wake_model", "seed", "run_name"]):
        wake_model, seed, run_name = key
        df_seed = df_seed.sort_values("elapsed_sec")
        final_elapsed = df_seed["elapsed_sec"].max()

        if final_elapsed <= 0:
            continue

        if "step_schedule" in df_seed.columns and df_seed["step_schedule"].notna().any():
            schedule = df_seed["step_schedule"].dropna().iloc[0]
            schedule_text = str(schedule).replace("coarse_to_fine_", "")
            parts = [part.replace("D", "") for part in schedule_text.split("_")]
            step_case = " -> ".join(f"{part}D" for part in parts)
            step_case_order = 1000
        else:
            step_case = f"{int(df_seed['max_step_D'].iloc[0])}D"
            step_case_order = float(df_seed["max_step_D"].iloc[0])

        target_progress = np.linspace(0.0, 1.0, 501)
        source_progress = df_seed["elapsed_sec"].to_numpy(dtype=float) / final_elapsed
        aep_values = df_seed["AEP [GWh]"].to_numpy(dtype=float)
        elapsed_values = df_seed["elapsed_sec"].to_numpy(dtype=float)

        progress_rows.append(pd.DataFrame({
            "wake_model": wake_model,
            "seed": seed,
            "run_name": run_name,
            "step_case": step_case,
            "step_case_order": step_case_order,
            "progress": target_progress,
            "elapsed_sec": np.interp(target_progress, source_progress, elapsed_values),
            "AEP [GWh]": np.interp(target_progress, source_progress, aep_values)
        }))

    df_progress = pd.concat(progress_rows, ignore_index=True)

    return (
        df_progress.groupby(
            ["wake_model", "step_case", "step_case_order", "progress"],
            as_index=False
        )
        .agg(
            elapsed_mean=("elapsed_sec", "mean"),
            aep_mean=("AEP [GWh]", "mean"),
            n_seeds=("seed", "nunique")
        )
        .sort_values(["wake_model", "step_case_order", "progress"])
    )


def build_step_iteration_progress(df):
    progress_rows = []

    for key, df_seed in df.groupby(["wake_model", "seed", "run_name"]):
        wake_model, seed, run_name = key
        df_seed = df_seed.sort_values("iteration")
        recorded_max = df_seed["iteration"].max()
        iterations_setting = int(df_seed["iterations_setting"].iloc[0])

        if recorded_max <= 0:
            continue

        if "step_schedule" in df_seed.columns and df_seed["step_schedule"].notna().any():
            schedule = df_seed["step_schedule"].dropna().iloc[0]
            schedule_text = str(schedule).replace("coarse_to_fine_", "")
            parts = [part.replace("D", "") for part in schedule_text.split("_")]
            step_case = " -> ".join(f"{part}D" for part in parts)
            step_case_order = 1000
        else:
            step_case = f"{int(df_seed['max_step_D'].iloc[0])}D"
            step_case_order = float(df_seed["max_step_D"].iloc[0])

        target_iterations = np.arange(0, iterations_setting + 1)
        source_iterations = (
            df_seed["iteration"].to_numpy(dtype=float)
            / recorded_max
            * float(iterations_setting)
        )
        aep_values = df_seed["AEP [GWh]"].to_numpy(dtype=float)

        progress_rows.append(pd.DataFrame({
            "wake_model": wake_model,
            "seed": seed,
            "run_name": run_name,
            "step_case": step_case,
            "step_case_order": step_case_order,
            "rs_iteration": target_iterations,
            "AEP [GWh]": np.interp(target_iterations, source_iterations, aep_values)
        }))

    df_progress = pd.concat(progress_rows, ignore_index=True)

    return (
        df_progress.groupby(
            ["wake_model", "step_case", "step_case_order", "rs_iteration"],
            as_index=False
        )
        .agg(
            aep_mean=("AEP [GWh]", "mean"),
            n_seeds=("seed", "nunique")
        )
        .sort_values(["wake_model", "step_case_order", "rs_iteration"])
    )


iterations_summary_path = latest_csv("rs_iterations_summary_*.csv")
iterations_history_path = latest_csv("rs_iterations_history_*.csv")
stepsize_summary_path = latest_csv("rs_stepsize_summary_*.csv")
stepsize_history_path = latest_csv("rs_stepsize_history_*.csv")

df_summary = pd.read_csv(iterations_summary_path)
df_history = pd.read_csv(iterations_history_path)
df_stepsize_summary = pd.read_csv(stepsize_summary_path)
df_stepsize_history = pd.read_csv(stepsize_history_path)

df_random = df_summary[df_summary["method"] == "RandomSearch"].copy()
history_mean = summarize_history(df_history, "iterations_setting")
iteration_progress = build_iteration_progress(df_history)
stepsize_summary = summarize_summary(df_stepsize_summary, "max_step_D")
stepsize_cases = summarize_step_cases(df_stepsize_summary)
stepsize_cases_single = stepsize_cases[stepsize_cases["step_case_order"] < 1000].copy()

if "step_schedule" in df_stepsize_history.columns:
    df_stepsize_history_single = df_stepsize_history[df_stepsize_history["step_schedule"].isna()].copy()
else:
    df_stepsize_history_single = df_stepsize_history.copy()

stepsize_history_mean = summarize_history(df_stepsize_history_single, "max_step_D")
stepsize_time_progress = build_step_time_progress(df_stepsize_history_single)

print(f"Using summary: {iterations_summary_path}")
print(f"Using history: {iterations_history_path}")
print(f"Using step-size summary: {stepsize_summary_path}")
print(f"Using step-size history: {stepsize_history_path}")

for wake_model, df_wake in df_random.groupby("wake_model"):
    n_seeds = df_wake["seed"].nunique()
    iterations_setting = int(df_wake["iterations"].max())
    mean_aep = df_wake["AEP [GWh]"].mean()
    std_aep = df_wake["AEP [GWh]"].std()
    mean_runtime = df_wake["runtime_sec"].mean()
    std_runtime = df_wake["runtime_sec"].std()

    print()
    print(f"{wake_model} - {iterations_setting} iterations")
    print(f"Seeds: {n_seeds}")
    print(f"Final AEP: {mean_aep:.3f} +/- {std_aep:.3f} GWh")
    print(f"Runtime: {mean_runtime:.1f} +/- {std_runtime:.1f} s")


# -------------------------
# AEP over time - individual seeds
# -------------------------
plt.figure(figsize=(10, 6))

for (wake_model, seed), df_seed in df_history.groupby(["wake_model", "seed"]):
    final_aep = df_seed["AEP [GWh]"].iloc[-1]
    runtime = df_seed["elapsed_sec"].iloc[-1]
    plt.plot(
        df_seed["elapsed_sec"],
        df_seed["AEP [GWh]"],
        linewidth=1.2,
        alpha=0.75,
        label=f"{wake_model} seed {seed} - {final_aep:.3f} GWh, {runtime:.0f} s"
    )

plt.title("RandomSearch AEP over time - 10,000 iteration runs")
plt.xlabel("Elapsed time [s]")
plt.ylabel("AEP [GWh]")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


# -------------------------
# AEP over time - mean of seeds
# -------------------------
plt.figure(figsize=(10, 6))

for (wake_model, iterations_setting), df_sub in history_mean.groupby(["wake_model", "iterations_setting"]):
    n_seeds = int(df_sub["n_seeds"].max())
    final_aep = df_sub["aep_mean"].iloc[-1]
    plt.plot(
        df_sub["elapsed_mean"],
        df_sub["aep_mean"],
        linewidth=2.5,
        label=f"{wake_model}, {int(iterations_setting)} iterations - mean of {n_seeds} seeds, AEP {final_aep:.3f} GWh"
    )
    plt.fill_between(
        df_sub["elapsed_mean"],
        df_sub["aep_mean"] - df_sub["aep_std"],
        df_sub["aep_mean"] + df_sub["aep_std"],
        alpha=0.18
    )

plt.title("RandomSearch mean AEP over time")
plt.xlabel("Mean elapsed time [s]")
plt.ylabel("AEP [GWh]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------
# Step size summary - AEP
# -------------------------
plt.figure(figsize=(9, 5))

for wake_model, df_sub in stepsize_summary.groupby("wake_model"):
    n_seeds = int(df_sub["n_seeds"].max())
    plt.plot(
        df_sub["max_step_D"],
        df_sub["aep_mean"],
        marker="o",
        linewidth=2,
        label=f"{wake_model} (mean of {n_seeds} seeds)"
    )
    plt.fill_between(
        df_sub["max_step_D"],
        df_sub["aep_mean"] - df_sub["aep_std"],
        df_sub["aep_mean"] + df_sub["aep_std"],
        alpha=0.18
    )

plt.title("RandomSearch AEP vs max step size")
plt.xlabel("Maximum random step size [D]")
plt.ylabel("AEP [GWh]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------
# Step size final AEP - boxplot
# -------------------------
df_step_box = add_step_case_labels(
    df_stepsize_summary[
        df_stepsize_summary["method"].isin(["RandomSearch", "RandomSearch staged"])
    ].copy()
)
df_step_box = df_step_box[df_step_box["step_case"] != "1D"].copy()
box_order = (
    df_step_box[["step_case", "step_case_order"]]
    .drop_duplicates()
    .sort_values("step_case_order")
)
box_data = [
    df_step_box[df_step_box["step_case"] == step_case]["AEP [GWh]"].to_numpy()
    for step_case in box_order["step_case"]
]

plt.figure(figsize=(10, 6))
plt.boxplot(
    box_data,
    tick_labels=box_order["step_case"],
    showmeans=True,
    meanline=True,
    medianprops={"color": "tab:orange", "linewidth": 1.5},
    meanprops={"color": "tab:green", "linewidth": 1.5, "linestyle": "--"}
)

for x_pos, step_case in enumerate(box_order["step_case"], start=1):
    y_values = df_step_box[df_step_box["step_case"] == step_case]["AEP [GWh]"]
    plt.scatter(
        np.full(len(y_values), x_pos),
        y_values,
        s=35,
        alpha=0.75,
        zorder=3
    )

plt.title("RandomSearch final AEP by step size")
plt.xlabel("Step size")
plt.ylabel("Final AEP [GWh]")
plt.grid(True, axis="y", alpha=0.3)
plt.legend(
    handles=[
        Line2D([0], [0], color="tab:orange", linewidth=1.5, label="Median"),
        Line2D([0], [0], color="tab:green", linewidth=1.5, linestyle="--", label="Mean"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:blue",
               markeredgecolor="tab:blue", markersize=6, label="Individual seed"),
        Line2D([0], [0], color="black", linewidth=1.2, label="Box: 25-75% range"),
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="-", label="Whiskers: spread"),
    ],
    fontsize=8,
    loc="best"
)
plt.tight_layout()
plt.show()


# -------------------------
# Step size summary - improvement
# -------------------------
plt.figure(figsize=(9, 5))

for wake_model, df_sub in stepsize_summary.groupby("wake_model"):
    n_seeds = int(df_sub["n_seeds"].max())
    plt.plot(
        df_sub["max_step_D"],
        df_sub["improvement_mean"],
        marker="o",
        linewidth=2,
        label=f"{wake_model} (mean of {n_seeds} seeds)"
    )
    plt.fill_between(
        df_sub["max_step_D"],
        df_sub["improvement_mean"] - df_sub["improvement_std"],
        df_sub["improvement_mean"] + df_sub["improvement_std"],
        alpha=0.18
    )

plt.title("RandomSearch improvement vs max step size")
plt.xlabel("Maximum random step size [D]")
plt.ylabel("Improvement over Horns Rev [GWh]")
plt.axhline(0, color="black", linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------
# Step size summary - runtime
# -------------------------
plt.figure(figsize=(9, 5))

for wake_model, df_sub in stepsize_summary.groupby("wake_model"):
    n_seeds = int(df_sub["n_seeds"].max())
    plt.plot(
        df_sub["max_step_D"],
        df_sub["runtime_mean"],
        marker="o",
        linewidth=2,
        label=f"{wake_model} (mean of {n_seeds} seeds)"
    )
    plt.fill_between(
        df_sub["max_step_D"],
        df_sub["runtime_mean"] - df_sub["runtime_std"],
        df_sub["runtime_mean"] + df_sub["runtime_std"],
        alpha=0.18
    )

plt.title("RandomSearch runtime vs max step size")
plt.xlabel("Maximum random step size [D]")
plt.ylabel("Runtime [s]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------
# Step size runtime - bar chart
# -------------------------
plt.figure(figsize=(10, 6))

for wake_model, df_sub in stepsize_cases_single.groupby("wake_model"):
    df_sub = df_sub.sort_values("step_case_order")
    x_pos = np.arange(len(df_sub))
    plt.bar(
        x_pos,
        df_sub["runtime_mean"] / 60.0,
        yerr=df_sub["runtime_std"] / 60.0,
        capsize=5,
        alpha=0.8,
        label=wake_model
    )
    plt.xticks(x_pos, df_sub["step_case"])

plt.title("RandomSearch runtime by step size")
plt.xlabel("Step size")
plt.ylabel("Mean runtime [min]")
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------
# Step size runtime vs final AEP
# -------------------------
plt.figure(figsize=(10, 6))

for wake_model, df_sub in stepsize_cases_single.groupby("wake_model"):
    df_sub = df_sub.sort_values("step_case_order")
    plt.scatter(
        df_sub["runtime_mean"] / 60.0,
        df_sub["aep_mean"],
        s=80,
        label=wake_model
    )

    for _, row in df_sub.iterrows():
        plt.annotate(
            row["step_case"],
            (row["runtime_mean"] / 60.0, row["aep_mean"]),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=9
        )

plt.title("RandomSearch runtime vs final AEP")
plt.xlabel("Mean runtime [min]")
plt.ylabel("Mean final AEP [GWh]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------
# Step size history - AEP over time
# -------------------------
plt.figure(figsize=(10, 6))

for (wake_model, step_case), df_sub in stepsize_time_progress.groupby(["wake_model", "step_case"], sort=False):
    n_seeds = int(df_sub["n_seeds"].max())
    final_aep = df_sub["aep_mean"].iloc[-1]
    plt.plot(
        df_sub["elapsed_mean"],
        df_sub["aep_mean"],
        linewidth=2,
        label=f"{wake_model}, {step_case} - mean of {n_seeds} seeds, AEP {final_aep:.3f} GWh"
    )

plt.title("RandomSearch AEP over time - max step sizes")
plt.xlabel("Mean elapsed time [s]")
plt.ylabel("AEP [GWh]")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


# -------------------------
# Step size history - time over evaluations
# -------------------------
plt.figure(figsize=(10, 6))

for (wake_model, max_step_D), df_sub in stepsize_history_mean.groupby(["wake_model", "max_step_D"]):
    n_seeds = int(df_sub["n_seeds"].max())
    plt.plot(
        df_sub["iteration"],
        df_sub["elapsed_mean"],
        linewidth=2,
        label=f"{wake_model}, max step {max_step_D}D (mean of {n_seeds} seeds)"
    )

plt.title("RandomSearch time over recorded evaluations - max step sizes")
plt.xlabel("Recorded evaluation")
plt.ylabel("Mean elapsed time [s]")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


# -------------------------
# AEP over RandomSearch iterations - mean of seeds
# -------------------------
plt.figure(figsize=(10, 6))

for (wake_model, iterations_setting), df_sub in iteration_progress.groupby(["wake_model", "iterations_setting"]):
    n_seeds = int(df_sub["n_seeds"].max())
    final_aep = df_sub["aep_mean"].iloc[-1]
    plt.plot(
        df_sub["rs_iteration"],
        df_sub["aep_mean"],
        linewidth=2.5,
        label=f"{wake_model}, {int(iterations_setting)} iterations - mean of {n_seeds} seeds, AEP {final_aep:.3f} GWh"
    )
    plt.fill_between(
        df_sub["rs_iteration"],
        df_sub["aep_mean"] - df_sub["aep_std"],
        df_sub["aep_mean"] + df_sub["aep_std"],
        alpha=0.18
    )

plt.title("RandomSearch mean AEP over iterations")
plt.xlabel("RandomSearch iteration")
plt.ylabel("AEP [GWh]")
plt.xlim(0, int(iteration_progress["iterations_setting"].max()))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------
# Final AEP per seed
# -------------------------
plt.figure(figsize=(9, 5))

for wake_model, df_sub in df_random.groupby("wake_model"):
    df_sub = df_sub.sort_values("seed")
    plt.plot(
        df_sub["seed"],
        df_sub["AEP [GWh]"],
        marker="o",
        linewidth=2,
        label=wake_model
    )
    plt.axhline(
        df_sub["AEP [GWh]"].mean(),
        color="black",
        linewidth=1,
        linestyle="--",
        alpha=0.6,
        label="Mean final AEP"
    )

plt.title("RandomSearch final AEP per seed")
plt.xlabel("Seed")
plt.ylabel("Final AEP [GWh]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------
# Runtime per seed
# -------------------------
plt.figure(figsize=(9, 5))

for wake_model, df_sub in df_random.groupby("wake_model"):
    df_sub = df_sub.sort_values("seed")
    plt.plot(
        df_sub["seed"],
        df_sub["runtime_sec"],
        marker="o",
        linewidth=2,
        label=wake_model
    )
    plt.axhline(
        df_sub["runtime_sec"].mean(),
        color="black",
        linewidth=1,
        linestyle="--",
        alpha=0.6,
        label="Mean runtime"
    )

plt.title("RandomSearch runtime per seed")
plt.xlabel("Seed")
plt.ylabel("Runtime [s]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
