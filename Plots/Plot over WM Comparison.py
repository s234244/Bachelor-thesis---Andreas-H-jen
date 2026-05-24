from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


base_dir = Path(__file__).resolve().parent.parent
csv_dir = base_dir / "CSV_WM_Comparison"


def latest_csv(pattern):
    paths = sorted(
        csv_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    if not paths:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    return paths[0]


history_path = latest_csv("results_ss_history_*.csv")
summary_path = latest_csv("results_ss_summary_*.csv")
df_history = pd.read_csv(history_path)
df_summary = pd.read_csv(summary_path)
df_smartstart = df_summary[df_summary["method"] == "SmartStart"].copy()

print(f"Using history: {history_path}")
print(f"Using summary: {summary_path}")

wake_models = sorted(df_smartstart["wake_model"].unique())
bar_width = 0.2
bar_spacing = 0.3
aep_means = []
aep_stds = []

plt.figure(figsize=(8, 5))
x_pos = np.arange(len(wake_models)) * bar_spacing

for wake_model in wake_models:
    values = df_smartstart[df_smartstart["wake_model"] == wake_model]["Surrogate AEP [GWh]"]
    aep_means.append(values.mean())
    aep_stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)

plt.bar(
    x_pos,
    aep_means,
    yerr=aep_stds,
    width=bar_width,
    capsize=5,
    alpha=0.8,
)

for i, wake_model in enumerate(wake_models):
    values = df_smartstart[df_smartstart["wake_model"] == wake_model]["Surrogate AEP [GWh]"]
    plt.scatter(
        np.full(len(values), x_pos[i]),
        values,
        s=35,
        alpha=0.75,
        zorder=3
    )

plt.title("AEP by wake model - SmartStart")
plt.xlabel("Wake model")
plt.ylabel("SmartStart AEP [GWh]")
plt.xticks(x_pos, wake_models)
plt.xlim(x_pos[0] - bar_spacing, x_pos[-1] + bar_spacing)
plt.ylim(70, 73)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# -------------------------
# Runtime by wake model
# -------------------------
df_runtime = df_smartstart.copy()
runtime_means = []
runtime_stds = []

plt.figure(figsize=(8, 5))
x_pos = np.arange(len(wake_models)) * bar_spacing

for wake_model in wake_models:
    values = df_runtime[df_runtime["wake_model"] == wake_model]["runtime_sec"].to_numpy() / 60.0
    runtime_means.append(values.mean())
    runtime_stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)

plt.bar(
    x_pos,
    runtime_means,
    yerr=runtime_stds,
    width=bar_width,
    capsize=5,
    alpha=0.8,
)

for i, wake_model in enumerate(wake_models):
    values = df_runtime[df_runtime["wake_model"] == wake_model]["runtime_sec"].to_numpy() / 60.0
    plt.scatter(
        np.full(len(values), x_pos[i]),
        values,
        s=35,
        alpha=0.75,
        zorder=3
    )

plt.title("Runtime by wake model - SmartStart")
plt.xlabel("Wake model")
plt.ylabel("Runtime [min]")
plt.xticks(x_pos, wake_models)
plt.xlim(x_pos[0] - bar_spacing, x_pos[-1] + bar_spacing)
plt.ylim(0, 5)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
