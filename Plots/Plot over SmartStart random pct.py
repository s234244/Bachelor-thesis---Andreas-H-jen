from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


base_dir = Path(__file__).resolve().parent.parent
csv_dir = base_dir / "CSV_SS"
table_random_pcts = [0, 3, 5, 10, 20, 50]


def latest_csv():
    latest_path = csv_dir / "smartstart_random_pct_compare.csv"
    if latest_path.exists():
        return latest_path

    paths = sorted(
        csv_dir.glob("smartstart_random_pct_compare_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    if not paths:
        raise FileNotFoundError("No SmartStart random_pct CSV found")

    return paths[0]


def random_pct_label(random_pct):
    return f"{random_pct:g}%"


def latex_random_pct_table(df_table):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Sensitivity analysis of the SmartStart \texttt{random\_pct} parameter}",
        r"\label{tab:random_pct}",
        "",
        r"\begin{tabular}{cccccc}",
        r"\hline",
        r"random\_pct & Mean AEP [GWh] & Best AEP [GWh] & Std. Dev. & Runtime [min] & Seed count \\",
        r"\hline",
    ]

    for _, row in df_table.iterrows():
        lines.append(
            f"{row['random_pct_fraction']:.2f} & "
            f"{row['Mean AEP [GWh]']:.3f} & "
            f"{row['Best AEP [GWh]']:.3f} & "
            f"{row['Std. Dev.']:.3f} & "
            f"{row['Runtime [min]']} & "
            f"{int(row['Seed count'])} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def build_random_pct_table(final_rows):
    df_final = final_rows[final_rows["random_pct"].isin(table_random_pcts)]
    aggregations = {
        "Mean AEP [GWh]": ("AEP [GWh]", "mean"),
        "Best AEP [GWh]": ("AEP [GWh]", "max"),
        "Std. Dev.": ("AEP [GWh]", "std"),
        "Seed count": ("seed", "nunique"),
    }

    if "Runtime [min]" in df_final.columns:
        aggregations["Runtime [min]"] = ("Runtime [min]", "mean")

    df_table = (
        df_final.groupby("random_pct", as_index=False)
        .agg(**aggregations)
        .fillna({"Std. Dev.": 0.0})
        .sort_values("random_pct")
    )
    df_table["random_pct_fraction"] = df_table["random_pct"] / 100.0
    if "Runtime [min]" in df_table.columns:
        df_table["Runtime [min]"] = df_table["Runtime [min]"].map(lambda value: f"{value:.2f}")
    else:
        df_table["Runtime [min]"] = ""

    return df_table[
        [
            "random_pct_fraction",
            "random_pct",
            "Mean AEP [GWh]",
            "Best AEP [GWh]",
            "Std. Dev.",
            "Runtime [min]",
            "Seed count",
        ]
    ]


csv_path = latest_csv()
df = pd.read_csv(csv_path)

df_mean = (
    df.groupby(["random_pct", "iteration"], as_index=False)
    .agg(
        aep_mean=("AEP [GWh]", "mean"),
        aep_std=("AEP [GWh]", "std"),
        n_seeds=("seed", "nunique")
    )
    .fillna({"aep_std": 0.0})
    .sort_values(["random_pct", "iteration"])
)

random_pcts = sorted(df_mean["random_pct"].unique())
n_seeds = int(df_mean["n_seeds"].max())
baseline_pct = random_pcts[0]

print(f"Using: {csv_path}")

final_rows = (
    df.sort_values("iteration")
    .groupby(["seed", "random_pct"], as_index=False)
    .tail(1)
)

random_pct_table = build_random_pct_table(final_rows)
random_pct_table_csv_path = csv_dir / "random_pct_sensitivity_table.csv"
random_pct_table_latex_path = csv_dir / "random_pct_sensitivity_table.tex"

random_pct_table.to_csv(random_pct_table_csv_path, index=False)
random_pct_table_latex_path.write_text(
    latex_random_pct_table(random_pct_table),
    encoding="utf-8"
)

print(f"Saved table to: {random_pct_table_csv_path}")
print(f"Saved table to: {random_pct_table_latex_path}")

# -------------------------
# AEP over iterations
# -------------------------
plt.figure(figsize=(10, 6))

for random_pct in random_pcts:
    df_sub = df_mean[df_mean["random_pct"] == random_pct]
    plt.plot(
        df_sub["iteration"],
        df_sub["aep_mean"],
        marker="o",
        markersize=4,
        markevery=8,
        linewidth=2,
        label=f"random_pct = {random_pct_label(random_pct)}"
    )

plt.title(f"SmartStart AEP vs random_pct (mean of {n_seeds} seeds)")
plt.xlabel("Iteration")
plt.ylabel("AEP [GWh]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# Difference from baseline random_pct
# -------------------------
baseline = (
    df_mean[df_mean["random_pct"] == baseline_pct][["iteration", "aep_mean"]]
    .rename(columns={"aep_mean": "baseline_aep"})
)

plt.figure(figsize=(10, 6))

for random_pct in random_pcts:
    if random_pct == baseline_pct:
        continue

    df_sub = df_mean[df_mean["random_pct"] == random_pct].merge(baseline, on="iteration")
    delta = df_sub["aep_mean"] - df_sub["baseline_aep"]

    plt.plot(
        df_sub["iteration"],
        delta,
        marker="o",
        markersize=4,
        markevery=8,
        linewidth=2,
        label=f"{random_pct_label(random_pct)} - {random_pct_label(baseline_pct)}"
    )

plt.axhline(0, color="black", linewidth=0.8)
plt.title(f"SmartStart difference from random_pct = {random_pct_label(baseline_pct)}")
plt.xlabel("Iteration")
plt.ylabel("Delta AEP [GWh]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# Final AEP vs random_pct
# -------------------------
final_mean = (
    final_rows.groupby("random_pct", as_index=False)
    .agg(
        final_aep_mean=("AEP [GWh]", "mean"),
        final_aep_std=("AEP [GWh]", "std"),
        n_seeds=("seed", "nunique")
    )
    .fillna({"final_aep_std": 0.0})
    .sort_values("random_pct")
)

plt.figure(figsize=(9, 5))
plt.errorbar(
    final_mean["random_pct"],
    final_mean["final_aep_mean"],
    yerr=final_mean["final_aep_std"],
    marker="o",
    linewidth=2,
    capsize=4
)

plt.title(f"Final SmartStart AEP vs random_pct (mean of {n_seeds} seeds)")
plt.xlabel("random_pct [%]")
plt.ylabel("Final AEP [GWh]")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
