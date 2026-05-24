from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY_FILES = sorted(
    path
    for path in ROOT.glob("**/*_summary_*.csv")
    if path.name not in {"gb_fullbins_all_summary_runs.csv"}
)


def first_present(row, names):
    for name in names:
        if name in row and pd.notna(row[name]) and row[name] != "":
            return row[name]
    return ""


def to_float(value):
    if value == "" or pd.isna(value):
        return None
    return float(value)


def normalize_summary(path):
    df = pd.read_csv(path)
    rows = []
    batch = path.parent.name

    for _, row in df.iterrows():
        method = str(row.get("method", ""))
        aep = to_float(first_present(row, ["AEP GB bins [GWh]", "AEP SS bins [GWh]", "AEP [GWh]"]))
        gain = to_float(
            first_present(
                row,
                [
                    "Improvement over source layout using GB bins [GWh]",
                    "Improvement over SS layout using GB bins [GWh]",
                    "Improvement over SmartStart [GWh]",
                ],
            )
        )

        rows.append(
            {
                "batch": batch,
                "summary_file": path.name,
                "script": row.get("script", ""),
                "source_script": row.get("source_script", ""),
                "source_method": row.get("source_method", ""),
                "layout_method": row.get("layout_method", ""),
                "source_opt_bin_label": row.get("source_opt_bin_label", ""),
                "wake_model": row.get("wake_model", ""),
                "seed": row.get("seed", ""),
                "method": method,
                "run_type": "GB full bins" if "GB" in method else "baseline",
                "aep_gwh": aep,
                "gain_gwh": gain,
                "runtime_min": to_float(row.get("runtime_sec", 0)) / 60,
                "iterations": row.get("iterations", ""),
                "gb_bin_setup": row.get("gb_bin_setup", ""),
                "ss_bin_setup": row.get("ss_bin_setup", ""),
            }
        )

    return rows


def write_markdown_table(df, path):
    visible = df[
        [
            "batch",
            "source_method",
            "layout_method",
            "seed",
            "method",
            "run_type",
            "aep_gwh",
            "gain_gwh",
            "runtime_min",
            "iterations",
        ]
    ].copy()

    visible["aep_gwh"] = visible["aep_gwh"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    visible["gain_gwh"] = visible["gain_gwh"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    visible["runtime_min"] = visible["runtime_min"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Results GB FullBins - samlet tabel\n\n")
        handle.write("Genereret fra alle `*summary*.csv` filer i denne mappe.\n\n")
        handle.write(visible.to_markdown(index=False))
        handle.write("\n\n")

        gb = df[df["run_type"] == "GB full bins"].copy()
        best = gb.sort_values("aep_gwh", ascending=False).head(10)
        best_visible = best[
            ["batch", "source_method", "layout_method", "seed", "method", "aep_gwh", "gain_gwh", "runtime_min"]
        ].copy()
        best_visible["aep_gwh"] = best_visible["aep_gwh"].map(lambda x: f"{x:.3f}")
        best_visible["gain_gwh"] = best_visible["gain_gwh"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        best_visible["runtime_min"] = best_visible["runtime_min"].map(lambda x: f"{x:.1f}")

        handle.write("## Top GB full-bins runs\n\n")
        handle.write(best_visible.to_markdown(index=False))
        handle.write("\n")


def make_plot(df, path):
    gb = df[df["run_type"] == "GB full bins"].copy()
    gb = gb.sort_values(["method", "seed"])
    gb["label"] = gb.apply(lambda r: f"{r['method']} s{r['seed']}", axis=1)

    colors = {
        "SS--GB": "#287c8e",
        "SS--GB-fullbins": "#287c8e",
        "SS--2S--GB-fullbins": "#d1495b",
        "RS--GB-fullbins": "#edae49",
        "RS--2S--GB-fullbins": "#4f7cac",
    }
    bar_colors = [colors.get(method, "#5f6c7b") for method in gb["method"]]

    fig, ax = plt.subplots(figsize=(13, 6.8), constrained_layout=True)
    ax.bar(gb["label"], gb["aep_gwh"], color=bar_colors, edgecolor="#263238", linewidth=0.45)
    ax.axhline(701.865, color="#555555", linestyle="--", linewidth=1.1, label="Current Horns Rev 701.865 GWh")
    ax.set_title("Gradient Based FullBins runs")
    ax.set_ylabel("AEP [GWh]")
    ax.set_xlabel("Method and seed")
    ax.set_ylim(700, max(gb["aep_gwh"]) + 0.35)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=55)

    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")

    ax.legend(loc="upper left")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    if not SUMMARY_FILES:
        raise SystemExit("No summary CSV files found.")

    rows = []
    for path in SUMMARY_FILES:
        rows.extend(normalize_summary(path))

    df = pd.DataFrame(rows)
    df = df.sort_values(["run_type", "method", "seed", "batch"], ascending=[True, True, True, True])

    df.to_csv(ROOT / "gb_fullbins_all_summary_runs.csv", index=False)
    write_markdown_table(df, ROOT / "gb_fullbins_all_summary_runs.md")
    make_plot(df, ROOT / "gb_fullbins_aep_plot.png")

    gb_count = int((df["run_type"] == "GB full bins").sum())
    print(f"Wrote {len(df)} summary rows, including {gb_count} GB full-bins runs.")
    print(ROOT / "gb_fullbins_all_summary_runs.csv")
    print(ROOT / "gb_fullbins_all_summary_runs.md")
    print(ROOT / "gb_fullbins_aep_plot.png")


if __name__ == "__main__":
    main()
