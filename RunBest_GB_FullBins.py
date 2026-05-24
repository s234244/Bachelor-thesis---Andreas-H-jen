import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from Windfarm_utilsv3 import set_wt, get_site, set_wake_model, calc_aep
from SmartstartgradientV2 import run_gradient_from_layout


# ==================================================
# SETTINGS
# ==================================================

run_script_name = "RunSourceLayouts_GB_FullBins"
site_name = "HornsRev1"
source_wake_model = "NOJ"
source_opt_bin_label = "5 deg bins"
seeds = [1, 2, 3, 4, 5]

spacing_D = 4
boundary_pad = 400
x_points = 20
y_points = 20

highres_eval_path = Path("Results_CSV_PL_Comparison/final_layouts_highres_evaluation.csv")
sources = [
    {
        "source_script": "ResultsSS",
        "layout_source_path": Path("Results_CSV_PL_Comparison/results_ss_layouts_wd5deg_20260519_113744.csv"),
        "source_methods": [
            {"source_method": "SS", "layout_method": "SmartStart"},
            {"source_method": "SS--2S", "layout_method": "SS--2S"},
        ],
    },
    {
        "source_script": "ResultsRS",
        "layout_source_path": Path("Results_CSV_PL_Comparison/results_rs_layouts_wd5deg_20260519_121750.csv"),
        "source_methods": [
            {"source_method": "RS", "layout_method": "RS"},
            {"source_method": "RS--2S", "layout_method": "RS--2S"},
        ],
    },
]

gb_wd = np.arange(0, 360, 1)
gb_ws = np.arange(3, 26, 1)
max_gradient_iterations = 200

results_dir = Path("Results_GB_FullBins")
save_layouts = True


# ==================================================
# HELPERS
# ==================================================

def bin_description(wd, ws):
    wd_step = int(wd[1] - wd[0]) if len(wd) > 1 else 0
    if len(ws) == 1:
        ws_desc = f"mean_ws_{float(ws[0]):.1f}"
    else:
        ws_desc = f"ws{int(ws[0])}-{int(ws[-1])}_step{int(ws[1] - ws[0])}"
    return f"wd{wd_step}deg_{ws_desc}"


def gb_method_name(source_method):
    return f"{source_method}--GB-fullbins"


def get_source_fine_aep(eval_df, source_script, seed, source_method):
    mask = (
        (eval_df["script"] == source_script)
        & (eval_df["wake_model"] == source_wake_model)
        & (eval_df["seed"] == seed)
        & (eval_df["opt_bin_label"] == source_opt_bin_label)
        & (eval_df["method"] == source_method)
    )
    rows = eval_df.loc[mask]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0]["Fine AEP [GWh]"])


def load_source_layout(layout_df, source_script, seed, layout_method):
    mask = (
        (layout_df["script"] == source_script)
        & (layout_df["wake_model"] == source_wake_model)
        & (layout_df["seed"] == seed)
        & (layout_df["method"] == layout_method)
    )
    rows = layout_df.loc[mask].copy()
    if rows.empty:
        raise ValueError(f"No {source_script} {layout_method} layout found for seed {seed}.")

    rows = rows.sort_values("turbine_id")
    return rows[["x", "y"]].to_numpy(dtype=float)


def append_layout_rows(
    rows,
    layout_xy,
    source_script,
    source_method,
    layout_method,
    seed,
    method,
    aep_gb_bins,
    runtime_sec,
):
    for turbine_id, (x, y) in enumerate(layout_xy):
        rows.append({
            "script": run_script_name,
            "source_script": source_script,
            "source_opt_bin_label": source_opt_bin_label,
            "source_method": source_method,
            "layout_method": layout_method,
            "wake_model": source_wake_model,
            "seed": seed,
            "method": method,
            "turbine_id": turbine_id,
            "x": x,
            "y": y,
            "AEP GB bins [GWh]": aep_gb_bins,
            "runtime_sec": runtime_sec,
            "gb_bin_setup": bin_description(gb_wd, gb_ws),
        })


def write_outputs(summary_rows, history_rows, layout_rows, summary_path, history_path, layouts_path):
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    if save_layouts:
        pd.DataFrame(layout_rows).to_csv(layouts_path, index=False)


# ==================================================
# MAIN
# ==================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(exist_ok=True)

    summary_path = results_dir / f"source_layouts_gb_fullbins_summary_{timestamp}.csv"
    history_path = results_dir / f"source_layouts_gb_fullbins_history_{timestamp}.csv"
    layouts_path = results_dir / f"source_layouts_gb_fullbins_layouts_{timestamp}.csv"

    wt, _ = set_wt(site_name)
    boundary, x_ref, y_ref, site, n_wt, *_ = get_site(
        site_name=site_name,
        wt=wt,
        boundary_pad=boundary_pad,
        x_points=x_points,
        y_points=y_points,
    )
    horns_rev_layout = np.column_stack([x_ref, y_ref])
    wf_model = set_wake_model(source_wake_model, site, wt)

    eval_df = pd.read_csv(highres_eval_path)
    baseline_gb = calc_aep(wf_model, horns_rev_layout, with_wake_loss=True, wd=gb_wd, ws=gb_ws)

    summary_rows = []
    history_rows = []
    layout_rows = []
    total_runs = sum(len(source["source_methods"]) for source in sources) * len(seeds)
    run_idx = 0

    print()
    print("==================================================")
    print("Source layouts followed by full-bin GB")
    print("==================================================")
    print(f"Sources: SS, SS--2S, RS, RS--2S")
    print(f"Seeds: {seeds}")
    print(f"Gradient bins: {bin_description(gb_wd, gb_ws)}")
    print(f"Gradient max iterations: {max_gradient_iterations}")
    print(f"Total GB runs: {total_runs}")
    print()

    for source in sources:
        source_script = source["source_script"]
        layout_df = pd.read_csv(source["layout_source_path"])

        for method_config in source["source_methods"]:
            source_method = method_config["source_method"]
            layout_method = method_config["layout_method"]

            for seed in seeds:
                run_idx += 1
                output_method = gb_method_name(source_method)
                source_layout = load_source_layout(layout_df, source_script, seed, layout_method)
                source_fine_aep = get_source_fine_aep(eval_df, source_script, seed, source_method)
                source_aep_gb = calc_aep(wf_model, source_layout, with_wake_loss=True, wd=gb_wd, ws=gb_ws)

                print()
                print("--------------------------------------------------")
                print(f"Run {run_idx}/{total_runs}: {source_script} {source_method}, seed {seed}")
                print("--------------------------------------------------")
                if not np.isnan(source_fine_aep):
                    print(f"Source high-res AEP from evaluation file: {source_fine_aep:.3f} GWh")
                print(f"Source layout AEP with full bins: {source_aep_gb:.3f} GWh")

                grad_res = run_gradient_from_layout(
                    wf_model=wf_model,
                    wt=wt,
                    boundary=boundary,
                    n_wt=n_wt,
                    initial_layout=source_layout,
                    spacing_D=spacing_D,
                    wd=gb_wd,
                    ws=gb_ws,
                    maxiter=max_gradient_iterations,
                )

                gradient_layout = grad_res["layout_xy"]
                gradient_runtime = grad_res.get("runtime_sec", 0.0)
                gradient_aep_gb = calc_aep(
                    wf_model,
                    gradient_layout,
                    with_wake_loss=True,
                    wd=gb_wd,
                    ws=gb_ws,
                )

                common = {
                    "script": run_script_name,
                    "source_script": source_script,
                    "source_opt_bin_label": source_opt_bin_label,
                    "source_method": source_method,
                    "layout_method": layout_method,
                    "wake_model": source_wake_model,
                    "seed": seed,
                    "source_highres_AEP [GWh]": source_fine_aep,
                    "gb_bin_setup": bin_description(gb_wd, gb_ws),
                }

                summary_rows.extend([
                    {
                        **common,
                        "method": "Current Horns Rev",
                        "AEP GB bins [GWh]": baseline_gb,
                        "Improvement over source layout using GB bins [GWh]": np.nan,
                        "runtime_sec": 0.0,
                        "iterations": 0,
                    },
                    {
                        **common,
                        "method": source_method,
                        "AEP GB bins [GWh]": source_aep_gb,
                        "Improvement over source layout using GB bins [GWh]": 0.0,
                        "runtime_sec": 0.0,
                        "iterations": 0,
                    },
                    {
                        **common,
                        "method": output_method,
                        "AEP GB bins [GWh]": gradient_aep_gb,
                        "Improvement over source layout using GB bins [GWh]": gradient_aep_gb - source_aep_gb,
                        "runtime_sec": gradient_runtime,
                        "iterations": max_gradient_iterations,
                    },
                ])

                for row in grad_res["aep_history"]:
                    history_rows.append({
                        **common,
                        "run_name": f"{source_script}_{source_method}_GB_seed{seed}",
                        "iteration": row["iteration"],
                        "phase": "Gradient",
                        "AEP [GWh]": row["AEP [GWh]"],
                        "elapsed_sec": row.get("elapsed_sec", 0.0),
                        "aep_bin_source": "GB bins",
                    })

                if save_layouts:
                    append_layout_rows(
                        layout_rows,
                        horns_rev_layout,
                        source_script,
                        source_method,
                        layout_method,
                        seed,
                        "Current Horns Rev",
                        baseline_gb,
                        0.0,
                    )
                    append_layout_rows(
                        layout_rows,
                        source_layout,
                        source_script,
                        source_method,
                        layout_method,
                        seed,
                        source_method,
                        source_aep_gb,
                        0.0,
                    )
                    append_layout_rows(
                        layout_rows,
                        gradient_layout,
                        source_script,
                        source_method,
                        layout_method,
                        seed,
                        output_method,
                        gradient_aep_gb,
                        gradient_runtime,
                    )

                write_outputs(summary_rows, history_rows, layout_rows, summary_path, history_path, layouts_path)

                print()
                print(f"GB layout AEP with full bins: {gradient_aep_gb:.3f} GWh")
                print(f"GB improvement: {gradient_aep_gb - source_aep_gb:.3f} GWh")
                print(f"GB runtime: {gradient_runtime:.2f} s")
                print("Saved progress:")
                print(summary_path)
                print(history_path)
                if save_layouts:
                    print(layouts_path)

    print()
    print("Finished all source-layout GB runs.")


if __name__ == "__main__":
    main()
