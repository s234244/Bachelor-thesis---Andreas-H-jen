import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from Windfarm_utils import set_wt, get_site, set_wake_model, calc_aep
from RandomSearch import run_randomsearch
from GradientBasedAlgorithm import run_gradient_from_layout


# ==================================================
# SETTINGS
# ==================================================

site_name = "HornsRev1"
mean_ws = 9.6

# Set True if RandomSearch should also use the expensive full bin setup.
# Set False to test whether Gradient Based improves when only GB uses full bins.
run_randomsearch_with_full_bins = True

rs_wd = np.arange(0, 360, 1) if run_randomsearch_with_full_bins else np.arange(0, 360, 10)
rs_ws = np.arange(3, 26, 1) if run_randomsearch_with_full_bins else np.array([mean_ws])

gb_wd = np.arange(0, 360, 1)
gb_ws = np.arange(3, 26, 1)

max_gradient_iterations = 200
spacing_D = 4
boundary_pad = 400
x_points = 20
y_points = 20

random_max_time_sec = 24 * 60 * 60
random_step_schedule_D = [
    (20, 1200),
    (5, 400),
    (1, 400),
]

wake_models = ["NOJ"]
seeds = [1]

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


def combine_histories(random_histories, gradient_history, random_runtime):
    combined = []
    iteration_offset = 0
    time_offset = 0.0

    for stage_idx, history in enumerate(random_histories):
        if not history:
            continue

        rows_to_add = history if stage_idx == 0 else history[1:]
        for row in rows_to_add:
            combined.append({
                "iteration": iteration_offset + row["iteration"],
                "phase": "RandomSearch",
                "AEP [GWh]": row["AEP [GWh]"],
                "elapsed_sec": time_offset + row.get("elapsed_sec", 0.0),
                "aep_bin_source": "RS bins",
            })

        iteration_offset += history[-1]["iteration"]
        time_offset += history[-1].get("elapsed_sec", 0.0)

    for row in gradient_history:
        combined.append({
            "iteration": iteration_offset + row["iteration"],
            "phase": "Gradient",
            "AEP [GWh]": row["AEP [GWh]"],
            "elapsed_sec": random_runtime + row.get("elapsed_sec", 0.0),
            "aep_bin_source": "GB bins",
        })

    return combined


def append_layout_rows(rows, layout_xy, script, wake_model, seed, method, aep_rs_bins, aep_gb_bins, runtime_sec):
    for turbine_id, (x, y) in enumerate(layout_xy):
        rows.append({
            "script": script,
            "wake_model": wake_model,
            "seed": seed,
            "method": method,
            "turbine_id": turbine_id,
            "x": x,
            "y": y,
            "AEP RS bins [GWh]": aep_rs_bins,
            "AEP GB bins [GWh]": aep_gb_bins,
            "runtime_sec": runtime_sec,
            "rs_bin_setup": bin_description(rs_wd, rs_ws),
            "gb_bin_setup": bin_description(gb_wd, gb_ws),
        })


# ==================================================
# MAIN
# ==================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(exist_ok=True)

    wt, _ = set_wt(site_name)
    boundary, x_ref, y_ref, site, n_wt, *_ = get_site(
        site_name=site_name,
        wt=wt,
        boundary_pad=boundary_pad,
        x_points=x_points,
        y_points=y_points,
    )
    horns_rev_layout = np.column_stack([x_ref, y_ref])

    summary_rows = []
    history_rows = []
    layout_rows = []

    print()
    print("==================================================")
    print("RS--GB full-bin Gradient test")
    print("==================================================")
    print(f"RandomSearch bins: {bin_description(rs_wd, rs_ws)}")
    print(f"Gradient bins:     {bin_description(gb_wd, gb_ws)}")
    print(f"RandomSearch full bins enabled: {run_randomsearch_with_full_bins}")
    print(f"Seeds: {seeds}")
    print()

    for wake_model_name in wake_models:
        wf_model = set_wake_model(wake_model_name, site, wt)

        baseline_rs = calc_aep(wf_model, horns_rev_layout, with_wake_loss=True, wd=rs_wd, ws=rs_ws)
        baseline_gb = calc_aep(wf_model, horns_rev_layout, with_wake_loss=True, wd=gb_wd, ws=gb_ws)

        for seed in seeds:
            print()
            print("--------------------------------------------------")
            print(f"Wake model: {wake_model_name}, seed: {seed}")
            print("--------------------------------------------------")

            random_layout = horns_rev_layout.copy()
            random_histories = []
            random_runtime = 0.0

            for stage_idx, (stage_step_D, stage_iterations) in enumerate(random_step_schedule_D, start=1):
                print()
                print(
                    f"RandomSearch stage {stage_idx}: "
                    f"{stage_iterations} iterations, max step {stage_step_D}D"
                )

                stage_res = run_randomsearch(
                    wf_model=wf_model,
                    wt=wt,
                    boundary=boundary,
                    n_wt=n_wt,
                    initial_layout=random_layout,
                    spacing_D=spacing_D,
                    max_iter=stage_iterations,
                    max_time=random_max_time_sec,
                    max_step=stage_step_D * wt.diameter(),
                    seed=seed + 1000 * stage_idx,
                    wd=rs_wd,
                    ws=rs_ws,
                )

                random_layout = stage_res["layout_xy"]
                random_histories.append(stage_res["aep_history"])
                random_runtime += stage_res["runtime_sec"]

            aep_rs_rs_bins = calc_aep(wf_model, random_layout, with_wake_loss=True, wd=rs_wd, ws=rs_ws)
            aep_rs_gb_bins = calc_aep(wf_model, random_layout, with_wake_loss=True, wd=gb_wd, ws=gb_ws)

            print()
            print(f"RS layout AEP with RS bins: {aep_rs_rs_bins:.3f} GWh")
            print(f"RS layout AEP with GB/full bins: {aep_rs_gb_bins:.3f} GWh")
            print(f"RandomSearch runtime: {random_runtime:.2f} s")

            grad_res = run_gradient_from_layout(
                wf_model=wf_model,
                wt=wt,
                boundary=boundary,
                n_wt=n_wt,
                initial_layout=random_layout,
                spacing_D=spacing_D,
                wd=gb_wd,
                ws=gb_ws,
                maxiter=max_gradient_iterations,
            )

            gradient_layout = grad_res["layout_xy"]
            gradient_runtime = grad_res.get("runtime_sec", 0.0)

            aep_gb_rs_bins = calc_aep(wf_model, gradient_layout, with_wake_loss=True, wd=rs_wd, ws=rs_ws)
            aep_gb_gb_bins = calc_aep(wf_model, gradient_layout, with_wake_loss=True, wd=gb_wd, ws=gb_ws)

            print()
            print(f"RS--GB layout AEP with RS bins: {aep_gb_rs_bins:.3f} GWh")
            print(f"RS--GB layout AEP with GB/full bins: {aep_gb_gb_bins:.3f} GWh")
            print(f"GB improvement using full bins: {aep_gb_gb_bins - aep_rs_gb_bins:.3f} GWh")
            print(f"Total RS--GB runtime: {random_runtime + gradient_runtime:.2f} s")

            summary_rows.extend([
                {
                    "script": "RunRS_GB_FullBins",
                    "wake_model": wake_model_name,
                    "seed": seed,
                    "method": "Current Horns Rev",
                    "AEP RS bins [GWh]": baseline_rs,
                    "AEP GB bins [GWh]": baseline_gb,
                    "Improvement over RS layout using GB bins [GWh]": np.nan,
                    "runtime_sec": 0.0,
                    "iterations": 0,
                    "rs_bin_setup": bin_description(rs_wd, rs_ws),
                    "gb_bin_setup": bin_description(gb_wd, gb_ws),
                    "random_step_schedule_D": str(random_step_schedule_D),
                },
                {
                    "script": "RunRS_GB_FullBins",
                    "wake_model": wake_model_name,
                    "seed": seed,
                    "method": "RS",
                    "AEP RS bins [GWh]": aep_rs_rs_bins,
                    "AEP GB bins [GWh]": aep_rs_gb_bins,
                    "Improvement over RS layout using GB bins [GWh]": 0.0,
                    "runtime_sec": random_runtime,
                    "iterations": sum(iterations for _, iterations in random_step_schedule_D),
                    "rs_bin_setup": bin_description(rs_wd, rs_ws),
                    "gb_bin_setup": bin_description(gb_wd, gb_ws),
                    "random_step_schedule_D": str(random_step_schedule_D),
                },
                {
                    "script": "RunRS_GB_FullBins",
                    "wake_model": wake_model_name,
                    "seed": seed,
                    "method": "RS--GB",
                    "AEP RS bins [GWh]": aep_gb_rs_bins,
                    "AEP GB bins [GWh]": aep_gb_gb_bins,
                    "Improvement over RS layout using GB bins [GWh]": aep_gb_gb_bins - aep_rs_gb_bins,
                    "runtime_sec": random_runtime + gradient_runtime,
                    "iterations": sum(iterations for _, iterations in random_step_schedule_D) + max_gradient_iterations,
                    "rs_bin_setup": bin_description(rs_wd, rs_ws),
                    "gb_bin_setup": bin_description(gb_wd, gb_ws),
                    "random_step_schedule_D": str(random_step_schedule_D),
                },
            ])

            combined_history = combine_histories(
                random_histories=random_histories,
                gradient_history=grad_res["aep_history"],
                random_runtime=random_runtime,
            )

            for row in combined_history:
                history_rows.append({
                    "run_name": f"RS_GB_seed{seed}",
                    "script": "RunRS_GB_FullBins",
                    "wake_model": wake_model_name,
                    "seed": seed,
                    "iteration": row["iteration"],
                    "phase": row["phase"],
                    "AEP [GWh]": row["AEP [GWh]"],
                    "elapsed_sec": row["elapsed_sec"],
                    "aep_bin_source": row["aep_bin_source"],
                    "rs_bin_setup": bin_description(rs_wd, rs_ws),
                    "gb_bin_setup": bin_description(gb_wd, gb_ws),
                })

            if save_layouts:
                append_layout_rows(
                    layout_rows,
                    horns_rev_layout,
                    "RunRS_GB_FullBins",
                    wake_model_name,
                    seed,
                    "Current Horns Rev",
                    baseline_rs,
                    baseline_gb,
                    0.0,
                )
                append_layout_rows(
                    layout_rows,
                    random_layout,
                    "RunRS_GB_FullBins",
                    wake_model_name,
                    seed,
                    "RS",
                    aep_rs_rs_bins,
                    aep_rs_gb_bins,
                    random_runtime,
                )
                append_layout_rows(
                    layout_rows,
                    gradient_layout,
                    "RunRS_GB_FullBins",
                    wake_model_name,
                    seed,
                    "RS--GB",
                    aep_gb_rs_bins,
                    aep_gb_gb_bins,
                    random_runtime + gradient_runtime,
                )

    summary_path = results_dir / f"rs_gb_fullbins_summary_{timestamp}.csv"
    history_path = results_dir / f"rs_gb_fullbins_history_{timestamp}.csv"
    layouts_path = results_dir / f"rs_gb_fullbins_layouts_{timestamp}.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    if save_layouts:
        pd.DataFrame(layout_rows).to_csv(layouts_path, index=False)

    print()
    print("Saved CSV files:")
    print(summary_path)
    print(history_path)
    if save_layouts:
        print(layouts_path)


if __name__ == "__main__":
    main()
