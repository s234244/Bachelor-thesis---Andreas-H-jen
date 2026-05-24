import os
import time
from datetime import datetime
from importlib import import_module

import numpy as np
import pandas as pd

from Windfarm_utils import set_wt, get_site, set_wake_model, calc_aep
from SmartStart import run_smartstart

run_two_step_from_layout = import_module("2StepAlgorithm").run_two_step_from_layout


results_dir = "CSV_bins"
os.makedirs(results_dir, exist_ok=True)

seeds = [1, 2, 3]
site_name = "HornsRev1"
wake_models = ["NOJ"]

mean_ws = 9.6
ws_mean = np.array([mean_ws])

max_cycles = 2
x_points = 18
y_points = 18
spacing_D = 4
boundary_pad = 400

mean_ws_case_specs = [
    ("360wd_mean_ws_1deg", np.arange(0, 360, 1)),
    ("180wd_mean_ws_2deg", np.arange(0, 360, 2)),
    ("72wd_mean_ws", np.arange(0, 360, 5)),
    ("36wd_mean_ws", np.arange(0, 360, 10)),
]


def timed_aep(wf_model, layout, wd=None, ws=None):
    t0 = time.time()
    aep = calc_aep(wf_model, layout, with_wake_loss=True, wd=wd, ws=ws)
    return aep, time.time() - t0


def build_history(run_name, seed, smart_aep, smart_runtime, two_step_history):
    history = [{
        "run_name": run_name,
        "seed": seed,
        "iteration": 0,
        "phase": "SmartStart",
        "AEP [GWh]": smart_aep,
        "elapsed_sec": smart_runtime,
        "using_site_bins": False
    }]

    for i, row in enumerate(two_step_history[1:], start=1):
        history.append({
            "run_name": run_name,
            "seed": seed,
            "iteration": i,
            "phase": "Two-step",
            "AEP [GWh]": row["AEP [GWh]"],
            "elapsed_sec": smart_runtime + row.get("elapsed_sec", 0.0),
            "using_site_bins": False
        })

    return history


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    wt, _ = set_wt(site_name)
    boundary, x_ref, y_ref, site, n_wt, site_x_points, site_y_points, XX, YY, _ = get_site(
        site_name=site_name,
        wt=wt,
        boundary_pad=boundary_pad,
        x_points=x_points,
        y_points=y_points
    )
    horns_rev_layout = np.column_stack([x_ref, y_ref])

    bin_cases = []
    for case_name, wd in mean_ws_case_specs:
        bin_cases.append({
            "name": case_name,
            "wd": wd,
            "ws": ws_mean,
            "speed_mode": "mean_ws"
        })

    rows = []
    history_rows = []

    for wake_model_name in wake_models:
        wf_model = set_wake_model(wake_model_name, site, wt)

        for seed in seeds:
            np.random.seed(seed)

            for case in bin_cases:
                wd = case["wd"]
                ws = case["ws"]
                case_name = case["name"]

                print(f"\nRunning {wake_model_name} - {case_name} - seed {seed}")

                aep_baseline, baseline_runtime = timed_aep(wf_model, horns_rev_layout, wd, ws)

                t0 = time.time()
                smart_res = run_smartstart(
                    wf_model=wf_model,
                    wt=wt,
                    boundary=boundary,
                    n_wt=n_wt,
                    XX=XX,
                    YY=YY,
                    spacing_D=spacing_D,
                    seed=seed,
                    wd=wd,
                    ws=ws
                )
                smart_runtime = time.time() - t0
                smart_layout = smart_res["layout_xy"]
                aep_smart, _ = timed_aep(wf_model, smart_layout, wd, ws)

                t0 = time.time()
                two_step_res = run_two_step_from_layout(
                    wf_model=wf_model,
                    wt=wt,
                    boundary=boundary,
                    n_wt=n_wt,
                    initial_layout=smart_layout,
                    x_points=site_x_points,
                    y_points=site_y_points,
                    spacing_D=spacing_D,
                    max_cycles=max_cycles,
                    wd=wd,
                    ws=ws
                )
                two_step_runtime = time.time() - t0
                two_step_layout = two_step_res["layout_xy"]
                aep_two_step, _ = timed_aep(wf_model, two_step_layout, wd, ws)
                case_history = build_history(
                    run_name=case_name,
                    seed=seed,
                    smart_aep=aep_smart,
                    smart_runtime=smart_runtime,
                    two_step_history=two_step_res["aep_history"]
                )
                history_rows.extend(case_history)

                history_path = os.path.join(results_dir, f"{case_name}_seed{seed}.csv")
                pd.DataFrame(case_history).to_csv(history_path, index=False)

                rows.extend([
                    {
                        "wake_model": wake_model_name,
                        "bin_case": case_name,
                        "seed": seed,
                        "n_wd": len(wd),
                        "n_ws": len(ws),
                        "speed_mode": case["speed_mode"],
                        "method": "Baseline",
                        "AEP [GWh]": aep_baseline,
                        "runtime_sec": baseline_runtime
                    },
                    {
                        "wake_model": wake_model_name,
                        "bin_case": case_name,
                        "seed": seed,
                        "n_wd": len(wd),
                        "n_ws": len(ws),
                        "speed_mode": case["speed_mode"],
                        "method": "SmartStart",
                        "AEP [GWh]": aep_smart,
                        "runtime_sec": smart_runtime
                    },
                    {
                        "wake_model": wake_model_name,
                        "bin_case": case_name,
                        "seed": seed,
                        "n_wd": len(wd),
                        "n_ws": len(ws),
                        "speed_mode": case["speed_mode"],
                        "method": "SS--2S",
                        "AEP [GWh]": aep_two_step,
                        "runtime_sec": smart_runtime + two_step_runtime
                    },
                ])

                print(f"Baseline: {aep_baseline:.3f} GWh, {baseline_runtime:.2f} s")
                print(f"SmartStart: {aep_smart:.3f} GWh, {smart_runtime:.2f} s")
                print(f"SS--2S: {aep_two_step:.3f} GWh, {smart_runtime + two_step_runtime:.2f} s")

    df = pd.DataFrame(rows)
    df_history = pd.DataFrame(history_rows)
    df_summary = (
        df.groupby(
            ["wake_model", "bin_case", "n_wd", "n_ws", "speed_mode", "method"],
            dropna=False
        )
        .agg(
            aep_mean=("AEP [GWh]", "mean"),
            aep_std=("AEP [GWh]", "std"),
            runtime_mean_sec=("runtime_sec", "mean"),
            runtime_std_sec=("runtime_sec", "std"),
            n_runs=("runtime_sec", "size")
        )
        .reset_index()
    )

    output_path = os.path.join(results_dir, f"bins_runtime_comparison_{timestamp}.csv")
    history_output_path = os.path.join(results_dir, f"bins_runtime_history_{timestamp}.csv")
    summary_output_path = os.path.join(results_dir, f"bins_runtime_summary_{timestamp}.csv")
    df.to_csv(output_path, index=False)
    df_history.to_csv(history_output_path, index=False)
    df_summary.to_csv(summary_output_path, index=False)

    print("\nSaved:", output_path)
    print("Saved:", history_output_path)
    print("Saved:", summary_output_path)
    print(df)
    print(df_summary)


if __name__ == "__main__":
    main()
