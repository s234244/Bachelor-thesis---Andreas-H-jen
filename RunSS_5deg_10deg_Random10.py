import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from Windfarm_utilsv3 import set_wt, get_site, set_wake_model, calc_aep
from SmartstartV2 import run_smartstart


# ==================================================
# SETTINGS
# ==================================================

site_name = "HornsRev1"
mean_ws = 9.6
ws_mean = np.array([mean_ws])

wd_steps_deg = [5, 10]
random_pct = 10
seeds = [1, 2, 3, 4, 5]

x_points = 20
y_points = 20
spacing_D = 4
boundary_pad = 400

wake_models = ["NOJ"]
results_dir = Path("Results_CSV_PL_Comparison_SS")
save_layouts = True


# ==================================================
# HELPERS
# ==================================================

def append_layout_rows(
    rows,
    layout_xy,
    wake_model_name,
    seed,
    method,
    aep,
    surrogate_aep,
    runtime_sec,
    wd_step_deg,
    n_wd,
):
    for turbine_id, (x, y) in enumerate(layout_xy):
        rows.append({
            "script": "ResultsSS",
            "wake_model": wake_model_name,
            "seed": seed,
            "method": method,
            "turbine_id": turbine_id,
            "x": x,
            "y": y,
            "AEP [GWh]": aep,
            "Surrogate AEP [GWh]": surrogate_aep,
            "runtime_sec": runtime_sec,
            "wd_step_deg": wd_step_deg,
            "n_wd": n_wd,
            "ws": mean_ws,
            "random_pct": random_pct,
        })


def save_outputs(summary_rows, history_rows, layout_rows, wd_step_deg, timestamp):
    wd_tag = f"wd{wd_step_deg}deg"
    summary_output_path = results_dir / f"results_ss_summary_{wd_tag}_{timestamp}.csv"
    history_output_path = results_dir / f"results_ss_history_{wd_tag}_{timestamp}.csv"
    layouts_output_path = results_dir / f"results_ss_layouts_{wd_tag}_{timestamp}.csv"

    pd.DataFrame(summary_rows).to_csv(summary_output_path, index=False)
    pd.DataFrame(history_rows).to_csv(history_output_path, index=False)
    if save_layouts:
        pd.DataFrame(layout_rows).to_csv(layouts_output_path, index=False)

    print()
    print("Saved:")
    print(summary_output_path)
    print(history_output_path)
    if save_layouts:
        print(layouts_output_path)


# ==================================================
# MAIN
# ==================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(exist_ok=True)

    wt, _ = set_wt(site_name)
    boundary, x_ref, y_ref, site, n_wt, _, _, XX, YY, _ = get_site(
        site_name=site_name,
        wt=wt,
        boundary_pad=boundary_pad,
        x_points=x_points,
        y_points=y_points,
    )
    horns_rev_layout = np.column_stack([x_ref, y_ref])

    print()
    print("==================================================")
    print("SmartStart-only rerun with explicit 10 percent randomness")
    print("==================================================")
    print(f"wd steps: {wd_steps_deg}")
    print(f"seeds: {seeds}")
    print(f"random_pct passed to SmartStart: {random_pct}")
    print("This script runs SmartStart only. It does not run 2S or GB.")

    for wd_step_deg in wd_steps_deg:
        wd_all = np.arange(0, 360, wd_step_deg)
        n_wd = len(wd_all)

        summary_rows = []
        history_rows = []
        layout_rows = []

        print()
        print("==================================================")
        print(f"Running SmartStart layouts with {wd_step_deg} degree bins")
        print("==================================================")
        print(f"wd_all: {wd_all[0]} to {wd_all[-1]} deg, step {wd_step_deg} deg, n={n_wd}")
        print(f"ws_mean: {ws_mean.tolist()}")

        for wake_model_name in wake_models:
            wf_model = set_wake_model(wake_model_name, site, wt)

            aep_baseline = calc_aep(
                wf_model,
                horns_rev_layout,
                with_wake_loss=True,
                wd=wd_all,
                ws=ws_mean,
            )
            aep_baseline_true = calc_aep(wf_model, horns_rev_layout, with_wake_loss=True)

            for seed in seeds:
                np.random.seed(seed)
                print()
                print("--------------------------------------------------")
                print(f"Wake model: {wake_model_name}, wd step: {wd_step_deg}, seed: {seed}")
                print("--------------------------------------------------")

                smart_res = run_smartstart(
                    wf_model=wf_model,
                    wt=wt,
                    boundary=boundary,
                    n_wt=n_wt,
                    XX=XX,
                    YY=YY,
                    spacing_D=spacing_D,
                    seed=seed,
                    random_pct=random_pct,
                    wd=wd_all,
                    ws=ws_mean,
                )

                smart_layout = smart_res["layout_xy"]
                smart_runtime = smart_res.get("runtime_sec", 0.0)
                aep_smart = calc_aep(wf_model, smart_layout, with_wake_loss=True, wd=wd_all, ws=ws_mean)
                aep_smart_true = calc_aep(wf_model, smart_layout, with_wake_loss=True)

                common = {
                    "script": "ResultsSS",
                    "wake_model": wake_model_name,
                    "seed": seed,
                    "wd_step_deg": wd_step_deg,
                    "n_wd": n_wd,
                    "ws": mean_ws,
                    "random_pct": random_pct,
                }

                summary_rows.extend([
                    {
                        **common,
                        "method": "Current Horns Rev",
                        "AEP [GWh]": aep_baseline_true,
                        "Surrogate AEP [GWh]": aep_baseline,
                        "Improvement over Horns Rev [GWh]": 0.0,
                        "runtime_sec": 0.0,
                        "iterations": 0,
                    },
                    {
                        **common,
                        "method": "SmartStart",
                        "AEP [GWh]": aep_smart_true,
                        "Surrogate AEP [GWh]": aep_smart,
                        "Improvement over Horns Rev [GWh]": aep_smart_true - aep_baseline_true,
                        "runtime_sec": smart_runtime,
                        "iterations": 0,
                    },
                ])

                history_rows.append({
                    "run_name": f"SS_seed{seed}",
                    **common,
                    "iteration": 0,
                    "phase": "SmartStart",
                    "AEP [GWh]": aep_smart,
                    "elapsed_sec": smart_runtime,
                })

                if save_layouts:
                    append_layout_rows(
                        layout_rows,
                        horns_rev_layout,
                        wake_model_name,
                        seed,
                        "Current Horns Rev",
                        aep_baseline_true,
                        aep_baseline,
                        0.0,
                        wd_step_deg,
                        n_wd,
                    )
                    append_layout_rows(
                        layout_rows,
                        smart_layout,
                        wake_model_name,
                        seed,
                        "SmartStart",
                        aep_smart_true,
                        aep_smart,
                        smart_runtime,
                        wd_step_deg,
                        n_wd,
                    )

                save_outputs(summary_rows, history_rows, layout_rows, wd_step_deg, timestamp)

    print()
    print("Finished SmartStart-only rerun for 5 and 10 degree bins with explicit random_pct=10.")


if __name__ == "__main__":
    main()
