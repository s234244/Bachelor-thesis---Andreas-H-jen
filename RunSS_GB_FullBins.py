import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from Windfarm_utilsv3 import set_wt, get_site, set_wake_model, calc_aep
from SmartstartV2 import run_smartstart
from SmartstartgradientV2 import run_gradient_from_layout


# ==================================================
# SETTINGS
# ==================================================

site_name = "HornsRev1"
mean_ws = 9.6

# Set True if SmartStart evaluation should also use the expensive full bin setup.
# Set False to test whether Gradient Based improves when only GB uses full bins.
run_smartstart_with_full_bins = True

ss_wd = np.arange(0, 360, 1) if run_smartstart_with_full_bins else np.arange(0, 360, 10)
ss_ws = np.arange(3, 26, 1) if run_smartstart_with_full_bins else np.array([mean_ws])

gb_wd = np.arange(0, 360, 1)
gb_ws = np.arange(3, 26, 1)

max_gradient_iterations = 200
spacing_D = 4
boundary_pad = 400
x_points = 20
y_points = 20
random_pct = 10

wake_models = ["NOJ"]
seeds = [1]

results_dir = Path("Results_CSV_SS_GB_FullBins")
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


def append_layout_rows(rows, layout_xy, script, wake_model, seed, method, aep_ss_bins, aep_gb_bins, runtime_sec):
    for turbine_id, (x, y) in enumerate(layout_xy):
        rows.append({
            "script": script,
            "wake_model": wake_model,
            "seed": seed,
            "method": method,
            "turbine_id": turbine_id,
            "x": x,
            "y": y,
            "AEP SS bins [GWh]": aep_ss_bins,
            "AEP GB bins [GWh]": aep_gb_bins,
            "runtime_sec": runtime_sec,
            "ss_bin_setup": bin_description(ss_wd, ss_ws),
            "gb_bin_setup": bin_description(gb_wd, gb_ws),
            "random_pct": random_pct,
        })


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

    summary_rows = []
    history_rows = []
    layout_rows = []

    print()
    print("==================================================")
    print("SS--GB full-bin Gradient test")
    print("==================================================")
    print(f"SmartStart bins: {bin_description(ss_wd, ss_ws)}")
    print(f"Gradient bins:   {bin_description(gb_wd, gb_ws)}")
    print(f"SmartStart full bins enabled: {run_smartstart_with_full_bins}")
    print(f"SmartStart random_pct: {random_pct}")
    print(f"Seeds: {seeds}")
    print()

    for wake_model_name in wake_models:
        wf_model = set_wake_model(wake_model_name, site, wt)

        baseline_ss = calc_aep(wf_model, horns_rev_layout, with_wake_loss=True, wd=ss_wd, ws=ss_ws)
        baseline_gb = calc_aep(wf_model, horns_rev_layout, with_wake_loss=True, wd=gb_wd, ws=gb_ws)

        for seed in seeds:
            print()
            print("--------------------------------------------------")
            print(f"Wake model: {wake_model_name}, seed: {seed}")
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
                wd=ss_wd,
                ws=ss_ws,
            )

            smart_layout = smart_res["layout_xy"]
            smart_runtime = smart_res.get("runtime_sec", 0.0)

            aep_ss_ss_bins = calc_aep(wf_model, smart_layout, with_wake_loss=True, wd=ss_wd, ws=ss_ws)
            aep_ss_gb_bins = calc_aep(wf_model, smart_layout, with_wake_loss=True, wd=gb_wd, ws=gb_ws)

            print()
            print(f"SS layout AEP with SS bins: {aep_ss_ss_bins:.3f} GWh")
            print(f"SS layout AEP with GB/full bins: {aep_ss_gb_bins:.3f} GWh")
            print(f"SmartStart runtime: {smart_runtime:.2f} s")

            grad_res = run_gradient_from_layout(
                wf_model=wf_model,
                wt=wt,
                boundary=boundary,
                n_wt=n_wt,
                initial_layout=smart_layout,
                spacing_D=spacing_D,
                wd=gb_wd,
                ws=gb_ws,
                maxiter=max_gradient_iterations,
            )

            gradient_layout = grad_res["layout_xy"]
            gradient_runtime = grad_res.get("runtime_sec", 0.0)

            aep_gb_ss_bins = calc_aep(wf_model, gradient_layout, with_wake_loss=True, wd=ss_wd, ws=ss_ws)
            aep_gb_gb_bins = calc_aep(wf_model, gradient_layout, with_wake_loss=True, wd=gb_wd, ws=gb_ws)

            print()
            print(f"SS--GB layout AEP with SS bins: {aep_gb_ss_bins:.3f} GWh")
            print(f"SS--GB layout AEP with GB/full bins: {aep_gb_gb_bins:.3f} GWh")
            print(f"GB improvement using full bins: {aep_gb_gb_bins - aep_ss_gb_bins:.3f} GWh")
            print(f"Total SS--GB runtime: {smart_runtime + gradient_runtime:.2f} s")

            summary_rows.extend([
                {
                    "script": "RunSS_GB_FullBins",
                    "wake_model": wake_model_name,
                    "seed": seed,
                    "method": "Current Horns Rev",
                    "AEP SS bins [GWh]": baseline_ss,
                    "AEP GB bins [GWh]": baseline_gb,
                    "Improvement over SS layout using GB bins [GWh]": np.nan,
                    "runtime_sec": 0.0,
                    "iterations": 0,
                    "ss_bin_setup": bin_description(ss_wd, ss_ws),
                    "gb_bin_setup": bin_description(gb_wd, gb_ws),
                    "random_pct": random_pct,
                },
                {
                    "script": "RunSS_GB_FullBins",
                    "wake_model": wake_model_name,
                    "seed": seed,
                    "method": "SS",
                    "AEP SS bins [GWh]": aep_ss_ss_bins,
                    "AEP GB bins [GWh]": aep_ss_gb_bins,
                    "Improvement over SS layout using GB bins [GWh]": 0.0,
                    "runtime_sec": smart_runtime,
                    "iterations": 0,
                    "ss_bin_setup": bin_description(ss_wd, ss_ws),
                    "gb_bin_setup": bin_description(gb_wd, gb_ws),
                    "random_pct": random_pct,
                },
                {
                    "script": "RunSS_GB_FullBins",
                    "wake_model": wake_model_name,
                    "seed": seed,
                    "method": "SS--GB",
                    "AEP SS bins [GWh]": aep_gb_ss_bins,
                    "AEP GB bins [GWh]": aep_gb_gb_bins,
                    "Improvement over SS layout using GB bins [GWh]": aep_gb_gb_bins - aep_ss_gb_bins,
                    "runtime_sec": smart_runtime + gradient_runtime,
                    "iterations": max_gradient_iterations,
                    "ss_bin_setup": bin_description(ss_wd, ss_ws),
                    "gb_bin_setup": bin_description(gb_wd, gb_ws),
                    "random_pct": random_pct,
                },
            ])

            history_rows.append({
                "run_name": f"SS_GB_seed{seed}",
                "script": "RunSS_GB_FullBins",
                "wake_model": wake_model_name,
                "seed": seed,
                "iteration": 0,
                "phase": "SmartStart",
                "AEP [GWh]": aep_ss_ss_bins,
                "elapsed_sec": smart_runtime,
                "aep_bin_source": "SS bins",
                "ss_bin_setup": bin_description(ss_wd, ss_ws),
                "gb_bin_setup": bin_description(gb_wd, gb_ws),
                "random_pct": random_pct,
            })

            for row in grad_res["aep_history"]:
                history_rows.append({
                    "run_name": f"SS_GB_seed{seed}",
                    "script": "RunSS_GB_FullBins",
                    "wake_model": wake_model_name,
                    "seed": seed,
                    "iteration": row["iteration"],
                    "phase": "Gradient",
                    "AEP [GWh]": row["AEP [GWh]"],
                    "elapsed_sec": smart_runtime + row.get("elapsed_sec", 0.0),
                    "aep_bin_source": "GB bins",
                    "ss_bin_setup": bin_description(ss_wd, ss_ws),
                    "gb_bin_setup": bin_description(gb_wd, gb_ws),
                    "random_pct": random_pct,
                })

            if save_layouts:
                append_layout_rows(
                    layout_rows,
                    horns_rev_layout,
                    "RunSS_GB_FullBins",
                    wake_model_name,
                    seed,
                    "Current Horns Rev",
                    baseline_ss,
                    baseline_gb,
                    0.0,
                )
                append_layout_rows(
                    layout_rows,
                    smart_layout,
                    "RunSS_GB_FullBins",
                    wake_model_name,
                    seed,
                    "SS",
                    aep_ss_ss_bins,
                    aep_ss_gb_bins,
                    smart_runtime,
                )
                append_layout_rows(
                    layout_rows,
                    gradient_layout,
                    "RunSS_GB_FullBins",
                    wake_model_name,
                    seed,
                    "SS--GB",
                    aep_gb_ss_bins,
                    aep_gb_gb_bins,
                    smart_runtime + gradient_runtime,
                )

    summary_path = results_dir / f"ss_gb_fullbins_summary_{timestamp}.csv"
    history_path = results_dir / f"ss_gb_fullbins_history_{timestamp}.csv"
    layouts_path = results_dir / f"ss_gb_fullbins_layouts_{timestamp}.csv"

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
