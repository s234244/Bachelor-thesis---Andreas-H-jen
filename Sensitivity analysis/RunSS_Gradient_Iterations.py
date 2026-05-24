import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from Windfarm_utilsv3 import set_wt, get_site, set_wake_model, calc_aep
from SmartstartV2 import run_smartstart
from SmartstartgradientV2 import run_gradient_from_layout


mean_ws = 9.6
wd_all = np.arange(0, 360, 10)
ws_mean = np.array([mean_ws])

x_points = 20
y_points = 20
spacing_D = 4
boundary_pad = 400

gradient_iterations_list = [10, 20, 50, 100, 200, 500]

wake_models = ['NOJ']
results_dir = Path("Results_Gradient_CSV")
seeds = [1, 2, 3]
smartstart_random_pct = 10

# False gives a fair comparison of gradient max iterations because every
# maxiter value starts from the same SmartStart layout for a given seed.
# True reruns SmartStart with a derived seed for each maxiter value, so the
# gradient runs start from different layouts.
different_smartstart_layout_per_gradient_setting = False
smartstart_seed_offset = 10000


def print_run_parameters(
    site_name,
    seeds,
    mean_ws,
    wd_all,
    ws_mean,
    x_points,
    y_points,
    spacing_D,
    boundary_pad,
    gradient_iterations_list,
    wake_models
):
    print()
    print("==================================================")
    print("RUN PARAMETERS")
    print("==================================================")
    print(f"site_name: {site_name}")
    print(f"seeds: {seeds}")
    print(f"mean_ws: {mean_ws} m/s")
    print(f"ws_mean: {ws_mean.tolist()}")
    print(f"wd_all: {wd_all[0]} to {wd_all[-1]} deg, step {wd_all[1] - wd_all[0]} deg, n={len(wd_all)}")
    print(f"x_points: {x_points}")
    print(f"y_points: {y_points}")
    print(f"spacing_D: {spacing_D}")
    print(f"boundary_pad: {boundary_pad} m")
    print(f"gradient_iterations_list: {gradient_iterations_list}")
    print(f"smartstart_random_pct: {smartstart_random_pct}")
    print(f"different_smartstart_layout_per_gradient_setting: {different_smartstart_layout_per_gradient_setting}")
    print(f"wake_models: {wake_models}")


def smartstart_seed_for(seed, gradient_iterations=None):
    if gradient_iterations is None or not different_smartstart_layout_per_gradient_setting:
        return seed

    return seed * smartstart_seed_offset + int(gradient_iterations)


def main():
    site_name = 'HornsRev1'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(exist_ok=True)

    wt, rated_power = set_wt(site_name)
    boundary, x_ref, y_ref, site, n_wt, site_x_points, site_y_points, XX, YY, candidate_points = get_site(
        site_name=site_name,
        wt=wt,
        boundary_pad=boundary_pad,
        x_points=x_points,
        y_points=y_points
    )

    horns_rev_layout = np.column_stack([x_ref, y_ref])
    rows = []
    history_rows = []

    for wake_model_name in wake_models:
        print()
        print("==================================================")
        print(f"Wake model: {wake_model_name}")
        print("==================================================")

        wf_model = set_wake_model(wake_model_name, site, wt)

        aep_baseline = calc_aep(
            wf_model,
            horns_rev_layout,
            with_wake_loss=True,
            wd=wd_all,
            ws=ws_mean
        )

        print(f"Current Horns Rev AEP: {aep_baseline:.3f} GWh")

        for seed in seeds:
            np.random.seed(seed)

            print()
            print("--------------------------------------------------")
            print(f"Running SmartStart for {wake_model_name}")
            print(f"Seed = {seed}")
            print("--------------------------------------------------")

            smart_res = run_smartstart(
                wf_model=wf_model,
                wt=wt,
                boundary=boundary,
                n_wt=n_wt,
                XX=XX,
                YY=YY,
                spacing_D=spacing_D,
                seed=smartstart_seed_for(seed),
                random_pct=smartstart_random_pct,
                wd=wd_all,
                ws=ws_mean
            )

            smart_layout = smart_res['layout_xy']
            aep_smart = calc_aep(
                wf_model,
                smart_layout,
                with_wake_loss=True,
                wd=wd_all,
                ws=ws_mean
            )

            rows.append({
                'wake_model': wake_model_name,
                'seed': seed,
                'gradient_iterations': 0,
                'method': 'SmartStart',
                'AEP [GWh]': aep_smart,
                'Improvement over Horns Rev [GWh]': aep_smart - aep_baseline,
                'Improvement over SmartStart [GWh]': 0.0,
                'runtime_sec': smart_res.get('runtime_sec', 0.0),
                'recorded_gradient_iterations': 0,
                'smartstart_seed': smartstart_seed_for(seed),
                'smartstart_random_pct': smartstart_random_pct
            })

            for gradient_iterations in gradient_iterations_list:
                if different_smartstart_layout_per_gradient_setting:
                    gradient_smartstart_seed = smartstart_seed_for(seed, gradient_iterations)

                    print()
                    print("--------------------------------------------------")
                    print(f"Rerunning SmartStart for gradient_iterations = {gradient_iterations}")
                    print(f"Base seed = {seed}, SmartStart seed = {gradient_smartstart_seed}")
                    print("--------------------------------------------------")

                    grad_smart_res = run_smartstart(
                        wf_model=wf_model,
                        wt=wt,
                        boundary=boundary,
                        n_wt=n_wt,
                        XX=XX,
                        YY=YY,
                        spacing_D=spacing_D,
                        seed=gradient_smartstart_seed,
                        random_pct=smartstart_random_pct,
                        wd=wd_all,
                        ws=ws_mean
                    )

                    gradient_initial_layout = grad_smart_res['layout_xy']
                    gradient_initial_aep = calc_aep(
                        wf_model,
                        gradient_initial_layout,
                        with_wake_loss=True,
                        wd=wd_all,
                        ws=ws_mean
                    )
                else:
                    gradient_smartstart_seed = smartstart_seed_for(seed)
                    gradient_initial_layout = smart_layout
                    gradient_initial_aep = aep_smart

                print()
                print("--------------------------------------------------")
                print(f"Running SS-Gradient for {wake_model_name}")
                print(f"Seed = {seed}")
                print(f"SmartStart seed = {gradient_smartstart_seed}")
                print(f"Gradient max iterations = {gradient_iterations}")
                print("--------------------------------------------------")

                grad_res = run_gradient_from_layout(
                    wf_model=wf_model,
                    wt=wt,
                    boundary=boundary,
                    n_wt=n_wt,
                    initial_layout=gradient_initial_layout,
                    spacing_D=spacing_D,
                    wd=wd_all,
                    ws=ws_mean,
                    maxiter=gradient_iterations
                )

                gradient_layout = grad_res['layout_xy']
                aep_gradient = calc_aep(
                    wf_model,
                    gradient_layout,
                    with_wake_loss=True,
                    wd=wd_all,
                    ws=ws_mean
                )

                improvement_baseline = aep_gradient - aep_baseline
                improvement_smartstart = aep_gradient - gradient_initial_aep
                recorded_gradient_iterations = max(len(grad_res['aep_history']) - 1, 0)

                print(f"SS-Gradient AEP: {aep_gradient:.3f} GWh")
                print(f"Improvement over SmartStart: {improvement_smartstart:.3f} GWh")
                print(f"Improvement over Horns Rev: {improvement_baseline:.3f} GWh")

                rows.append({
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'gradient_iterations': gradient_iterations,
                    'method': 'SS-Gradient',
                    'AEP [GWh]': aep_gradient,
                    'Improvement over Horns Rev [GWh]': improvement_baseline,
                    'Improvement over SmartStart [GWh]': improvement_smartstart,
                    'runtime_sec': grad_res['runtime_sec'],
                    'recorded_gradient_iterations': recorded_gradient_iterations,
                    'smartstart_seed': gradient_smartstart_seed,
                    'smartstart_random_pct': smartstart_random_pct
                })

                for hist_row in grad_res['aep_history']:
                    history_rows.append({
                        'run_name': f"gradient_iterations_{gradient_iterations}_seed{seed}",
                        'script': 'RunSS_Gradient_Iterations',
                        'wake_model': wake_model_name,
                        'seed': seed,
                        'smartstart_seed': gradient_smartstart_seed,
                        'smartstart_random_pct': smartstart_random_pct,
                        'gradient_iterations_setting': gradient_iterations,
                        'iteration': hist_row['iteration'],
                        'phase': hist_row['phase'],
                        'AEP [GWh]': hist_row['AEP [GWh]'],
                        'elapsed_sec': hist_row.get('elapsed_sec', 0.0),
                        'timing_source': 'interpolated_from_total_runtime'
                    })

    df = pd.DataFrame(rows)
    df_history = pd.DataFrame(history_rows)

    output_path = results_dir / f"ss_gradient_iterations_all_{timestamp}.csv"

    df_all = pd.concat(
        [
            df.assign(row_type='summary'),
            df_history.assign(row_type='history')
        ],
        ignore_index=True,
        sort=False
    )
    df_all.to_csv(output_path, index=False)

    print()
    print("Saved:", output_path)

    print()
    print("==================================================")
    print("SUMMARY RESULTS")
    print("==================================================")
    print(df)

    print_run_parameters(
        site_name=site_name,
        seeds=seeds,
        mean_ws=mean_ws,
        wd_all=wd_all,
        ws_mean=ws_mean,
        x_points=x_points,
        y_points=y_points,
        spacing_D=spacing_D,
        boundary_pad=boundary_pad,
        gradient_iterations_list=gradient_iterations_list,
        wake_models=wake_models
    )


if __name__ == "__main__":
    main()
