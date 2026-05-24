import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime
from importlib import import_module
from pathlib import Path

from Windfarm_utils import set_wt, get_site, set_wake_model, calc_aep
from SmartStart import run_smartstart
from GradientBasedAlgorithm import run_gradient_from_layout

run_two_step_from_layout = import_module("2StepAlgorithm").run_two_step_from_layout


# ==================================================
# SETTINGS
# ==================================================

mean_ws = 9.6
wd_all = np.arange(0, 360, 10)
ws_mean = np.array([mean_ws])

max_iterations = 2000  # Max SLSQP iterations for Gradient Based
max_cycles = 10 # Max cycles for Two-step (outer loop)
x_points = 20 # Number of candidate points in x direction for Two-step
y_points = 20 # Number of candidate points in y direction for Two-step
spacing_D = 4 # Minimum spacing in terms of rotor diameters for both Random Search and Two-step
boundary_pad = 400 # Additional padding from the original site boundary to allow for more exploration in Random Search and Two-step

wake_models = ['NOJ']
# wake_models = ['BastankhahGaussian']
# wake_models = ['NOJ', 'BastankhahGaussian']

results_dir = Path("CSV_PL_Comparison")
seeds = [2, 3, 4, 5]
plot_layout_seed = seeds[0]
make_plots = False
save_layouts = True


# ==================================================
# PLOTTING FUNCTIONS
# ==================================================

def plot_layout(layout_xy, boundary, wt, title, spacing_D=4):
    fig, ax = plt.subplots(figsize=(10, 8))

    bx = np.r_[boundary[:, 0], boundary[0, 0]]
    by = np.r_[boundary[:, 1], boundary[0, 1]]
    ax.plot(bx, by, color='black', linewidth=1.5, label='Boundary')

    ax.scatter(
        layout_xy[:, 0],
        layout_xy[:, 1],
        c='red',
        marker='x',
        s=65,
        label='Optimized layout',
        zorder=3
    )

    for x, y in layout_xy:
        circle = Circle(
            (x, y),
            radius=(spacing_D / 2) * wt.diameter(),
            edgecolor='black',
            facecolor='none',
            linestyle='--',
            alpha=0.35,
            zorder=1
        )
        ax.add_patch(circle)

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_layout_with_arrows(old_layout, new_layout, boundary, wt, title, spacing_D=4):
    fig, ax = plt.subplots(figsize=(10, 8))

    bx = np.r_[boundary[:, 0], boundary[0, 0]]
    by = np.r_[boundary[:, 1], boundary[0, 1]]
    ax.plot(bx, by, color='black', linewidth=1.5, label='Boundary', zorder=2)

    for i in range(len(old_layout)):
        x_old, y_old = old_layout[i]
        x_new, y_new = new_layout[i]

        dx = x_new - x_old
        dy = y_new - y_old

        ax.arrow(
            x_old,
            y_old,
            dx,
            dy,
            color='gray',
            alpha=0.28,
            width=4,
            head_width=35,
            head_length=45,
            length_includes_head=True,
            zorder=1
        )

    ax.scatter(
        new_layout[:, 0],
        new_layout[:, 1],
        c='red',
        marker='x',
        s=65,
        label='Optimized layout',
        zorder=4
    )

    for x, y in new_layout:
        circle = Circle(
            (x, y),
            radius=(spacing_D / 2) * wt.diameter(),
            edgecolor='black',
            facecolor='none',
            linestyle='--',
            alpha=0.35,
            zorder=2
        )
        ax.add_patch(circle)

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_aep_history(aep_history, wake_model_name, title_suffix=""):
    df_hist = pd.DataFrame(aep_history)

    fig, ax = plt.subplots(figsize=(10, 6))

    phase_styles = {
        'SmartStart': '--',
        'Two-step': '-',
        'Gradient': '-.'
    }

    for phase in df_hist['phase'].unique():
        df_phase = df_hist[df_hist['phase'] == phase].copy()

        ax.plot(
            df_phase['iteration'],
            df_phase['AEP [GWh]'],
            marker='o',
            linestyle=phase_styles.get(phase, '-'),
            label=phase
        )

    ax.set_title(f"Surrogate AEP development - {wake_model_name} {title_suffix}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Surrogate AEP [GWh]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_aep_time_history(aep_history, wake_model_name, title_suffix=""):
    df_hist = pd.DataFrame(aep_history)

    fig, ax = plt.subplots(figsize=(10, 6))

    phase_styles = {
        'SmartStart': '--',
        'Two-step': '-',
        'Gradient': '-.'
    }

    for phase in df_hist['phase'].unique():
        df_phase = df_hist[df_hist['phase'] == phase].copy()

        ax.plot(
            df_phase['elapsed_sec'],
            df_phase['AEP [GWh]'],
            marker='o',
            linestyle=phase_styles.get(phase, '-'),
            label=phase
        )

    ax.set_title(f"Surrogate AEP development over time - {wake_model_name} {title_suffix}")
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Surrogate AEP [GWh]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def print_run_parameters(
    site_name,
    seed,
    mean_ws,
    wd_all,
    ws_mean,
    max_iterations,
    max_cycles,
    x_points,
    y_points,
    spacing_D,
    boundary_pad,
    wake_models
):
    print()
    print("==================================================")
    print("RUN PARAMETERS")
    print("==================================================")
    print(f"site_name: {site_name}")
    print(f"seed: {seed}")
    print(f"mean_ws: {mean_ws} m/s")
    print(f"ws_mean: {ws_mean.tolist()}")
    print(f"wd_all: {wd_all[0]} to {wd_all[-1]} deg, step {wd_all[1] - wd_all[0]} deg, n={len(wd_all)}")
    print(f"x_points: {x_points}")
    print(f"y_points: {y_points}")
    print(f"max_cycles: {max_cycles}")
    print(f"max_iterations: {max_iterations}")
    print(f"spacing_D: {spacing_D}")
    print(f"boundary_pad: {boundary_pad} m")
    print(f"wake_models: {wake_models}")


def build_combined_history(smart_aep, smart_runtime, two_step_history, full_history):
    combined_history = [{
        'iteration': 0,
        'phase': 'SmartStart',
        'AEP [GWh]': smart_aep,
        'elapsed_sec': smart_runtime
    }]

    for i, row in enumerate(two_step_history[1:], start=1):
        combined_history.append({
            'iteration': i,
            'phase': 'Two-step',
            'AEP [GWh]': row['AEP [GWh]'],
            'elapsed_sec': smart_runtime + row.get('elapsed_sec', 0.0)
        })

    start_iter = len(combined_history)
    full_time_offset = smart_runtime + two_step_history[-1].get('elapsed_sec', 0.0)

    for j, row in enumerate(full_history[1:], start=1):
        combined_history.append({
            'iteration': start_iter + j - 1,
            'phase': 'Gradient',
            'AEP [GWh]': row['AEP [GWh]'],
            'elapsed_sec': full_time_offset + row.get('elapsed_sec', 0.0)
        })

    return combined_history


def append_layout_rows(
    layout_rows,
    layout_xy,
    script,
    wake_model_name,
    seed,
    method,
    aep,
    surrogate_aep,
    runtime_sec
):
    for turbine_id, (x, y) in enumerate(layout_xy):
        layout_rows.append({
            'script': script,
            'wake_model': wake_model_name,
            'seed': seed,
            'method': method,
            'turbine_id': turbine_id,
            'x': x,
            'y': y,
            'AEP [GWh]': aep,
            'Surrogate AEP [GWh]': surrogate_aep,
            'runtime_sec': runtime_sec,
            'wd_step_deg': wd_all[1] - wd_all[0],
            'n_wd': len(wd_all),
            'ws': mean_ws
        })


# ==================================================
# MAIN
# ==================================================

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
    layout_rows = []

    for wake_model_name in wake_models:
        print()
        print("==================================================")
        print(f"Running combined comparison for {wake_model_name}")
        print("==================================================")
        print(f"Using mean wind speed ws = {mean_ws} m/s and 5 degree wind direction bins")
        print("Mean wind speed is used for optimization; site bins are used for reported AEP.")

        wf_model = set_wake_model(wake_model_name, site, wt)

        # ==================================================
        # 0. COMMON BASELINE: CURRENT HORNS REV LAYOUT
        # ==================================================

        aep_baseline = calc_aep(
            wf_model,
            horns_rev_layout,
            with_wake_loss=True,
            wd=wd_all,
            ws=ws_mean
        )
        aep_baseline_true = calc_aep(
            wf_model,
            horns_rev_layout,
            with_wake_loss=True
        )

        print(f"Current Horns Rev surrogate AEP: {aep_baseline:.3f} GWh")
        print(f"Current Horns Rev site-bin AEP: {aep_baseline_true:.3f} GWh")

        for seed in seeds:
            np.random.seed(seed)
            print()
            print("--------------------------------------------------")
            print(f"Seed = {seed}")
            print("--------------------------------------------------")

            # ==================================================
            # 1. SMARTSTART
            # ==================================================

            smart_res = run_smartstart(
                wf_model=wf_model,
                wt=wt,
                boundary=boundary,
                n_wt=n_wt,
                XX=XX,
                YY=YY,
                spacing_D=spacing_D,
                seed=seed,
                wd=wd_all,
                ws=ws_mean
            )

            smart_layout = smart_res['layout_xy']
            aep_smart = calc_aep(wf_model, smart_layout, with_wake_loss=True, wd=wd_all, ws=ws_mean)
            aep_smart_true = calc_aep(wf_model, smart_layout, with_wake_loss=True)

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
                wd=wd_all,
                ws=ws_mean
            )
            two_step_layout = two_step_res['layout_xy']
            aep_two_step = calc_aep(wf_model, two_step_layout, with_wake_loss=True, wd=wd_all, ws=ws_mean)
            aep_two_step_true = calc_aep(wf_model, two_step_layout, with_wake_loss=True)

            grad_res = run_gradient_from_layout(
                wf_model=wf_model,
                wt=wt,
                boundary=boundary,
                n_wt=n_wt,
                initial_layout=smart_layout,
                spacing_D=spacing_D,
                wd=wd_all,
                ws=ws_mean,
                maxiter=max_iterations
            )
            gradient_layout = grad_res['layout_xy']
            aep_gradient = calc_aep(wf_model, gradient_layout, with_wake_loss=True, wd=wd_all, ws=ws_mean)
            aep_gradient_true = calc_aep(wf_model, gradient_layout, with_wake_loss=True)

            full_res = run_gradient_from_layout(
                wf_model=wf_model,
                wt=wt,
                boundary=boundary,
                n_wt=n_wt,
                initial_layout=two_step_layout,
                spacing_D=spacing_D,
                wd=wd_all,
                ws=ws_mean,
                maxiter=max_iterations
            )
            full_layout = full_res['layout_xy']
            aep_full = calc_aep(wf_model, full_layout, with_wake_loss=True, wd=wd_all, ws=ws_mean)
            aep_full_true = calc_aep(wf_model, full_layout, with_wake_loss=True)

            smart_runtime = smart_res.get('runtime_sec', 0.0)
            two_step_runtime = two_step_res['aep_history'][-1].get('elapsed_sec', 0.0)
            grad_runtime = grad_res.get('runtime_sec', 0.0)
            full_grad_runtime = full_res.get('runtime_sec', 0.0)

            rows.extend([
                {
                    'script': 'ResultsSS',
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'method': 'Current Horns Rev',
                    'AEP [GWh]': aep_baseline_true,
                    'Surrogate AEP [GWh]': aep_baseline,
                    'Improvement over Horns Rev [GWh]': 0.0,
                    'runtime_sec': 0.0,
                    'iterations': 0,
                    'wd_step_deg': wd_all[1] - wd_all[0],
                    'n_wd': len(wd_all),
                    'ws': mean_ws
                },
                {
                    'script': 'ResultsSS',
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'method': 'SmartStart',
                    'AEP [GWh]': aep_smart_true,
                    'Surrogate AEP [GWh]': aep_smart,
                    'Improvement over Horns Rev [GWh]': aep_smart_true - aep_baseline_true,
                    'runtime_sec': smart_runtime,
                    'iterations': 0,
                    'wd_step_deg': wd_all[1] - wd_all[0],
                    'n_wd': len(wd_all),
                    'ws': mean_ws
                },
                {
                    'script': 'ResultsSS',
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'method': 'SS--2S',
                    'AEP [GWh]': aep_two_step_true,
                    'Surrogate AEP [GWh]': aep_two_step,
                    'Improvement over Horns Rev [GWh]': aep_two_step_true - aep_baseline_true,
                    'runtime_sec': smart_runtime + two_step_runtime,
                    'iterations': two_step_res['aep_history'][-1]['iteration'],
                    'wd_step_deg': wd_all[1] - wd_all[0],
                    'n_wd': len(wd_all),
                    'ws': mean_ws
                },
                {
                    'script': 'ResultsSS',
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'method': 'SS--GB',
                    'AEP [GWh]': aep_gradient_true,
                    'Surrogate AEP [GWh]': aep_gradient,
                    'Improvement over Horns Rev [GWh]': aep_gradient_true - aep_baseline_true,
                    'runtime_sec': smart_runtime + grad_runtime,
                    'iterations': max_iterations,
                    'wd_step_deg': wd_all[1] - wd_all[0],
                    'n_wd': len(wd_all),
                    'ws': mean_ws
                },
                {
                    'script': 'ResultsSS',
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'method': 'SS--2S--GB',
                    'AEP [GWh]': aep_full_true,
                    'Surrogate AEP [GWh]': aep_full,
                    'Improvement over Horns Rev [GWh]': aep_full_true - aep_baseline_true,
                    'runtime_sec': smart_runtime + two_step_runtime + full_grad_runtime,
                    'iterations': two_step_res['aep_history'][-1]['iteration'] + max_iterations,
                    'wd_step_deg': wd_all[1] - wd_all[0],
                    'n_wd': len(wd_all),
                    'ws': mean_ws
                },
            ])

            if save_layouts:
                append_layout_rows(
                    layout_rows, horns_rev_layout, 'ResultsSS', wake_model_name, seed,
                    'Current Horns Rev', aep_baseline_true, aep_baseline, 0.0
                )
                append_layout_rows(
                    layout_rows, smart_layout, 'ResultsSS', wake_model_name, seed,
                    'SmartStart', aep_smart_true, aep_smart, smart_runtime
                )
                append_layout_rows(
                    layout_rows, two_step_layout, 'ResultsSS', wake_model_name, seed,
                    'SS--2S', aep_two_step_true, aep_two_step, smart_runtime + two_step_runtime
                )
                append_layout_rows(
                    layout_rows, gradient_layout, 'ResultsSS', wake_model_name, seed,
                    'SS--GB', aep_gradient_true, aep_gradient, smart_runtime + grad_runtime
                )
                append_layout_rows(
                    layout_rows, full_layout, 'ResultsSS', wake_model_name, seed,
                    'SS--2S--GB', aep_full_true, aep_full, smart_runtime + two_step_runtime + full_grad_runtime
                )

            combined_history = build_combined_history(
                smart_aep=aep_smart,
                smart_runtime=smart_runtime,
                two_step_history=two_step_res['aep_history'],
                full_history=full_res['aep_history']
            )

            for hist_row in combined_history:
                history_rows.append({
                    'run_name': f"SS_seed{seed}",
                    'script': 'ResultsSS',
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'iteration': hist_row['iteration'],
                    'phase': hist_row['phase'],
                    'AEP [GWh]': hist_row['AEP [GWh]'],
                    'elapsed_sec': hist_row['elapsed_sec'],
                    'wd_step_deg': wd_all[1] - wd_all[0],
                    'n_wd': len(wd_all),
                    'ws': mean_ws
                })

            if make_plots and seed == plot_layout_seed:
                plot_layout(smart_layout, boundary, wt, f"{wake_model_name} - SmartStart", spacing_D)
                plot_layout(two_step_layout, boundary, wt, f"{wake_model_name} - SS--2S", spacing_D)
                plot_layout(gradient_layout, boundary, wt, f"{wake_model_name} - SS--GB", spacing_D)
                plot_layout(full_layout, boundary, wt, f"{wake_model_name} - SS--2S--GB", spacing_D)

                plot_aep_history(combined_history, wake_model_name, f"(SS combined pipeline, seed {seed})")
                plot_aep_time_history(combined_history, wake_model_name, f"(SS combined pipeline, seed {seed})")

    # ==================================================
    # SUMMARY TABLE
    # ==================================================

    df = pd.DataFrame(rows)
    df_history = pd.DataFrame(history_rows)
    df_layouts = pd.DataFrame(layout_rows)
    wd_tag = f"wd{int(wd_all[1] - wd_all[0])}deg"
    summary_output_path = results_dir / f"results_ss_summary_{wd_tag}_{timestamp}.csv"
    history_output_path = results_dir / f"results_ss_history_{wd_tag}_{timestamp}.csv"
    layouts_output_path = results_dir / f"results_ss_layouts_{wd_tag}_{timestamp}.csv"
    df.to_csv(summary_output_path, index=False)
    df_history.to_csv(history_output_path, index=False)
    if save_layouts:
        df_layouts.to_csv(layouts_output_path, index=False)

    df_plot = (
        df.groupby(['wake_model', 'method'], as_index=False)
        .agg(
            **{
                'AEP [GWh]': ('AEP [GWh]', 'mean'),
                'Improvement over Horns Rev [GWh]': ('Improvement over Horns Rev [GWh]', 'mean'),
                'runtime_sec': ('runtime_sec', 'mean'),
                'n_runs': ('seed', 'nunique')
            }
        )
    )

    print()
    print("Saved:", summary_output_path)
    print("Saved:", history_output_path)
    if save_layouts:
        print("Saved:", layouts_output_path)
    print(df)

    if make_plots:
        for wake_model_name in wake_models:
            df_sub = df_plot[df_plot['wake_model'] == wake_model_name]

            plt.figure(figsize=(10, 5))
            plt.bar(df_sub['method'], df_sub['AEP [GWh]'])
            plt.title(f"AEP comparison - {wake_model_name}")
            plt.ylabel("AEP [GWh]")
            plt.xlabel("Pipeline")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.bar(df_sub['method'], df_sub['Improvement over Horns Rev [GWh]'])
            plt.title(f"Improvement over current Horns Rev layout - {wake_model_name}")
            plt.ylabel("Improvement [GWh]")
            plt.xlabel("Pipeline")
            plt.xticks(rotation=20)
            plt.axhline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            plt.show()

    print_run_parameters(
        site_name=site_name,
        seed=seeds,
        mean_ws=mean_ws,
        wd_all=wd_all,
        ws_mean=ws_mean,
        max_iterations=max_iterations,
        max_cycles=max_cycles,
        x_points=x_points,
        y_points=y_points,
        spacing_D=spacing_D,
        boundary_pad=boundary_pad,
        wake_models=wake_models
    )


if __name__ == "__main__":
    main()
