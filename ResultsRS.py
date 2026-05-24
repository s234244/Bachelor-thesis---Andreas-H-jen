import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime
from pathlib import Path

from Windfarm_utilsv3 import set_wt, get_site, set_wake_model, calc_aep
from RandomSearch import run_randomsearch
from SmartStart2StepV4 import run_two_step_from_layout
from SmartstartgradientV2 import run_gradient_from_layout


# ==================================================
# SETTINGS
# ==================================================

mean_ws = 9.6
wd_all = np.arange(0, 360, 5)       # 1 degree wind direction bins
ws_mean = np.array([mean_ws])        # 1 m/s wind speed bins

max_iterations = 200  # Max SLSQP iterations for Gradient Based
max_cycles = 10 # Max cycles for Two-step (outer loop)
x_points = 20 # Number of candidate points in x direction for Two-step
y_points = 20 # Number of candidate points in y direction for Two-step
spacing_D = 4 # Minimum spacing in terms of rotor diameters for both Random Search and Two-step
boundary_pad = 400 # Additional padding from the original site boundary to allow for more exploration in Random Search and Two-step

random_search_layouts = 2000
random_max_time_sec = 24 * 60 * 60
# random_step_schedule_D = [(20, random_search_layouts),]
# To re-enable the coarse-to-fine schedule later, replace random_step_schedule_D with:
random_step_schedule_D = [(20, 1200),(5, 400),(1, 400),]

wake_models = ['NOJ']
# wake_models = ['BastankhahGaussian']
# wake_models = ['NOJ', 'BastankhahGaussian']

results_dir = Path("Results_CSV_PL_Comparison_test")
seeds = [1]
plot_layout_seed = seeds[0]
make_plots = False
save_layouts = True
resume_from_latest = True
checkpoint_after_seed = True

required_methods = {
    'Current Horns Rev',
    'RS',
    'RS--2S',
    'RS--GB',
    'RS--2S--GB'
}


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

    # 80 subtle arrows showing movement
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
        'RandomSearch': '--',
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

    ax.set_title(f"AEP development - {wake_model_name} {title_suffix}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("AEP [GWh]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_aep_time_history(aep_history, wake_model_name, title_suffix=""):
    df_hist = pd.DataFrame(aep_history)

    fig, ax = plt.subplots(figsize=(10, 6))

    phase_styles = {
        'RandomSearch': '--',
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

    ax.set_title(f"AEP development over time - {wake_model_name} {title_suffix}")
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("AEP [GWh]")
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
    random_search_layouts,
    random_step_schedule_D,
    random_max_time_sec,
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
    print(f"random_search_layouts: {random_search_layouts}")
    print(f"random_step_schedule_D: {random_step_schedule_D}")
    print(f"random_max_time_sec: {random_max_time_sec}")
    print(f"wake_models: {wake_models}")


def build_combined_history(stage_histories):
    combined_history = []
    iteration_offset = 0
    time_offset = 0.0

    for stage_idx, stage_history in enumerate(stage_histories):
        if not stage_history:
            continue

        rows_to_add = stage_history if stage_idx == 0 else stage_history[1:]

        for row in rows_to_add:
            combined_history.append({
                'iteration': iteration_offset + row['iteration'],
                'phase': row['phase'],
                'AEP [GWh]': row['AEP [GWh]'],
                'elapsed_sec': time_offset + row.get('elapsed_sec', 0.0)
            })

        iteration_offset += stage_history[-1]['iteration']
        time_offset += stage_history[-1].get('elapsed_sec', 0.0)

    return combined_history


def append_layout_rows(
    layout_rows,
    layout_xy,
    script,
    wake_model_name,
    seed,
    method,
    aep,
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
            'runtime_sec': runtime_sec,
            'random_step_schedule_D': str(random_step_schedule_D),
            'wd_step_deg': wd_all[1] - wd_all[0],
            'n_wd': len(wd_all),
            'ws': mean_ws
        })


def latest_existing_csv(directory, pattern):
    paths = sorted(
        directory.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )
    return paths[0] if paths else None


def save_results(rows, history_rows, layout_rows, summary_path, history_path, layouts_path):
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    if save_layouts:
        pd.DataFrame(layout_rows).to_csv(layouts_path, index=False)


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
        print(f"RandomSearch step schedule = {random_step_schedule_D}")
        print(f"RandomSearch iterations = {random_search_layouts}")

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

        print(f"Current Horns Rev AEP: {aep_baseline:.3f} GWh")

        for seed in seeds:
            np.random.seed(seed)
            print()
            print("--------------------------------------------------")
            print(f"Seed = {seed}")
            print("--------------------------------------------------")

            random_layout = horns_rev_layout.copy()
            random_histories = []
            random_runtime = 0.0

            for stage_idx, (stage_step_D, stage_iterations) in enumerate(random_step_schedule_D, start=1):
                print()
                print(f"RandomSearch stage {stage_idx}: {stage_iterations} iterations, max step {stage_step_D}D")

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
                    wd=wd_all,
                    ws=ws_mean
                )

                random_layout = stage_res['layout_xy']
                random_histories.append(stage_res['aep_history'])
                random_runtime += stage_res['runtime_sec']

            aep_random = calc_aep(wf_model, random_layout, with_wake_loss=True, wd=wd_all, ws=ws_mean)
            print(f"RandomSearch only AEP: {aep_random:.3f} GWh")
            print(f"RandomSearch improvement over Horns Rev: {aep_random - aep_baseline:.3f} GWh")
            print(f"RandomSearch total runtime: {random_runtime:.2f} s")

            two_step_res = run_two_step_from_layout(
                wf_model=wf_model,
                wt=wt,
                boundary=boundary,
                n_wt=n_wt,
                initial_layout=random_layout,
                x_points=site_x_points,
                y_points=site_y_points,
                spacing_D=spacing_D,
                max_cycles=max_cycles,
                wd=wd_all,
                ws=ws_mean
            )
            two_step_layout = two_step_res['layout_xy']
            aep_two_step = calc_aep(wf_model, two_step_layout, with_wake_loss=True, wd=wd_all, ws=ws_mean)

            grad_res = run_gradient_from_layout(
                wf_model=wf_model,
                wt=wt,
                boundary=boundary,
                n_wt=n_wt,
                initial_layout=random_layout,
                spacing_D=spacing_D,
                wd=wd_all,
                ws=ws_mean,
                maxiter=max_iterations
            )
            gradient_layout = grad_res['layout_xy']
            aep_gradient = calc_aep(wf_model, gradient_layout, with_wake_loss=True, wd=wd_all, ws=ws_mean)

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

            two_step_runtime = two_step_res['aep_history'][-1].get('elapsed_sec', 0.0)
            grad_runtime = grad_res.get('runtime_sec', 0.0)
            full_grad_runtime = full_res.get('runtime_sec', 0.0)

            rows.extend([
                {
                    'script': 'ResultsRS',
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'method': 'Current Horns Rev',
                    'AEP [GWh]': aep_baseline,
                    'Improvement over Horns Rev [GWh]': 0.0,
                    'runtime_sec': 0.0,
                    'iterations': 0,
                    'random_step_schedule_D': str(random_step_schedule_D),
                    'wd_step_deg': wd_all[1] - wd_all[0],
                    'n_wd': len(wd_all),
                    'ws': mean_ws
                },
                {
                    'script': 'ResultsRS',
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'method': 'RS',
                    'AEP [GWh]': aep_random,
                    'Improvement over Horns Rev [GWh]': aep_random - aep_baseline,
                    'runtime_sec': random_runtime,
                    'iterations': random_search_layouts,
                    'random_step_schedule_D': str(random_step_schedule_D),
                    'wd_step_deg': wd_all[1] - wd_all[0],
                    'n_wd': len(wd_all),
                    'ws': mean_ws
                },
                {
                    'script': 'ResultsRS',
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'method': 'RS--2S',
                    'AEP [GWh]': aep_two_step,
                    'Improvement over Horns Rev [GWh]': aep_two_step - aep_baseline,
                    'runtime_sec': random_runtime + two_step_runtime,
                    'iterations': random_search_layouts + two_step_res['aep_history'][-1]['iteration'],
                    'random_step_schedule_D': str(random_step_schedule_D),
                    'wd_step_deg': wd_all[1] - wd_all[0],
                    'n_wd': len(wd_all),
                    'ws': mean_ws
                },
                {
                    'script': 'ResultsRS',
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'method': 'RS--GB',
                    'AEP [GWh]': aep_gradient,
                    'Improvement over Horns Rev [GWh]': aep_gradient - aep_baseline,
                    'runtime_sec': random_runtime + grad_runtime,
                    'iterations': random_search_layouts + max_iterations,
                    'random_step_schedule_D': str(random_step_schedule_D),
                    'wd_step_deg': wd_all[1] - wd_all[0],
                    'n_wd': len(wd_all),
                    'ws': mean_ws
                },
                {
                    'script': 'ResultsRS',
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'method': 'RS--2S--GB',
                    'AEP [GWh]': aep_full,
                    'Improvement over Horns Rev [GWh]': aep_full - aep_baseline,
                    'runtime_sec': random_runtime + two_step_runtime + full_grad_runtime,
                    'iterations': random_search_layouts + two_step_res['aep_history'][-1]['iteration'] + max_iterations,
                    'random_step_schedule_D': str(random_step_schedule_D),
                    'wd_step_deg': wd_all[1] - wd_all[0],
                    'n_wd': len(wd_all),
                    'ws': mean_ws
                },
            ])

            if save_layouts:
                append_layout_rows(
                    layout_rows, horns_rev_layout, 'ResultsRS', wake_model_name, seed,
                    'Current Horns Rev', aep_baseline, 0.0
                )
                append_layout_rows(
                    layout_rows, random_layout, 'ResultsRS', wake_model_name, seed,
                    'RS', aep_random, random_runtime
                )
                append_layout_rows(
                    layout_rows, two_step_layout, 'ResultsRS', wake_model_name, seed,
                    'RS--2S', aep_two_step, random_runtime + two_step_runtime
                )
                append_layout_rows(
                    layout_rows, gradient_layout, 'ResultsRS', wake_model_name, seed,
                    'RS--GB', aep_gradient, random_runtime + grad_runtime
                )
                append_layout_rows(
                    layout_rows, full_layout, 'ResultsRS', wake_model_name, seed,
                    'RS--2S--GB', aep_full, random_runtime + two_step_runtime + full_grad_runtime
                )

            combined_history = build_combined_history([
                *random_histories,
                two_step_res['aep_history'],
                full_res['aep_history']
            ])

            for hist_row in combined_history:
                history_rows.append({
                    'run_name': f"RS_seed{seed}",
                    'script': 'ResultsRS',
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'iteration': hist_row['iteration'],
                    'phase': hist_row['phase'],
                    'AEP [GWh]': hist_row['AEP [GWh]'],
                    'elapsed_sec': hist_row['elapsed_sec'],
                    'random_step_schedule_D': str(random_step_schedule_D),
                    'wd_step_deg': wd_all[1] - wd_all[0],
                    'n_wd': len(wd_all),
                    'ws': mean_ws
                })

            if make_plots and seed == plot_layout_seed:
                plot_layout(random_layout, boundary, wt, f"{wake_model_name} - RS", spacing_D)
                plot_layout(two_step_layout, boundary, wt, f"{wake_model_name} - RS--2S", spacing_D)
                plot_layout(gradient_layout, boundary, wt, f"{wake_model_name} - RS--GB", spacing_D)
                plot_layout(full_layout, boundary, wt, f"{wake_model_name} - RS--2S--GB", spacing_D)

                plot_aep_history(combined_history, wake_model_name, f"(RS combined pipeline, seed {seed})")
                plot_aep_time_history(combined_history, wake_model_name, f"(RS combined pipeline, seed {seed})")

    # ==================================================
    # SUMMARY TABLE
    # ==================================================

    df = pd.DataFrame(rows)
    df_history = pd.DataFrame(history_rows)
    df_layouts = pd.DataFrame(layout_rows)
    wd_tag = f"wd{int(wd_all[1] - wd_all[0])}deg"
    summary_output_path = results_dir / f"results_rs_summary_{wd_tag}_{timestamp}.csv"
    history_output_path = results_dir / f"results_rs_history_{wd_tag}_{timestamp}.csv"
    layouts_output_path = results_dir / f"results_rs_layouts_{wd_tag}_{timestamp}.csv"
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
        random_search_layouts=random_search_layouts,
        random_step_schedule_D=random_step_schedule_D,
        random_max_time_sec=random_max_time_sec,
        wake_models=wake_models
    )


if __name__ == "__main__":
    main()
