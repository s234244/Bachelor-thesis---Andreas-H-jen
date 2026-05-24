import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime
from pathlib import Path

from Windfarm_utilsv3 import set_wt, get_site, set_wake_model, calc_aep
from RandomSearch import run_randomsearch

# ==================================================
# SETTINGS
# ==================================================

mean_ws = 9.6
wd_all = np.arange(0, 360, 5)
ws_mean = np.array([mean_ws])

x_points = 20
y_points = 20
spacing_D = 4
boundary_pad = 400

# Fixed RandomSearch iterations
random_search_layouts = 2000
random_max_time_sec = 24 * 60 * 60 

# Different max step sizes to test
random_max_step_D_list = [1, 5, 10, 20, 30, 50]

# Coarse-to-fine schedule: starts with large layout moves, then refines with smaller moves.
# Total iterations match random_search_layouts, so it is comparable to the single-step runs.
staged_step_schedules = [
    {
        "name": "coarse_to_fine_20_5_1D",
        "stages": [
            (20, 1200),
            (5, 400),
            (1, 400),
        ],
    }
]

wake_models = ['NOJ']
results_dir = Path("Results_RS_CSV")
seeds = [1, 2, 3]
make_plots = False


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


def plot_aep_history(aep_history, wake_model_name, title_suffix="", final_aep=None):
    df_hist = pd.DataFrame(aep_history)
    if final_aep is None:
        final_aep = df_hist['AEP [GWh]'].iloc[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(
        df_hist['iteration'],
        df_hist['AEP [GWh]'],
        marker='o',
        linestyle='--',
        label='RandomSearch'
    )

    plt.title(
        f"{wake_model_name} - RandomSearch {title_suffix} - "
        f"{random_search_layouts} iterations - AEP {final_aep:.3f} GWh"
    )
    plt.xlabel("n evaluations")
    plt.ylabel("AEP [GWh]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_aep_time_history(aep_history, wake_model_name, title_suffix="", final_aep=None):
    df_hist = pd.DataFrame(aep_history)
    if final_aep is None:
        final_aep = df_hist['AEP [GWh]'].iloc[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(
        df_hist['elapsed_sec'],
        df_hist['AEP [GWh]'],
        marker='o',
        linestyle='--',
        label='RandomSearch'
    )

    plt.title(
        f"{wake_model_name} - RandomSearch {title_suffix} - "
        f"{random_search_layouts} iterations - AEP {final_aep:.3f} GWh"
    )
    plt.xlabel("Elapsed time [s]")
    plt.ylabel("AEP [GWh]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


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

    for wake_model_name in wake_models:

        print()
        print("==================================================")
        print(f"Wake model: {wake_model_name}")
        print("==================================================")

        wf_model = set_wake_model(wake_model_name, site, wt)

        # ==================================================
        # BASELINE: CURRENT HORNS REV LAYOUT
        # ==================================================

        aep_baseline = calc_aep(
            wf_model,
            horns_rev_layout,
            with_wake_loss=True,
            wd=wd_all,
            ws=ws_mean
        )

        print(f"Current Horns Rev AEP: {aep_baseline:.3f} GWh")

        # ==================================================
        # RANDOM SEARCH FOR DIFFERENT STEP SIZES
        # ==================================================

        for seed in seeds:
            np.random.seed(seed)

            rows.append({
                'wake_model': wake_model_name,
                'seed': seed,
                'iterations': random_search_layouts,
                'max_step_D': 0,
                'method': 'Current Horns Rev',
                'AEP [GWh]': aep_baseline,
                'Improvement over Horns Rev [GWh]': 0.0
            })

            for random_max_step_D in random_max_step_D_list:

                print()
                print("--------------------------------------------------")
                print(f"Running RandomSearch for {wake_model_name}")
                print(f"Seed = {seed}")
                print(f"Iterations = {random_search_layouts}")
                print(f"Max step size = {random_max_step_D}D")
                print("--------------------------------------------------")

                random_res = run_randomsearch(
                    wf_model=wf_model,
                    wt=wt,
                    boundary=boundary,
                    n_wt=n_wt,
                    initial_layout=horns_rev_layout,
                    spacing_D=spacing_D,
                    max_iter=random_search_layouts,
                    max_time=random_max_time_sec,
                    max_step=random_max_step_D * wt.diameter(),
                    seed=seed,
                    wd=wd_all,
                    ws=ws_mean
                )

                random_layout = random_res['layout_xy']

                aep_random = calc_aep(
                    wf_model,
                    random_layout,
                    with_wake_loss=True,
                    wd=wd_all,
                    ws=ws_mean
                )

                improvement = aep_random - aep_baseline

                print(f"RandomSearch AEP: {aep_random:.3f} GWh")
                print(f"Improvement over Horns Rev: {improvement:.3f} GWh")

                rows.append({
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'iterations': random_search_layouts,
                    'max_step_D': random_max_step_D,
                    'method': 'RandomSearch',
                    'AEP [GWh]': aep_random,
                    'Improvement over Horns Rev [GWh]': improvement,
                    'runtime_sec': random_res['runtime_sec']
                })

                for hist_row in random_res['aep_history']:
                    history_rows.append({
                        'run_name': f"max_step_{random_max_step_D}D_seed{seed}",
                        'script': 'RunRS_StepSize',
                        'wake_model': wake_model_name,
                        'seed': seed,
                        'iterations_setting': random_search_layouts,
                        'max_step_D': random_max_step_D,
                        'iteration': hist_row['iteration'],
                        'phase': hist_row['phase'],
                        'AEP [GWh]': hist_row['AEP [GWh]'],
                        'elapsed_sec': hist_row['elapsed_sec'],
                        'timing_source': 'interpolated_from_total_runtime'
                    })

                # ==================================================
                # PLOTS
                # ==================================================

                if make_plots:
                    random_plot_title = (
                        f"{wake_model_name} - RandomSearch - "
                        f"{random_search_layouts} iterations - "
                        f"max step {random_max_step_D}D - "
                        f"AEP {aep_random:.3f} GWh"
                    )

                    plot_layout(
                        random_layout,
                        boundary,
                        wt,
                        random_plot_title,
                        spacing_D
                    )

                    plot_aep_history(
                        random_res['aep_history'],
                        wake_model_name,
                        f"(max step {random_max_step_D}D)",
                        final_aep=aep_random
                    )

                    plot_aep_time_history(
                        random_res['aep_history'],
                        wake_model_name,
                        f"(max step {random_max_step_D}D)",
                        final_aep=aep_random
                    )

            # ==================================================
            # STAGED RANDOM SEARCH: LARGE STEPS THEN SMALL STEPS
            # ==================================================

            for schedule in staged_step_schedules:
                schedule_name = schedule["name"]
                stages = schedule["stages"]
                total_stage_iterations = sum(stage_iterations for _, stage_iterations in stages)

                print()
                print("--------------------------------------------------")
                print(f"Running staged RandomSearch for {wake_model_name}")
                print(f"Seed = {seed}")
                print(f"Schedule = {schedule_name}")
                print(f"Total iterations = {total_stage_iterations}")
                print("--------------------------------------------------")

                staged_layout = horns_rev_layout.copy()
                staged_runtime = 0.0
                cumulative_iterations = 0.0
                cumulative_elapsed_sec = 0.0
                staged_aep = aep_baseline

                for stage_index, (stage_step_D, stage_iterations) in enumerate(stages, start=1):
                    print()
                    print(f"Stage {stage_index}: {stage_iterations} iterations, max step {stage_step_D}D")

                    stage_res = run_randomsearch(
                        wf_model=wf_model,
                        wt=wt,
                        boundary=boundary,
                        n_wt=n_wt,
                        initial_layout=staged_layout,
                        spacing_D=spacing_D,
                        max_iter=stage_iterations,
                        max_time=random_max_time_sec,
                        max_step=stage_step_D * wt.diameter(),
                        seed=seed + 1000 * stage_index,
                        wd=wd_all,
                        ws=ws_mean
                    )

                    staged_layout = stage_res['layout_xy']
                    staged_runtime += stage_res['runtime_sec']
                    staged_aep = calc_aep(
                        wf_model,
                        staged_layout,
                        with_wake_loss=True,
                        wd=wd_all,
                        ws=ws_mean
                    )

                    stage_history = stage_res['aep_history']
                    max_recorded_iteration = max(row['iteration'] for row in stage_history)

                    for hist_row in stage_history:
                        if max_recorded_iteration > 0:
                            requested_iteration = (
                                cumulative_iterations
                                + hist_row['iteration'] / max_recorded_iteration * stage_iterations
                            )
                        else:
                            requested_iteration = cumulative_iterations

                        history_rows.append({
                            'run_name': f"{schedule_name}_seed{seed}",
                            'script': 'RunRS_StepSize',
                            'wake_model': wake_model_name,
                            'seed': seed,
                            'iterations_setting': total_stage_iterations,
                            'max_step_D': stage_step_D,
                            'step_schedule': schedule_name,
                            'stage_index': stage_index,
                            'stage_iterations': stage_iterations,
                            'iteration': requested_iteration,
                            'phase': 'RandomSearch staged',
                            'AEP [GWh]': hist_row['AEP [GWh]'],
                            'elapsed_sec': cumulative_elapsed_sec + hist_row['elapsed_sec'],
                            'timing_source': 'scaled_to_requested_stage_iterations'
                        })

                    cumulative_iterations += stage_iterations
                    cumulative_elapsed_sec += stage_res['runtime_sec']

                    print(f"Stage {stage_index} AEP: {staged_aep:.3f} GWh")

                staged_improvement = staged_aep - aep_baseline

                print(f"Staged RandomSearch AEP: {staged_aep:.3f} GWh")
                print(f"Improvement over Horns Rev: {staged_improvement:.3f} GWh")

                rows.append({
                    'wake_model': wake_model_name,
                    'seed': seed,
                    'iterations': total_stage_iterations,
                    'max_step_D': np.nan,
                    'step_schedule': schedule_name,
                    'method': 'RandomSearch staged',
                    'AEP [GWh]': staged_aep,
                    'Improvement over Horns Rev [GWh]': staged_improvement,
                    'runtime_sec': staged_runtime
                })

                if make_plots:
                    staged_plot_title = (
                        f"{wake_model_name} - Staged RandomSearch - "
                        f"{schedule_name} - AEP {staged_aep:.3f} GWh"
                    )
                    plot_layout(
                        staged_layout,
                        boundary,
                        wt,
                        staged_plot_title,
                        spacing_D
                    )

    # ==================================================
    # SUMMARY TABLE
    # ==================================================

    df = pd.DataFrame(rows)
    df_history = pd.DataFrame(history_rows)

    summary_output_path = results_dir / f"rs_stepsize_summary_{timestamp}.csv"
    history_output_path = results_dir / f"rs_stepsize_history_{timestamp}.csv"
    df.to_csv(summary_output_path, index=False)
    df_history.to_csv(history_output_path, index=False)

    print()
    print("Saved:", summary_output_path)
    print("Saved:", history_output_path)

    print()
    print("==================================================")
    print("SUMMARY RESULTS")
    print("==================================================")
    print(df)

    # ==================================================
    # SUMMARY PLOTS
    # ==================================================

    if make_plots:
        for wake_model_name in wake_models:

            df_sub = df[
                (df['wake_model'] == wake_model_name) &
                (df['method'] == 'RandomSearch')
            ]

            # ----------------------------------------------
            # AEP vs max step size
            # ----------------------------------------------

            plt.figure(figsize=(9, 5))

            plt.plot(
                df_sub['max_step_D'],
                df_sub['AEP [GWh]'],
                marker='o',
                linewidth=2
            )

            plt.title(f"RandomSearch AEP vs max step size - {wake_model_name}")
            plt.xlabel("Maximum random step size [D]")
            plt.ylabel("AEP [GWh]")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            # ----------------------------------------------
            # Improvement vs max step size
            # ----------------------------------------------

            plt.figure(figsize=(9, 5))

            plt.plot(
                df_sub['max_step_D'],
                df_sub['Improvement over Horns Rev [GWh]'],
                marker='o',
                linewidth=2
            )

            plt.title(f"RandomSearch improvement vs max step size - {wake_model_name}")
            plt.xlabel("Maximum random step size [D]")
            plt.ylabel("Improvement over Horns Rev [GWh]")
            plt.grid(True, alpha=0.3)
            plt.axhline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            plt.show()

            # ----------------------------------------------
            # Runtime vs iterations
            # ----------------------------------------------

            plt.figure(figsize=(9, 5))

            plt.scatter(
                [random_search_layouts] * len(df_sub),
                df_sub['runtime_sec'],
                s=70
            )

            for _, row in df_sub.iterrows():
                plt.annotate(
                    f"{row['max_step_D']}D",
                    (random_search_layouts, row['runtime_sec']),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=9
                )

            plt.title(f"RandomSearch runtime vs iterations - {wake_model_name}")
            plt.xlabel("RandomSearch iterations")
            plt.ylabel("Runtime [s]")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    # ==================================================
    # PRINT PARAMETERS
    # ==================================================

    print()
    print("==================================================")
    print("RUN PARAMETERS")
    print("==================================================")

    print(f"site_name: {site_name}")
    print(f"seeds: {seeds}")
    print(f"mean_ws: {mean_ws} m/s")
    print(f"ws_mean: {ws_mean.tolist()}")
    print(f"wd_all: {wd_all[0]} to {wd_all[-1]} deg")
    print(f"x_points: {x_points}")
    print(f"y_points: {y_points}")
    print(f"spacing_D: {spacing_D}")
    print(f"boundary_pad: {boundary_pad}")
    print(f"random_search_layouts: {random_search_layouts}")
    print(f"random_max_time_sec: {random_max_time_sec}")
    print(f"random_max_step_D_list: {random_max_step_D_list}")
    print(f"staged_step_schedules: {staged_step_schedules}")
    print(f"wake_models: {wake_models}")


if __name__ == "__main__":
    main()
