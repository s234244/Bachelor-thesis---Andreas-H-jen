import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime
from pathlib import Path

from Windfarm_utils import set_wt, get_site, set_wake_model, calc_aep
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

random_search_iterations_list = [10000]
random_max_step_D = 10
random_max_time_sec = 24 * 60 * 60

wake_models = ['NOJ']
results_dir = Path("CSV_RS")
seeds = [1, 2, 3, 4, 5, 6]
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
        f"AEP {final_aep:.3f} GWh"
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
        f"AEP {final_aep:.3f} GWh"
    )
    plt.xlabel("Elapsed time [s]")
    plt.ylabel("AEP [GWh]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_run_parameters(
    site_name,
    seed,
    mean_ws,
    wd_all,
    ws_mean,
    x_points,
    y_points,
    spacing_D,
    boundary_pad,
    random_search_layouts,
    random_max_step_D,
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
    print(f"spacing_D: {spacing_D}")
    print(f"boundary_pad: {boundary_pad} m")
    print(f"random_search_layouts: {random_search_layouts}")
    print(f"random_max_step_D: {random_max_step_D}")
    print(f"random_max_time_sec: {random_max_time_sec}")
    print(f"wake_models: {wake_models}")


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
        # RANDOM SEARCH FOR MULTIPLE ITERATION SETTINGS
        # ==================================================

        for seed in seeds:
            np.random.seed(seed)

            rows.append({
                'wake_model': wake_model_name,
                'seed': seed,
                'iterations': 0,
                'max_step_D': random_max_step_D,
                'method': 'Current Horns Rev',
                'AEP [GWh]': aep_baseline,
                'Improvement over Horns Rev [GWh]': 0.0
            })

            for random_search_layouts in random_search_iterations_list:

                print()
                print("--------------------------------------------------")
                print(f"Running RandomSearch for {wake_model_name}")
                print(f"Seed = {seed}")
                print(f"RandomSearch iterations = {random_search_layouts}")
                print(f"RandomSearch max step = {random_max_step_D}D")
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
                        'run_name': f"iterations_{random_search_layouts}_seed{seed}",
                        'script': 'RunRS_Iterations',
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
                        f"{random_search_layouts} iterations - AEP {aep_random:.3f} GWh"
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
                        f"(RandomSearch only, {random_search_layouts} iterations)",
                        final_aep=aep_random
                    )

                    plot_aep_time_history(
                        random_res['aep_history'],
                        wake_model_name,
                        f"(RandomSearch only, {random_search_layouts} iterations)",
                        final_aep=aep_random
                    )

    # ==================================================
    # SUMMARY TABLE
    # ==================================================

    df = pd.DataFrame(rows)
    df_history = pd.DataFrame(history_rows)

    summary_output_path = results_dir / f"rs_iterations_summary_{timestamp}.csv"
    history_output_path = results_dir / f"rs_iterations_history_{timestamp}.csv"
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

            plt.figure(figsize=(9, 5))
            plt.plot(
                df_sub['iterations'],
                df_sub['AEP [GWh]'],
                marker='o',
                linewidth=2
            )
            plt.title(f"RandomSearch AEP vs iterations - {wake_model_name}")
            plt.xlabel("RandomSearch iterations")
            plt.ylabel("AEP [GWh]")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(9, 5))
            plt.plot(
                df_sub['iterations'],
                df_sub['Improvement over Horns Rev [GWh]'],
                marker='o',
                linewidth=2
            )
            plt.title(f"RandomSearch improvement vs iterations - {wake_model_name}")
            plt.xlabel("RandomSearch iterations")
            plt.ylabel("Improvement over Horns Rev [GWh]")
            plt.grid(True, alpha=0.3)
            plt.axhline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(9, 5))
            plt.scatter(
                df_sub['iterations'],
                df_sub['runtime_sec'],
                s=70
            )
            plt.title(f"RandomSearch runtime vs iterations - {wake_model_name}")
            plt.xlabel("RandomSearch iterations")
            plt.ylabel("Runtime [s]")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    print_run_parameters(
        site_name=site_name,
        seed=seeds,
        mean_ws=mean_ws,
        wd_all=wd_all,
        ws_mean=ws_mean,
        x_points=x_points,
        y_points=y_points,
        spacing_D=spacing_D,
        boundary_pad=boundary_pad,
        random_search_layouts=random_search_iterations_list,
        random_max_step_D=random_max_step_D,
        wake_models=wake_models
    )


if __name__ == "__main__":
    main()
