import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from Windfarm_utilsv3 import set_wt, get_site, set_wake_model, calc_aep
from SmartstartV2 import run_smartstart
from SmartstartgradientV2 import run_smartstart_gradient

# Mean wind speed and all wind directions
mean_ws = 9.6
wd_all = np.arange(0, 360, 10)   # 10 degree bins for wind direction
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


def plot_layout(layout_xy, boundary, x_ref, y_ref, wt, title, spacing_D=4):
    fig, ax = plt.subplots(figsize=(10, 8))

    bx = np.r_[boundary[:, 0], boundary[0, 0]]
    by = np.r_[boundary[:, 1], boundary[0, 1]]
    ax.plot(bx, by, color='black', linewidth=1.5, label='Boundary')

    ax.scatter(
        x_ref, y_ref,
        c='gray',
        s=25,
        alpha=0.5,
        label='Original Horns Rev layout'
    )
    ax.scatter(
        layout_xy[:, 0],
        layout_xy[:, 1],
        c='red',
        marker='x',
        s=60,
        label='Optimized layout'
    )

    for x, y in layout_xy:
        circle = Circle(
            (x, y),
            radius=(spacing_D / 2) * wt.diameter(),
            edgecolor='black',
            facecolor='none',
            linestyle='--',
            alpha=0.5
        )
        ax.add_patch(circle)

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_aep_history(aep_history, wake_model_name):
    df_hist = pd.DataFrame(aep_history)

    fig, ax = plt.subplots(figsize=(10, 6))

    cycles = sorted(df_hist['cycle'].unique())
    cmap = plt.cm.get_cmap('tab10', len(cycles))

    for i, cycle in enumerate(cycles):
        color = cmap(i)
        df_cycle = df_hist[df_hist['cycle'] == cycle].copy()

        for phase, linestyle in [('Stage 1', '--'), ('Stage 2', '-')]:
            df_phase = df_cycle[df_cycle['phase'] == phase]
            if not df_phase.empty:
                ax.plot(
                    df_phase['iteration'],
                    df_phase['AEP [GWh]'],
                    marker='o',
                    linestyle=linestyle,
                    color=color,
                    label=f"Cycle {cycle} - {phase}"
                )

    ax.set_title(f"AEP development over time - {wake_model_name}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("AEP [GWh]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    site_name = 'HornsRev1'
    seed = 1
    np.random.seed(seed)

    wt, rated_power = set_wt(site_name)
    boundary, x_ref, y_ref, site, n_wt, site_x_points, site_y_points, XX, YY, candidate_points = get_site(
        site_name=site_name,
        wt=wt,
        boundary_pad=boundary_pad,
        x_points=x_points,
        y_points=y_points
    )

    rows = []

    for wake_model_name in wake_models:
        print()
        print("==================================================")
        print(f"Running comparison for {wake_model_name}")
        print("==================================================")
        print(f"Using mean wind speed ws = {mean_ws} m/s and all wind directions")

        wf_model = set_wake_model(wake_model_name, site, wt)

        # SmartStart only
        smart_res = run_smartstart(
            wf_model=wf_model,
            wt=wt,
            boundary=boundary,
            n_wt=n_wt,
            XX=XX,
            YY=YY,
            spacing_D=spacing_D,
            seed=seed
        )

        smart_layout = smart_res['layout_xy']
        aep_smart = calc_aep(
            wf_model,
            smart_layout,
            with_wake_loss=True,
            wd=wd_all,
            ws=ws_mean
        )

        # SmartStart + Gradient
        grad_res = run_smartstart_gradient(
            wf_model=wf_model,
            wt=wt,
            boundary=boundary,
            n_wt=n_wt,
            XX=XX,
            YY=YY,
            spacing_D=spacing_D,
            seed=seed,
            wd=wd_all,
            ws=ws_mean,
            maxiter=max_iterations
        )

        stage1_layout = grad_res['smartstart_layout']
        gradient_layout = grad_res['layout_xy']
        aep_history = grad_res['aep_history']

        aep_stage1 = calc_aep(
            wf_model,
            stage1_layout,
            with_wake_loss=True,
            wd=wd_all,
            ws=ws_mean
        )
        aep_gradient = calc_aep(
            wf_model,
            gradient_layout,
            with_wake_loss=True,
            wd=wd_all,
            ws=ws_mean
        )

        print(f"SmartStart only AEP: {aep_smart:.3f} GWh")
        print(f"Gradient Stage 2 AEP: {aep_gradient:.3f} GWh")
        print(f"Gradient improvement: {aep_gradient - aep_stage1:.3f} GWh")

        rows.extend([
            {'wake_model': wake_model_name, 'method': 'SmartStart only', 'AEP [GWh]': aep_smart},
            {'wake_model': wake_model_name, 'method': 'Smart start', 'AEP [GWh]': aep_stage1},
            {'wake_model': wake_model_name, 'method': 'Gradient Based', 'AEP [GWh]': aep_gradient},
        ])

        plot_layout(
            smart_layout,
            boundary,
            x_ref,
            y_ref,
            wt,
            f"{wake_model_name} - SmartStart only",
            spacing_D
        )

        plot_layout(
            gradient_layout,
            boundary,
            x_ref,
            y_ref,
            wt,
            f"{wake_model_name} - Gradient Stage 2",
            spacing_D
        )

        plot_aep_history(aep_history, wake_model_name)

    df = pd.DataFrame(rows)
    print()
    print(df)

    for wake_model_name in wake_models:
        df_sub = df[df['wake_model'] == wake_model_name]

        plt.figure(figsize=(8, 5))
        plt.bar(df_sub['method'], df_sub['AEP [GWh]'])
        plt.title(f"AEP comparison - {wake_model_name}")
        plt.ylabel("AEP [GWh]")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
