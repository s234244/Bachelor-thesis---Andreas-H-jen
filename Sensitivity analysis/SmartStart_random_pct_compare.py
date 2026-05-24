import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from Backup.Windfarm_utilsv2 import set_wt, get_site, set_wake_model, calc_aep
from SmartStart import run_smartstart


site_name = 'HornsRev1'
wake_model_name = 'NOJ'
random_pcts = [0, 3, 5, 10, 20, 50]
seeds = [1, 2, 3]
spacing_d = 4
x_points = 20
y_points = 20
boundary_pad = 400
mean_ws = 9.6
wd_all = np.arange(0, 360, 5)
ws_mean = np.array([mean_ws])


def build_history(wf_model, layout_xy, wd, ws):
    history = []

    for iteration in range(1, len(layout_xy) + 1):
        aep = calc_aep(
            wf_model,
            layout_xy[:iteration],
            with_wake_loss=True,
            wd=wd,
            ws=ws
        )
        history.append({
            'iteration': iteration,
            'AEP [GWh]': aep
        })

    return history


def random_pct_label(random_pct):
    return f'{random_pct:g}%'


def main():
    wt, rated_power = set_wt(site_name)
    boundary, x_ref, y_ref, site, n_wt, _, _, XX, YY = get_site(
        site_name=site_name,
        wt=wt,
        boundary_pad=boundary_pad,
        x_points=x_points,
        y_points=y_points
    )

    wf_model = set_wake_model(wake_model_name, site, wt)
    rows = []

    for seed in seeds:
        for random_pct in random_pcts:
            print(f"Running SmartStart with seed = {seed}, random_pct = {random_pct_label(random_pct)}")

            result = run_smartstart(
                wf_model=wf_model,
                wt=wt,
                boundary=boundary,
                n_wt=n_wt,
                XX=XX,
                YY=YY,
                spacing_D=spacing_d,
                seed=seed,
                random_pct=random_pct,
                wd=wd_all,
                ws=ws_mean
            )

            layout_xy = result['layout_xy']
            history = build_history(wf_model, layout_xy, wd_all, ws_mean)
            final_aep = history[-1]['AEP [GWh]']

            print(f"Final AEP: {final_aep:.3f} GWh")

            for row in history:
                rows.append({
                    'seed': seed,
                    'random_pct': random_pct,
                    'iteration': row['iteration'],
                    'AEP [GWh]': row['AEP [GWh]']
                })

    df = pd.DataFrame(rows)
    results_dir = Path('CSV_SS')
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / 'smartstart_random_pct_compare.csv'

    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(random_pcts)))
    df_mean = (
        df.groupby(['random_pct', 'iteration'], as_index=False)['AEP [GWh]']
        .mean()
    )

    for color, random_pct in zip(colors, random_pcts):
        df_sub = df_mean[df_mean['random_pct'] == random_pct]

        axes[0].plot(
            df_sub['iteration'],
            df_sub['AEP [GWh]'],
            color=color,
            linewidth=2,
            marker='o',
            markersize=4,
            markevery=8,
            label=f'random_pct = {random_pct_label(random_pct)}'
        )

    baseline = df_mean[df_mean['random_pct'] == random_pcts[0]][['iteration', 'AEP [GWh]']].rename(
        columns={'AEP [GWh]': 'baseline_aep'}
    )

    for color, random_pct in zip(colors, random_pcts[1:]):
        df_sub = df_mean[df_mean['random_pct'] == random_pct].merge(baseline, on='iteration')
        delta = df_sub['AEP [GWh]'] - df_sub['baseline_aep']

        axes[1].plot(
            df_sub['iteration'],
            delta,
            color=color,
            linewidth=2,
            marker='o',
            markersize=4,
            markevery=8,
            label=f'{random_pct_label(random_pct)} - {random_pct_label(random_pcts[0])}'
        )

    axes[0].set_title(f'Mean SmartStart comparison across {len(seeds)} seeds')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('AEP [GWh]')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axhline(0.0, color='black', linewidth=1, alpha=0.6)
    axes[1].set_title(f'Mean difference from random_pct = {random_pct_label(random_pcts[0])}')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Delta AEP [GWh]')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
