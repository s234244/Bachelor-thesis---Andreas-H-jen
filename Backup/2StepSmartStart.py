# ----------------------------------------
# Title: Two-Step Smart Start AEP - Horns Rev 1
# Author: Rewritten in same style as original Smart Start code
# Description: This script performs a 2-step smart start optimization maximizing AEP.
# ----------------------------------------

# %%---------------------------------------------------------------------------------#
#                       IMPORT ALL REQUIRED PACKAGES & FUNCTIONS                    #
#-----------------------------------------------------------------------------------#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from matplotlib.patches import Circle
from shapely.geometry import Point, Polygon

# PyWake
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80, wt_x, wt_y
from py_wake.superposition_models import SquaredSum, LinearSum
from py_wake.literature.noj import Jensen_1983
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014

# %% ---------------------------------------------------------------------------------#
#                                DEFINE WIND TURBINE                                  #
#------------------------------------------------------------------------------------#
def set_wt(site_name='HornsRev1'):
    if site_name in ['HornsRev1', 'HornsRev', 'HR1']:
        wt = V80()
        rated_power = 2  # MW
        return wt, rated_power
    else:
        raise ValueError(f"Unknown site name: {site_name}")


# %% ---------------------------------------------------------------------------------#
#                                DEFINE FUNCTIONS                                      #
#------------------------------------------------------------------------------------#

# -------------------------- Define function to get site data------------------------#
def get_site(site_name, wt):
    if site_name not in ['HornsRev1', 'HornsRev', 'HR1']:
        raise ValueError(f"Unknown site name: {site_name}")

    site = Hornsrev1Site()

    # Original Horns Rev coordinates
    x_ref = np.asarray(wt_x)
    y_ref = np.asarray(wt_y)

    # Boundary based on original layout
    boundary_pad = 400
    xmin, xmax = x_ref.min() - boundary_pad, x_ref.max() + boundary_pad
    ymin, ymax = y_ref.min() - boundary_pad, y_ref.max() + boundary_pad

    boundary = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ])

    # Number of turbines
    n_wt = len(x_ref)

    # Grid for candidate points
    x_points = 20
    y_points = 20
    x = np.linspace(xmin, xmax, x_points)
    y = np.linspace(ymin, ymax, y_points)
    XX, YY = np.meshgrid(x, y)

    return boundary, x_ref, y_ref, site, n_wt, x_points, y_points, XX, YY


# ------------------------------- Define wake model ------------------------------#
def set_wake_model(wake_model_name, site, wt, superposition='SquaredSum', k=0.0324555):
    if superposition == 'SquaredSum':
        superposition_model = SquaredSum()
    elif superposition == 'LinearSum':
        superposition_model = LinearSum()
    else:
        raise ValueError(f"Unknown superposition model: {superposition}")

    if wake_model_name == 'NOJ':
        return Jensen_1983(site, wt, superpositionModel=superposition_model)

    elif wake_model_name == 'BastankhahGaussian':
        return Bastankhah_PorteAgel_2014(site, wt, k=k, superpositionModel=superposition_model)

    else:
        raise ValueError(f"Unknown wake model name: {wake_model_name}")


# -------------------------- Build candidate grid inside boundary ------------------#
def build_candidate_grid(boundary, x_points, y_points):
    poly = Polygon(boundary)

    x = np.linspace(boundary[:, 0].min(), boundary[:, 0].max(), x_points)
    y = np.linspace(boundary[:, 1].min(), boundary[:, 1].max(), y_points)

    candidates = []
    for xi in x:
        for yi in y:
            p = Point(xi, yi)
            if poly.contains(p) or poly.touches(p):
                candidates.append([xi, yi])

    return np.array(candidates)


# -------------------------- Check spacing constraint ------------------------------#
def respects_spacing(candidate_xy, layout_xy, min_spacing):
    if len(layout_xy) == 0:
        return True

    dx = layout_xy[:, 0] - candidate_xy[0]
    dy = layout_xy[:, 1] - candidate_xy[1]
    dist = np.sqrt(dx**2 + dy**2)
    return np.all(dist >= min_spacing)


# -------------------------- Convert indices to layout -----------------------------#
def get_layout_xy(selected_indices, candidates):
    if len(selected_indices) == 0:
        return np.empty((0, 2))
    return candidates[np.array(selected_indices)]


# -------------------------- Calculate AEP for layout ------------------------------#
def calc_aep(wf_model, layout_xy, with_wake_loss=True):
    if len(layout_xy) == 0:
        return 0.0

    sim_res = wf_model(layout_xy[:, 0], layout_xy[:, 1])
    aep = sim_res.aep(with_wake_loss=with_wake_loss).sum().data
    return float(np.asarray(aep))


# %% ---------------------------------------------------------------------------------#
#                            DEFINE 2-STEP SMART START                                #
#------------------------------------------------------------------------------------#

# ------------------------------- Stage 1 -----------------------------------------#
def stage1_add_turbines(wf_model, candidates, n_wt, min_spacing):
    selected_indices = []
    insertion_order = []

    # First turbine
    best_idx = None
    best_val = -np.inf

    for idx in range(len(candidates)):
        test_layout = candidates[[idx]]
        val = calc_aep(wf_model, test_layout, with_wake_loss=True)

        if val > best_val:
            best_val = val
            best_idx = idx

    selected_indices.append(best_idx)
    insertion_order.append(best_idx)

    print(f"Stage 1: Added turbine 1/{n_wt}")

    # Remaining turbines
    while len(selected_indices) < n_wt:
        current_layout = get_layout_xy(selected_indices, candidates)

        best_idx = None
        best_val = -np.inf

        for idx in range(len(candidates)):
            if idx in selected_indices:
                continue

            cand_xy = candidates[idx]
            if not respects_spacing(cand_xy, current_layout, min_spacing):
                continue

            trial_indices = selected_indices + [idx]
            trial_layout = get_layout_xy(trial_indices, candidates)
            val = calc_aep(wf_model, trial_layout, with_wake_loss=True)

            if val > best_val:
                best_val = val
                best_idx = idx

        if best_idx is None:
            raise RuntimeError("No feasible grid point found in Stage 1")

        selected_indices.append(best_idx)
        insertion_order.append(best_idx)

        print(f"Stage 1: Added turbine {len(selected_indices)}/{n_wt}")

    return selected_indices, insertion_order


# ------------------------------- Stage 2 -----------------------------------------#
def stage2_relocate_turbines(wf_model, candidates, insertion_order, min_spacing, max_cycles=20):
    positions_by_turbine = insertion_order.copy()

    for cycle in range(max_cycles):
        changed = False
        print(f"Stage 2: Cycle {cycle + 1}")

        for i in range(len(positions_by_turbine)):
            old_idx = positions_by_turbine[i]

            other_positions = positions_by_turbine[:i] + positions_by_turbine[i+1:]
            other_layout = get_layout_xy(other_positions, candidates)

            best_idx = old_idx
            best_val = -np.inf

            for idx in range(len(candidates)):
                if idx in other_positions:
                    continue

                cand_xy = candidates[idx]
                if not respects_spacing(cand_xy, other_layout, min_spacing):
                    continue

                trial_positions = other_positions + [idx]
                trial_layout = get_layout_xy(trial_positions, candidates)
                val = calc_aep(wf_model, trial_layout, with_wake_loss=True)

                if val > best_val:
                    best_val = val
                    best_idx = idx

            if best_idx != old_idx:
                changed = True
                positions_by_turbine[i] = best_idx

        if not changed:
            print("Stage 2 converged: no turbine positions changed in full cycle")
            break

    return positions_by_turbine


# ------------------------------- Full 2-step algorithm ---------------------------#
def two_step_smart_start(wf_model, candidates, n_wt, min_spacing, max_cycles=20):
    selected_indices, insertion_order = stage1_add_turbines(
        wf_model=wf_model,
        candidates=candidates,
        n_wt=n_wt,
        min_spacing=min_spacing
    )

    final_indices = stage2_relocate_turbines(
        wf_model=wf_model,
        candidates=candidates,
        insertion_order=insertion_order,
        min_spacing=min_spacing,
        max_cycles=max_cycles
    )

    final_layout = get_layout_xy(final_indices, candidates)

    return final_layout, final_indices


# %% ---------------------------------------------------------------------------------#
#                               APPLY 2-STEP SMART START                              #
#------------------------------------------------------------------------------------#
results_df = pd.DataFrame(columns=[
    'site',
    'wake_model',
    'AEP [GWh]',
    'AEP without wake loss [GWh]',
    'Wake loss [%]'
])

results = {}

site_name = 'HornsRev1'

wt, rated_power = set_wt(site_name)
boundary, x_ref, y_ref, site, n_wt, x_points, y_points, XX, YY = get_site(site_name, wt)

wake_models = ['NOJ', 'BastankhahGaussian']

for wake_model_name in wake_models:
    print()
    print("--------------------------------------------------")
    print(f"Running 2-step smart start for {wake_model_name}")
    print("--------------------------------------------------")

    wf_model = set_wake_model(wake_model_name, site, wt)

    candidates = build_candidate_grid(boundary, x_points, y_points)

    min_spacing = 4 * wt.diameter()

    layout_xy, final_indices = two_step_smart_start(
        wf_model=wf_model,
        candidates=candidates,
        n_wt=n_wt,
        min_spacing=min_spacing,
        max_cycles=20
    )

    aep_with_wake = calc_aep(wf_model, layout_xy, with_wake_loss=True)
    aep_without_wake = calc_aep(wf_model, layout_xy, with_wake_loss=False)
    wake_loss_pct = ((aep_without_wake - aep_with_wake) / aep_without_wake) * 100

    results[wake_model_name] = {
        'layout_xy': layout_xy,
        'final_indices': final_indices,
        'cost': aep_with_wake,
        'aep_without_wake': aep_without_wake,
        'wake_loss_pct': wake_loss_pct
    }

    results_df = pd.concat([
        results_df,
        pd.DataFrame({
            'site': [site_name],
            'wake_model': [wake_model_name],
            'AEP [GWh]': [aep_with_wake],
            'AEP without wake loss [GWh]': [aep_without_wake],
            'Wake loss [%]': [wake_loss_pct]
        })
    ], ignore_index=True)

    print(f"\n--- {wake_model_name} ---")
    print(f"AEP: {aep_with_wake:.2f} GWh")
    print(f"Wake loss: {wake_loss_pct:.2f} %")

print()
print(results_df)


# %% ---------------------------------------------------------------------------------#
#                               PLOT THE RESULTS                                     #
# -----------------------------------------------------------------------------------#
n_spacing = 4

def plot_layout_hornsrev(layout_xy, boundary, x_ref, y_ref, wt, wake_model_name, cost=None):
    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot candidate boundary
    bx = np.r_[boundary[:, 0], boundary[0, 0]]
    by = np.r_[boundary[:, 1], boundary[0, 1]]
    ax.plot(bx, by, color='black', linewidth=1.5, label='Boundary')

    # Original layout
    ax.scatter(x_ref, y_ref, c='gray', s=25, alpha=0.5, label='Original Horns Rev layout')

    # Optimized layout
    ax.scatter(layout_xy[:, 0], layout_xy[:, 1], c='red', marker='x', s=60, label='Optimized layout')

    # Constraint circles
    for x, y in layout_xy:
        constraint_circle = Circle(
            (x, y),
            (n_spacing / 2) * wt.diameter(),
            edgecolor='black',
            facecolor='none',
            linestyle='--',
            alpha=0.6
        )
        ax.add_patch(constraint_circle)

    cost_text = f"{cost:.1f}" if cost is not None else "N/A"
    ax.set_title(f'{site_name} - {wake_model_name}\nAEP: {cost_text} GWh')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.axis('equal')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


# Plot both results
for wake_model_name in wake_models:
    plot_layout_hornsrev(
        layout_xy=results[wake_model_name]['layout_xy'],
        boundary=boundary,
        x_ref=x_ref,
        y_ref=y_ref,
        wt=wt,
        wake_model_name=wake_model_name,
        cost=results[wake_model_name]['cost']
    )