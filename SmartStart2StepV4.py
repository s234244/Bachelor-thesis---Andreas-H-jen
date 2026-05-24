import numpy as np
from time import perf_counter
from shapely.geometry import Point, Polygon
from Windfarm_utilsv3 import calc_aep


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

    return np.array(candidates, dtype=float)


def append_unique_points(candidates, points, tol=1e-9):
    if len(points) == 0:
        return candidates

    merged = candidates.tolist() if len(candidates) > 0 else []

    for point in np.asarray(points, dtype=float):
        if len(merged) == 0:
            merged.append(point.tolist())
            continue

        merged_arr = np.asarray(merged, dtype=float)
        d2 = np.sum((merged_arr - point) ** 2, axis=1)
        if not np.any(d2 <= tol ** 2):
            merged.append(point.tolist())

    return np.asarray(merged, dtype=float)


def get_layout_xy(selected_indices, candidates):
    if len(selected_indices) == 0:
        return np.empty((0, 2))
    return candidates[np.array(selected_indices)]


def map_layout_to_candidates(initial_layout, candidates):
    selected_indices = []
    used = set()

    for xy in initial_layout:
        d2 = np.sum((candidates - xy) ** 2, axis=1)
        for idx in np.argsort(d2):
            if idx not in used:
                selected_indices.append(idx)
                used.add(idx)
                break

    return selected_indices


def stage2_relocate_turbines(
    wf_model,
    candidates,
    insertion_order,
    min_spacing,
    max_cycles,
    wd=None,
    ws=None
):
    positions_by_turbine = insertion_order.copy()
    min_spacing2 = min_spacing ** 2
    stage_start = perf_counter()

    aep_history = []
    iteration = 0

    initial_layout = get_layout_xy(positions_by_turbine, candidates)
    initial_aep = calc_aep(wf_model, initial_layout, with_wake_loss=True, wd=wd, ws=ws)

    aep_history.append({
        'iteration': iteration,
        'phase': 'Two-step',
        'AEP [GWh]': initial_aep,
        'elapsed_sec': perf_counter() - stage_start
    })
    iteration += 1

    best_cycle_aep = initial_aep
    tolerance = 1e-3

    for cycle in range(max_cycles):
        changed = False
        print(f"Two-step: Cycle {cycle + 1}")

        for i in range(len(positions_by_turbine)):
            old_idx = positions_by_turbine[i]

            other_positions = positions_by_turbine[:i] + positions_by_turbine[i + 1:]
            other_positions_set = set(other_positions)
            other_layout = get_layout_xy(other_positions, candidates)

            best_idx = old_idx
            best_aep = -np.inf

            for idx in range(len(candidates)):
                if idx in other_positions_set:
                    continue

                cand_xy = candidates[idx]

                if len(other_layout) > 0:
                    dx = other_layout[:, 0] - cand_xy[0]
                    dy = other_layout[:, 1] - cand_xy[1]
                    dist2 = dx * dx + dy * dy
                    if np.any(dist2 < min_spacing2):
                        continue

                trial_positions = other_positions + [idx]
                trial_layout = get_layout_xy(trial_positions, candidates)
                trial_aep = calc_aep(wf_model, trial_layout, with_wake_loss=True, wd=wd, ws=ws)

                if trial_aep > best_aep:
                    best_aep = trial_aep
                    best_idx = idx

            if best_idx != old_idx:
                positions_by_turbine[i] = best_idx
                changed = True

            current_layout = get_layout_xy(positions_by_turbine, candidates)
            current_aep = calc_aep(wf_model, current_layout, with_wake_loss=True, wd=wd, ws=ws)

            aep_history.append({
                'iteration': iteration,
                'phase': 'Two-step',
                'AEP [GWh]': current_aep,
                'elapsed_sec': perf_counter() - stage_start
            })
            iteration += 1

        cycle_layout = get_layout_xy(positions_by_turbine, candidates)
        cycle_aep = calc_aep(wf_model, cycle_layout, with_wake_loss=True, wd=wd, ws=ws)

        if not changed:
            print("Two-step converged: no turbine positions changed in full cycle")
            break

        if cycle_aep <= best_cycle_aep + tolerance:
            print(f"Two-step stopped: no further AEP improvement after cycle {cycle + 1}")
            break

        best_cycle_aep = cycle_aep

    return positions_by_turbine, aep_history


def run_two_step_from_layout(
    wf_model,
    wt,
    boundary,
    n_wt,
    initial_layout,
    x_points,
    y_points,
    spacing_D,
    max_cycles,
    wd=None,
    ws=None
):
    candidates = build_candidate_grid(boundary, x_points, y_points)
    candidates = append_unique_points(candidates, initial_layout)
    min_spacing = spacing_D * wt.diameter()

    selected_indices = map_layout_to_candidates(initial_layout, candidates)

    final_indices, aep_history = stage2_relocate_turbines(
        wf_model=wf_model,
        candidates=candidates,
        insertion_order=selected_indices,
        min_spacing=min_spacing,
        max_cycles=max_cycles,
        wd=wd,
        ws=ws
    )

    final_layout = get_layout_xy(final_indices, candidates)

    return {
        'initial_layout': initial_layout,
        'layout_xy': final_layout,
        'aep_history': aep_history
    }
