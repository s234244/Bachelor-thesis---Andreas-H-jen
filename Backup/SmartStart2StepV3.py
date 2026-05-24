import numpy as np
from shapely.geometry import Point, Polygon
from Backup.Windfarm_utils import get_problem, calc_aep


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


def get_layout_xy(selected_indices, candidates):
    if len(selected_indices) == 0:
        return np.empty((0, 2))
    return candidates[np.array(selected_indices)]


def stage1_topfarm_smart_start(problem, XX, YY, candidates, min_spacing, seed=1):
    problem.smart_start(
        XX,
        YY,
        ZZ=problem.cost_comp.get_aep4smart_start(),
        min_space=min_spacing,
        random_pct=0.10,
        seed=seed,
        plot=False
    )

    _, state_smart = problem.evaluate()
    stage1_layout = np.column_stack([state_smart['x'], state_smart['y']])

    selected_indices = []
    used = set()

    for xy in stage1_layout:
        d2 = np.sum((candidates - xy) ** 2, axis=1)
        for idx in np.argsort(d2):
            if idx not in used:
                selected_indices.append(idx)
                used.add(idx)
                break

    return stage1_layout, selected_indices, state_smart


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

    aep_history = []
    iteration = 0

    initial_layout = get_layout_xy(positions_by_turbine, candidates)
    initial_aep = calc_aep(wf_model, initial_layout, with_wake_loss=True, wd=wd, ws=ws)

    aep_history.append({
        'iteration': iteration,
        'cycle': 0,
        'phase': 'Stage 2',
        'AEP [GWh]': initial_aep
    })
    iteration += 1

    best_cycle_aep = initial_aep
    tolerance = 1e-3

    for cycle in range(max_cycles):
        changed = False
        print(f"Stage 2: Cycle {cycle + 1}")

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
                'cycle': cycle + 1,
                'phase': 'Stage 2',
                'AEP [GWh]': current_aep
            })
            iteration += 1

        cycle_layout = get_layout_xy(positions_by_turbine, candidates)
        cycle_aep = calc_aep(wf_model, cycle_layout, with_wake_loss=True, wd=wd, ws=ws)

        if not changed:
            print("Stage 2 converged: no turbine positions changed in full cycle")
            break

        if cycle_aep <= best_cycle_aep + tolerance:
            print(f"Stage 2 stopped: no further AEP improvement after cycle {cycle + 1}")
            break

        best_cycle_aep = cycle_aep

    return positions_by_turbine, aep_history


def run_two_step(
    wf_model,
    wt,
    boundary,
    n_wt,
    XX,
    YY,
    x_points,
    y_points,
    spacing_D,
    seed,
    max_cycles,
    wd=None,
    ws=None
):
    candidates = build_candidate_grid(boundary, x_points, y_points)
    min_spacing = spacing_D * wt.diameter()

    problem = get_problem(
        wt=wt,
        boundary=boundary,
        wf_model=wf_model,
        n_wt=n_wt,
        spacing_D=spacing_D,
        wd=wd,
        ws=ws
    )

    stage1_layout, selected_indices, state_smart = stage1_topfarm_smart_start(
        problem=problem,
        XX=XX,
        YY=YY,
        candidates=candidates,
        min_spacing=min_spacing,
        seed=seed
    )

    aep_stage1 = calc_aep(wf_model, stage1_layout, with_wake_loss=True, wd=wd, ws=ws)

    final_indices, stage2_history = stage2_relocate_turbines(
        wf_model=wf_model,
        candidates=candidates,
        insertion_order=selected_indices,
        min_spacing=min_spacing,
        max_cycles=max_cycles,
        wd=wd,
        ws=ws
    )

    final_layout = get_layout_xy(final_indices, candidates)

    aep_history = [{
        'iteration': 0,
        'cycle': 0,
        'phase': 'Stage 1',
        'AEP [GWh]': aep_stage1
    }]

    for row in stage2_history:
        row_copy = row.copy()
        row_copy['iteration'] += 1
        aep_history.append(row_copy)

    return {
        'stage1_layout': stage1_layout,
        'stage1_state': state_smart,
        'layout_xy': final_layout,
        'aep_history': aep_history
    }