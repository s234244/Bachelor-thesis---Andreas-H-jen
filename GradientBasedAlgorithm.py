import numpy as np
from time import perf_counter
from Windfarm_utils import get_problem, calc_aep
from SmartStart import run_smartstart


def run_gradient_from_layout(
    wf_model,
    wt,
    boundary,
    n_wt,
    initial_layout,
    spacing_D,
    wd=None,
    ws=None,
    maxiter=300
):
    stage_start = perf_counter()
    aep_history = []

    aep_initial = calc_aep(
        wf_model,
        initial_layout,
        with_wake_loss=True,
        wd=wd,
        ws=ws
    )

    aep_history.append({
        'iteration': 0,
        'phase': 'Gradient',
        'AEP [GWh]': aep_initial
    })

    problem_grad = get_problem(
        wt=wt,
        boundary=boundary,
        wf_model=wf_model,
        n_wt=n_wt,
        x_init=initial_layout[:, 0],
        y_init=initial_layout[:, 1],
        spacing_D=spacing_D,
        wd=wd,
        ws=ws,
        maxiter=maxiter
    )

    opt_result = problem_grad.optimize()
    if isinstance(opt_result, tuple) and len(opt_result) >= 2:
        _, state_grad = opt_result[:2]
        recorder = opt_result[2] if len(opt_result) >= 3 else getattr(problem_grad, 'recorder', None)
    else:
        _, state_grad = problem_grad.evaluate()
        recorder = getattr(problem_grad, 'recorder', None)

    stage_runtime = perf_counter() - stage_start

    gradient_layout = np.column_stack([state_grad['x'], state_grad['y']])
    movement = np.linalg.norm(gradient_layout - initial_layout, axis=1)

    aep_gradient = calc_aep(
        wf_model,
        gradient_layout,
        with_wake_loss=True,
        wd=wd,
        ws=ws
    )

    print(f"Gradient max iterations: {maxiter}")
    print(f"Gradient-based AEP: {aep_gradient:.3f} GWh")
    print(f"Improvement over initial layout: {aep_gradient - aep_initial:.3f} GWh")
    print(f"Max turbine movement: {movement.max():.3f} m")
    print(f"Mean turbine movement: {movement.mean():.3f} m")
    print(f"Gradient runtime: {stage_runtime:.2f} seconds")

    try:
        cost_key = [k for k in recorder.keys() if 'cost' in k.lower()][0]
        recorded_costs = recorder[cost_key]

        for i, cost in enumerate(recorded_costs, start=1):
            aep_history.append({
                'iteration': i,
                'phase': 'Gradient',
                'AEP [GWh]': float(-cost)
            })

    except Exception:
        aep_history.append({
            'iteration': 1,
            'phase': 'Gradient',
            'AEP [GWh]': aep_gradient
        })

    if len(aep_history) == 1:
        aep_history[0]['elapsed_sec'] = stage_runtime
    elif len(aep_history) > 1:
        time_grid = np.linspace(0.0, stage_runtime, len(aep_history))
        for row, elapsed_sec in zip(aep_history, time_grid):
            row['elapsed_sec'] = float(elapsed_sec)

    return {
        'initial_layout': initial_layout,
        'layout_xy': gradient_layout,
        'aep_history': aep_history,
        'runtime_sec': stage_runtime
    }


def run_smartstart_gradient(
    wf_model,
    wt,
    boundary,
    n_wt,
    XX,
    YY,
    spacing_D=4,
    seed=1,
    random_pct=10,
    wd=None,
    ws=None,
    maxiter=300
):
    """Run SmartStart followed by gradient optimization.

    random_pct is given in percent, so 10 means 10%, not 0.10.
    """
    # Stage 1: SmartStart
    smart_res = run_smartstart(
        wf_model=wf_model,
        wt=wt,
        boundary=boundary,
        n_wt=n_wt,
        XX=XX,
        YY=YY,
        spacing_D=spacing_D,
        seed=seed,
        random_pct=random_pct,
        wd=wd,
        ws=ws
    )

    smartstart_layout = smart_res['layout_xy']

    # Calculate AEP for SmartStart
    aep_smart = calc_aep(
        wf_model,
        smartstart_layout,
        with_wake_loss=True,
        wd=wd,
        ws=ws
    )

    # Stage 2: Gradient from SmartStart layout
    grad_res = run_gradient_from_layout(
        wf_model=wf_model,
        wt=wt,
        boundary=boundary,
        n_wt=n_wt,
        initial_layout=smartstart_layout,
        spacing_D=spacing_D,
        wd=wd,
        ws=ws,
        maxiter=maxiter
    )

    gradient_layout = grad_res['layout_xy']

    # Combine AEP histories
    aep_history = []

    # Add SmartStart AEP at iteration 0
    aep_history.append({
        'iteration': 0,
        'cycle': 1,
        'phase': 'Stage 1',
        'AEP [GWh]': aep_smart
    })

    # Add Gradient history, adjusting iterations
    for entry in grad_res['aep_history']:
        aep_history.append({
            'iteration': entry['iteration'],
            'cycle': 1,
            'phase': 'Stage 2',
            'AEP [GWh]': entry['AEP [GWh]']
        })

    return {
        'smartstart_layout': smartstart_layout,
        'layout_xy': gradient_layout,
        'aep_history': aep_history
    }
