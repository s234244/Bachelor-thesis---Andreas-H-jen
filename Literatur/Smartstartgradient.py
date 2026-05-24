import numpy as np
from Backup.Windfarm_utilsv2 import get_problem, calc_aep


def run_smartstart_gradient(
    wf_model,
    wt,
    boundary,
    n_wt,
    XX,
    YY,
    spacing_D,
    seed,
    wd=None,
    ws=None
):
    """
    Two-step optimization:
      1. SmartStart  — heuristic initial layout
      2. Gradient-based SLSQP — continuous refinement via TopFarm

    aep_history matches the format of SmartStart2StepV2 so the same
    plot_aep_history function in Results.py works for both methods:
        [{'iteration': int, 'cycle': int, 'phase': str, 'AEP [GWh]': float}]
    """

    aep_history = []

    # ------------------------------------------------------------------
    # Stage 1: SmartStart
    # ------------------------------------------------------------------
    problem = get_problem(
        wt=wt,
        boundary=boundary,
        wf_model=wf_model,
        n_wt=n_wt,
        spacing_D=spacing_D,
        wd=wd,
        ws=ws
    )

    min_spacing = spacing_D * wt.diameter()

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
    smartstart_layout = np.column_stack([state_smart['x'], state_smart['y']])
    aep_smartstart = calc_aep(wf_model, smartstart_layout, with_wake_loss=True, wd=wd, ws=ws)

    print(f"SmartStart AEP: {aep_smartstart:.3f} GWh")

    # Record SmartStart result — iteration 0, cycle 0, phase 'Stage 1'
    aep_history.append({
        'iteration': 0,
        'cycle': 0,
        'phase': 'Stage 1',
        'AEP [GWh]': aep_smartstart
    })

    # ------------------------------------------------------------------
    # Stage 2: Gradient-based optimization (SLSQP) via TopFarm
    # Uses the SmartStart layout as the initial guess
    # ------------------------------------------------------------------
    problem_grad = get_problem(
        wt=wt,
        boundary=boundary,
        wf_model=wf_model,
        n_wt=n_wt,
        spacing_D=spacing_D,
        x_init=smartstart_layout[:, 0],
        y_init=smartstart_layout[:, 1],
        wd=wd,
        ws=ws
    )

    problem_grad.optimize()
    _, state_grad = problem_grad.evaluate()

    gradient_layout = np.column_stack([state_grad['x'], state_grad['y']])
    aep_gradient = calc_aep(wf_model, gradient_layout, with_wake_loss=True, wd=wd, ws=ws)

    print(f"Gradient-based AEP: {aep_gradient:.3f} GWh")
    print(f"Improvement over SmartStart: {aep_gradient - aep_smartstart:.3f} GWh")

    # Try to extract per-iteration AEP from TopFarm recorder
    try:
        recorder = problem_grad.recorder
        cost_key = [k for k in recorder.keys() if 'cost' in k.lower()][0]
        recorded_costs = recorder[cost_key]

        # TopFarm minimizes negative AEP, so costs are negative GWh values
        for i, cost in enumerate(recorded_costs):
            aep_history.append({
                'iteration': i + 1,
                'cycle': 1,
                'phase': 'Stage 2',
                'AEP [GWh]': float(-cost)
            })

    except Exception:
        # Fallback: just record the final gradient result as a single entry
        aep_history.append({
            'iteration': 1,
            'cycle': 1,
            'phase': 'Stage 2',
            'AEP [GWh]': aep_gradient
        })

    return {
        'smartstart_layout': smartstart_layout,
        'layout_xy': gradient_layout,
        'aep_smartstart': aep_smartstart,
        'aep_gradient': aep_gradient,
        'aep_history': aep_history
    }