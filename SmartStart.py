import numpy as np
from time import perf_counter
from Windfarm_utils import get_problem


def run_smartstart(
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
    ws=None
):
    """Run SmartStart.

    random_pct is given in percent, so 10 means 10%, not 0.10.
    """
    stage_start = perf_counter()

    problem = get_problem(
        wt=wt,
        boundary=boundary,
        wf_model=wf_model,
        n_wt=n_wt,
        spacing_D=spacing_D,
        wd=wd,
        ws=ws
    )

    problem.smart_start(
        XX,
        YY,
        ZZ=problem.cost_comp.get_aep4smart_start(),
        min_space=spacing_D * wt.diameter(),
        random_pct=random_pct,
        seed=seed,
        plot=False
    )

    cost_smart, state_smart = problem.evaluate()
    stage_runtime = perf_counter() - stage_start
    layout_xy = np.column_stack([state_smart['x'], state_smart['y']])

    return {
        'layout_xy': layout_xy,
        'state': state_smart,
        'aep': abs(cost_smart),
        'runtime_sec': stage_runtime
    }
