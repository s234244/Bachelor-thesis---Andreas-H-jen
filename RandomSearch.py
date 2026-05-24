import numpy as np
from time import perf_counter

from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyRandomSearchDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
from topfarm.plotting import NoPlot
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from py_wake.utils.gradients import autograd as pw_autograd

from Windfarm_utilsv3 import calc_aep


def run_randomsearch(
    wf_model,
    wt,
    boundary,
    n_wt,
    initial_layout,
    spacing_D=4,
    max_iter=100,
    max_time=600,
    max_step=None,
    seed=1,
    wd=None,
    ws=None
):
    if initial_layout is None:
        raise ValueError("initial_layout must be provided for TopFarm random search.")

    if max_step is None:
        max_step = wt.diameter()

    np.random.seed(seed)

    problem = TopFarmProblem(
        design_vars={
            'x': initial_layout[:, 0],
            'y': initial_layout[:, 1]
        },
        driver=EasyRandomSearchDriver(
            randomize_func=RandomizeTurbinePosition_Circle(max_step=max_step),
            max_iter=max_iter,
            max_time=max_time,
            disp=False
        ),
        cost_comp=PyWakeAEPCostModelComponent(
            wf_model,
            n_wt,
            grad_method=pw_autograd,
            objective=True,
            wd=wd,
            ws=ws
        ),
        constraints=[
            XYBoundaryConstraint(boundary, boundary_type='polygon'),
            SpacingConstraint(spacing_D * wt.diameter())
        ],
        plot_comp=NoPlot()
    )

    stage_start = perf_counter()
    cost_random, state_random, recorder = problem.optimize()
    stage_runtime = perf_counter() - stage_start

    aep_history = []
    best_layout = initial_layout.copy()
    best_aep = calc_aep(
        wf_model,
        best_layout,
        with_wake_loss=True,
        wd=wd,
        ws=ws
    )

    try:
        recorder_x = recorder['x']
        recorder_y = recorder['y']

        aep_history.append({
            'iteration': 0,
            'phase': 'RandomSearch',
            'AEP [GWh]': best_aep
        })

        for iteration, (x, y) in enumerate(zip(recorder_x, recorder_y), start=1):
            layout_iter = np.column_stack([x, y])
            aep_iter = calc_aep(
                wf_model,
                layout_iter,
                with_wake_loss=True,
                wd=wd,
                ws=ws
            )

            if aep_iter > best_aep:
                best_aep = aep_iter
                best_layout = layout_iter

            aep_history.append({
                'iteration': iteration,
                'phase': 'RandomSearch',
                'AEP [GWh]': best_aep
            })
    except Exception:
        aep_history.append({
            'iteration': 1,
            'phase': 'RandomSearch',
            'AEP [GWh]': best_aep
        })

    if len(aep_history) == 1:
        aep_history[0]['elapsed_sec'] = stage_runtime
    elif len(aep_history) > 1:
        time_grid = np.linspace(0.0, stage_runtime, len(aep_history))
        for row, elapsed_sec in zip(aep_history, time_grid):
            row['elapsed_sec'] = float(elapsed_sec)

    return {
        'initial_layout': initial_layout,
        'layout_xy': best_layout,
        'state': state_random,
        'aep': best_aep,
        'aep_history': aep_history,
        'raw_cost': cost_random,
        'runtime_sec': stage_runtime
    }
