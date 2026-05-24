import numpy as np
import types

try:
    import coverage
except ImportError:
    coverage = None
else:
    if not hasattr(coverage, "types"):
        coverage.types = types.SimpleNamespace(
            Tracer=coverage.PyTracer,
            TTraceData=dict,
            TShouldTraceFn=object,
            TFileDisposition=object,
            TShouldStartContextFn=object,
            TWarnFn=object,
            TTraceFn=object,
        )

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80, wt_x, wt_y
from py_wake.superposition_models import SquaredSum, LinearSum
from py_wake.literature.noj import Jensen_1983
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014

from topfarm import TopFarmProblem
from topfarm.plotting import XYPlotComp
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from py_wake.utils.gradients import autograd as pw_autograd


def set_wt(site_name='HornsRev1'):
    if site_name in ['HornsRev1', 'HornsRev', 'HR1']:
        wt = V80()
        rated_power = 2.0
        return wt, rated_power
    raise ValueError(f"Unknown site name: {site_name}")


def get_site(site_name='HornsRev1', wt=None, boundary_pad=400, x_points=20, y_points=20):
    if site_name not in ['HornsRev1', 'HornsRev', 'HR1']:
        raise ValueError(f"Unknown site name: {site_name}")

    site = Hornsrev1Site()

    x_ref = np.asarray(wt_x)
    y_ref = np.asarray(wt_y)
    n_wt = len(x_ref)

    xmin, xmax = x_ref.min() - boundary_pad, x_ref.max() + boundary_pad
    ymin, ymax = y_ref.min() - boundary_pad, y_ref.max() + boundary_pad

    boundary = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ])

    x = np.linspace(xmin, xmax, x_points)
    y = np.linspace(ymin, ymax, y_points)
    XX, YY = np.meshgrid(x, y)

    return boundary, x_ref, y_ref, site, n_wt, x_points, y_points, XX, YY


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


def get_problem(wt, boundary, wf_model, n_wt, x_init=None, y_init=None, spacing_D=4, wd=None, ws=None):
    spacing_constraint = spacing_D * wt.diameter()

    xmin, xmax = boundary[:, 0].min(), boundary[:, 0].max()
    ymin, ymax = boundary[:, 1].min(), boundary[:, 1].max()

    if x_init is None:
        x_init = np.random.uniform(xmin, xmax, size=n_wt)
    if y_init is None:
        y_init = np.random.uniform(ymin, ymax, size=n_wt)

    # Build keyword arguments for PyWakeAEPCostModelComponent
    aep_kwargs = {}
    if wd is not None:
        aep_kwargs['wd'] = wd
    if ws is not None:
        aep_kwargs['ws'] = ws

    problem = TopFarmProblem(
        design_vars={'x': x_init, 'y': y_init},
        driver=EasyScipyOptimizeDriver(optimizer='SLSQP',disp=True,maxiter=300,tol=1e-6),
        cost_comp=PyWakeAEPCostModelComponent(
            wf_model,
            n_wt,
            grad_method=pw_autograd,
            objective=True,
            **aep_kwargs
        ),
        constraints=[
            XYBoundaryConstraint(boundary, boundary_type='polygon'),
            SpacingConstraint(spacing_constraint)
        ],
        plot_comp=XYPlotComp()
    )
    return problem


def calc_aep(wf_model, layout_xy, with_wake_loss=False, wd=None, ws=None):
    if len(layout_xy) == 0:
        return 0.0

    # Build keyword arguments for the simulation
    sim_kwargs = {}
    if wd is not None:
        sim_kwargs['wd'] = wd
    if ws is not None:
        sim_kwargs['ws'] = ws

    sim_res = wf_model(layout_xy[:, 0], layout_xy[:, 1], **sim_kwargs)
    return float(np.asarray(sim_res.aep(with_wake_loss=with_wake_loss).sum().data))
