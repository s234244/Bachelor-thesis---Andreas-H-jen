from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


# ==================================================
# PATH SETUP
# ==================================================

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from Windfarm_utils import set_wt, get_site, set_wake_model, calc_aep


# ==================================================
# SETTINGS
# ==================================================

csv_dir = base_dir / "CSV_PL_Comparison"

method_order = [
    "SS",
    "SS--2S",
    "SS--GB",
    "SS--2S--GB",
    "RS",
    "RS--2S",
    "RS--GB",
    "RS--2S--GB",
]

method_colors = {
    "Current Horns Rev": "black",
    "SS": "tab:purple",
    "SS--2S": "tab:pink",
    "SS--GB": "tab:brown",
    "SS--2S--GB": "tab:gray",
    "RS": "tab:blue",
    "RS--2S": "tab:orange",
    "RS--GB": "tab:green",
    "RS--2S--GB": "tab:red",
}

spacing_D = 4
boundary_pad = 400
show_prevailing_wind_arrow = True
use_local_layout_coordinates_km = True
layout_axis_min_km = -1.0
wind_arrow_xmin = layout_axis_min_km  # Fixed left x-axis limit for local km layout plots with wind arrow. Set to None to disable.
wind_arrow_start_x_km = -0.75
wind_arrow_end_x_km = -0.25
wind_arrow_plot_wd_deg = None  # None uses the most frequent Horns Rev 1 sector.

# ==================================================
# HIGH-RESOLUTION FINAL EVALUATION SETTINGS
# ==================================================
# These settings are used only after optimization.
# The layouts are not optimized again.
# They are only re-evaluated with a finer AEP model.

use_highres_aep_for_comparison = True  # If True, the final comparison table and layout titles use the high-resolution AEP. If False, they use the original optimization/surrogate AEP.
show_layout_plots = False
show_movement_plots = False
show_development_plots = False
show_runtime_plot = False

wd_eval = np.arange(0, 360, 1)       # 1 degree wind direction bins
ws_eval = np.arange(3, 26, 1)        # 1 m/s wind speed bins


# ==================================================
# CSV HELPERS
# ==================================================

def bin_label_from_path(path):
    match = re.search(r"_wd(\d+)deg_", path.name)
    if match:
        return f"{match.group(1)} deg bins"
    return "unknown bins"


def load_latest_csv_per_bin(pattern):
    paths = sorted(
        csv_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    if not paths:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    selected_paths = {}
    for path in paths:
        bin_label = bin_label_from_path(path)
        if bin_label not in selected_paths:
            selected_paths[bin_label] = path

    dataframes = []
    for bin_label, path in sorted(selected_paths.items()):
        print(f"Using {pattern} [{bin_label}]: {path}")
        df = pd.read_csv(path)
        df["source_file"] = path.name
        df["opt_bin_label"] = bin_label
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


# ==================================================
# DATA PREPARATION
# ==================================================

def prepare_aep_column(df):
    df = df.copy()

    if "method" in df.columns:
        df["method"] = df["method"].replace({"SmartStart": "SS"})

    if "wd_step_deg" in df.columns:
        df["opt_bin_label"] = (
            pd.to_numeric(df["wd_step_deg"], errors="coerce")
            .astype("Int64")
            .astype(str)
            .str.replace("<NA>", "unknown", regex=False) +
            " deg bins"
        )
    elif "opt_bin_label" not in df.columns:
        df["opt_bin_label"] = "unknown bins"

    df["plot_AEP [GWh]"] = df["AEP [GWh]"]
    return df


def apply_ss_history_aep(df, df_history):
    """
    Uses the final history AEP for the SS pipeline phases where needed.
    This keeps the same logic as your original script.
    """
    df = df.copy()
    ss_history = df_history[df_history["script"].eq("ResultsSS")].copy()

    if ss_history.empty:
        return df

    history_maps = []

    phase_method_map = {
        "SmartStart": ["SS", "SS--GB"],
        "Two-step": ["SS--2S"],
        "Gradient": ["SS--2S--GB"],
    }

    for phase, methods in phase_method_map.items():
        df_phase = ss_history[ss_history["phase"].eq(phase)].copy()

        if df_phase.empty:
            continue

        df_final = (
            df_phase.sort_values("iteration")
            .groupby(["script", "wake_model", "seed", "opt_bin_label"], as_index=False)
            .tail(1)
        )

        for method in methods:
            df_method = df_final[
                ["script", "wake_model", "seed", "opt_bin_label", "AEP [GWh]"]
            ].copy()

            df_method["method"] = method

            history_maps.append(
                df_method.rename(columns={"AEP [GWh]": "history_AEP [GWh]"})
            )

    if not history_maps:
        return df

    df_history_map = pd.concat(history_maps, ignore_index=True)

    df = df.merge(
        df_history_map,
        on=["script", "wake_model", "seed", "opt_bin_label", "method"],
        how="left"
    )

    df["plot_AEP [GWh]"] = np.where(
        df["history_AEP [GWh]"].notna(),
        df["history_AEP [GWh]"],
        df["plot_AEP [GWh]"]
    )

    return df.drop(columns=["history_AEP [GWh]"])


def add_interpolated_smartstart_history(df_history, df_summary, n_points=25):
    """
    Adds a plotting-only SmartStart segment from 0 GWh at t=0
    to the recorded final SmartStart point.

    TOPFARM SmartStart does not record intermediate AEP values, so these rows
    are marked as interpolated and should only be used for visualization.
    """
    df_history = df_history.copy()
    interpolated_rows = []

    ss_start_rows = df_history[
        (df_history["script"] == "ResultsSS") &
        (df_history["phase"] == "SmartStart")
    ].copy()

    for _, smart_row in ss_start_rows.iterrows():
        start_aep = 0.0
        end_aep = float(smart_row["AEP [GWh]"])
        end_time = float(smart_row["elapsed_sec"])

        if end_time <= 0:
            continue

        for i, frac in enumerate(np.linspace(0.0, 1.0, n_points), start=0):
            interpolated_rows.append({
                "run_name": smart_row["run_name"],
                "script": smart_row["script"],
                "wake_model": smart_row["wake_model"],
                "seed": smart_row["seed"],
                "iteration": -n_points + i,
                "phase": "SmartStart interpolated",
                "AEP [GWh]": start_aep + frac * (end_aep - start_aep),
                "elapsed_sec": frac * end_time,
                "wd_step_deg": smart_row.get("wd_step_deg", np.nan),
                "n_wd": smart_row.get("n_wd", np.nan),
                "ws": smart_row.get("ws", np.nan),
                "source_file": smart_row.get("source_file", ""),
                "opt_bin_label": smart_row["opt_bin_label"],
                "data_source": "interpolated_from_total_smartstart_runtime",
            })

    if interpolated_rows:
        df_history["data_source"] = df_history.get("data_source", "recorded")
        df_history = df_history[
            ~(
                (df_history["script"] == "ResultsSS") &
                (df_history["phase"] == "SmartStart")
            )
        ]
        df_history = pd.concat(
            [df_history, pd.DataFrame(interpolated_rows)],
            ignore_index=True
        )

    return df_history


# ==================================================
# HIGH-RESOLUTION AEP EVALUATION
# ==================================================

def scalar_aep(aep_value):
    """
    Converts different possible calc_aep outputs into one scalar float.
    This makes the script robust if calc_aep returns a float, numpy value,
    array, or xarray-like object.
    """
    try:
        return float(aep_value)
    except TypeError:
        return float(np.asarray(aep_value).sum())


def evaluate_saved_layouts_highres(df_layouts, site, wt, existing_highres=None):
    """
    Re-evaluates each saved layout using:
    - 1 degree wind direction bins
    - 1 m/s wind speed bins

    This does not optimize the layouts again.
    It only evaluates the already saved x/y turbine coordinates.
    """
    df_layouts = df_layouts.copy()

    required_cols = ["script", "wake_model", "seed", "method", "turbine_id", "x", "y"]

    missing_cols = [col for col in required_cols if col not in df_layouts.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in df_layouts: {missing_cols}")

    group_cols = ["script", "wake_model", "seed", "opt_bin_label", "method"]

    if existing_highres is not None and not existing_highres.empty:
        existing_highres = existing_highres.copy()
    else:
        existing_highres = pd.DataFrame()

    highres_rows = []
    wf_model_cache = {}

    total_groups = df_layouts.groupby(group_cols).ngroups
    existing_keys = set()
    if not existing_highres.empty:
        existing_keys = set(
            map(tuple, existing_highres[group_cols].drop_duplicates().to_numpy())
        )

    missing_groups = 0
    for key, _ in df_layouts.groupby(group_cols):
        if key not in existing_keys:
            missing_groups += 1

    counter = 0

    print()
    print("Starting high-resolution AEP evaluation...")
    print(f"Evaluation wd bins: {len(wd_eval)}")
    print(f"Evaluation ws bins: {len(ws_eval)}")
    print(f"Number of layouts to evaluate: {total_groups}")
    print(f"Already cached layouts: {total_groups - missing_groups}")
    print(f"Missing layouts to evaluate: {missing_groups}")
    print()

    for (script, wake_model, seed, opt_bin_label, method), df_group in df_layouts.groupby(group_cols):
        counter += 1
        key = (script, wake_model, seed, opt_bin_label, method)

        if key in existing_keys:
            continue

        df_group = df_group.sort_values("turbine_id")
        layout_xy = df_group[["x", "y"]].to_numpy()

        if layout_xy.size == 0:
            continue

        if wake_model not in wf_model_cache:
            wf_model_cache[wake_model] = set_wake_model(wake_model, site, wt)

        wf_model = wf_model_cache[wake_model]

        print(
            f"[{counter}/{total_groups}] "
            f"Evaluating {script} | {wake_model} | seed {seed} | {method}"
        )

        fine_aep = calc_aep(
            wf_model,
            layout_xy,
            with_wake_loss=True,
            wd=wd_eval,
            ws=ws_eval
        )

        highres_rows.append({
            "script": script,
            "wake_model": wake_model,
            "seed": seed,
            "opt_bin_label": opt_bin_label,
            "method": method,
            "Fine AEP [GWh]": scalar_aep(fine_aep),
            "eval_wd_step_deg": 1,
            "eval_n_wd": len(wd_eval),
            "eval_ws_step": 1,
            "eval_ws_min": ws_eval.min(),
            "eval_ws_max": ws_eval.max(),
        })

    df_new_highres = pd.DataFrame(highres_rows)

    if existing_highres.empty:
        df_highres = df_new_highres
    elif df_new_highres.empty:
        df_highres = existing_highres
    else:
        df_highres = pd.concat([existing_highres, df_new_highres], ignore_index=True)

    if not df_highres.empty:
        df_highres = (
            df_highres
            .drop_duplicates(subset=group_cols, keep="last")
            .sort_values(group_cols)
            .reset_index(drop=True)
        )

    print()
    print("High-resolution AEP evaluation finished.")
    print()

    return df_highres


# ==================================================
# PLOTTING HELPERS
# ==================================================

def get_prevailing_wd_deg(site):
    ds = getattr(site, "ds", None)
    if ds is None or "Sector_frequency" not in ds:
        return None

    frequencies = np.asarray(ds["Sector_frequency"].values, dtype=float)
    wd_values = np.asarray(ds["wd"].values, dtype=float)
    if len(frequencies) == 0 or len(wd_values) == 0:
        return None

    return float(wd_values[int(np.nanargmax(frequencies))] % 360)


def layout_plot_coordinates(layout_xy, boundary, origin=None):
    if not use_local_layout_coordinates_km:
        return layout_xy, boundary, 1.0, "m"

    if origin is None:
        origin = boundary.min(axis=0)
    return (layout_xy - origin) / 1000.0, (boundary - origin) / 1000.0, 1 / 1000.0, "km"


def add_prevailing_wind_arrow(ax, boundary, wd_deg):
    if wd_deg is None:
        return

    xmin, xmax = boundary[:, 0].min(), boundary[:, 0].max()
    ymin, ymax = boundary[:, 1].min(), boundary[:, 1].max()
    width = xmax - xmin
    height = ymax - ymin

    # Meteorological wind directions are "from" directions.
    # The plotted vector points in the direction the wind travels.
    theta = np.deg2rad(wd_deg)
    ux = -np.sin(theta)
    uy = -np.cos(theta)

    if use_local_layout_coordinates_km:
        left_pad = boundary_pad / 1000.0
        end = np.array([
            xmin + 0.85 * left_pad,
            ymin + 0.24 * height,
        ])

        if wind_arrow_end_x_km is not None:
            end[0] = wind_arrow_end_x_km

        arrow_len = 0.75 * left_pad
        start = end - np.array([ux, uy]) * arrow_len
        if wind_arrow_start_x_km is not None and abs(ux) > 1e-9:
            arrow_len_from_x = (end[0] - wind_arrow_start_x_km) / ux
            start = end - np.array([ux, uy]) * arrow_len_from_x
    else:
        left_limit = wind_arrow_xmin if wind_arrow_xmin is not None else xmin - 0.25 * width
        arrow_len = 0.11 * width
        end = np.array([
            xmin - 0.08 * width,
            ymin + 0.22 * height,
        ])
        start = end - np.array([ux, uy]) * arrow_len
        start[0] = max(start[0], left_limit + 0.05 * width)

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "color": "tab:blue",
            "linewidth": 2.0,
            "mutation_scale": 14,
        },
        zorder=5
    )
    ax.text(
        start[0] + 0.08 * (boundary_pad / 1000.0 if use_local_layout_coordinates_km else width),
        start[1] - 0.07 * height,
        f"Prevailing wind\nfrom {wd_deg:.0f} deg",
        color="tab:blue",
        fontsize=10,
        ha="center",
        va="top",
        zorder=5
    )


def padded_limits_for_wind_arrow(boundary, wd_deg):
    xmin, xmax = boundary[:, 0].min(), boundary[:, 0].max()
    ymin, ymax = boundary[:, 1].min(), boundary[:, 1].max()

    if wd_deg is None:
        return xmin, xmax, ymin, ymax

    width = xmax - xmin
    height = ymax - ymin
    pad = 0.08 * max(width, height)

    theta = np.deg2rad(wd_deg)
    ux = -np.sin(theta)
    uy = -np.cos(theta)

    if ux > 0:
        xmin -= pad
    elif ux < 0:
        xmax += pad

    if uy > 0:
        ymin -= pad
    elif uy < 0:
        ymax += pad

    if wind_arrow_xmin is not None:
        xmin = wind_arrow_xmin

    xmax += 0.04 * width
    ymax += 0.08 * height

    if use_local_layout_coordinates_km:
        ymin = layout_axis_min_km
    else:
        ymin -= 0.08 * height

    return xmin, xmax, ymin, ymax


def plot_layout_on_axis(ax, layout_xy, boundary, wt, title, wind_arrow_wd_deg=None):
    layout_xy, boundary, length_scale, axis_unit = layout_plot_coordinates(layout_xy, boundary)
    bx = np.r_[boundary[:, 0], boundary[0, 0]]
    by = np.r_[boundary[:, 1], boundary[0, 1]]
    xmin, xmax = boundary[:, 0].min(), boundary[:, 0].max()
    ymin, ymax = boundary[:, 1].min(), boundary[:, 1].max()

    ax.plot(bx, by, color="black", linewidth=1.2)

    ax.scatter(
        layout_xy[:, 0],
        layout_xy[:, 1],
        c="red",
        marker="x",
        s=45,
        zorder=3
    )

    for x, y in layout_xy:
        circle = Circle(
            (x, y),
            radius=(spacing_D / 2) * wt.diameter() * length_scale,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            alpha=0.25,
            zorder=1
        )
        ax.add_patch(circle)

    ax.set_title(title)
    ax.set_xlabel(f"x [{axis_unit}]")
    ax.set_ylabel(f"y [{axis_unit}]")
    if show_prevailing_wind_arrow:
        add_prevailing_wind_arrow(ax, boundary, wind_arrow_wd_deg)
        xmin, xmax, ymin, ymax = padded_limits_for_wind_arrow(boundary, wind_arrow_wd_deg)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)


def plot_layout_movement_on_axis(ax, old_layout, new_layout, boundary, wt, title, moved_tol=0.1, wind_arrow_wd_deg=None):
    origin = boundary.min(axis=0)
    original_boundary = boundary
    old_layout, boundary, length_scale, axis_unit = layout_plot_coordinates(old_layout, original_boundary, origin=origin)
    new_layout, _, _, _ = layout_plot_coordinates(new_layout, original_boundary, origin=origin)
    moved_tol = moved_tol * length_scale

    bx = np.r_[boundary[:, 0], boundary[0, 0]]
    by = np.r_[boundary[:, 1], boundary[0, 1]]
    xmin, xmax = boundary[:, 0].min(), boundary[:, 0].max()
    ymin, ymax = boundary[:, 1].min(), boundary[:, 1].max()

    movement = np.linalg.norm(new_layout - old_layout, axis=1)
    moved = movement > moved_tol

    ax.plot(bx, by, color="black", linewidth=1.2, label="Boundary")

    ax.scatter(
        old_layout[:, 0],
        old_layout[:, 1],
        c="0.75",
        marker="o",
        s=35,
        label="Before GB",
        zorder=2
    )

    unmoved = ~moved

    ax.scatter(
        new_layout[unmoved, 0],
        new_layout[unmoved, 1],
        c="red",
        marker="x",
        s=45,
        label="Unmoved turbines",
        zorder=3
    )

    if moved.any():
        ax.scatter(
            new_layout[moved, 0],
            new_layout[moved, 1],
            c="tab:blue",
            marker="x",
            s=75,
            linewidths=2.0,
            label="Moved turbines",
            zorder=4
        )

        for (x_old, y_old), (x_new, y_new) in zip(old_layout[moved], new_layout[moved]):
            ax.annotate(
                "",
                xy=(x_new, y_new),
                xytext=(x_old, y_old),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "tab:blue",
                    "linewidth": 1.8,
                    "alpha": 0.9,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                zorder=3
            )

    for x, y in new_layout:
        circle = Circle(
            (x, y),
            radius=(spacing_D / 2) * wt.diameter() * length_scale,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            alpha=0.18,
            zorder=1
        )
        ax.add_patch(circle)

    movement_label = movement.max() / length_scale if length_scale != 0 else movement.max()
    ax.set_title(f"{title}\nMoved turbines: {int(moved.sum())}, max movement: {movement_label:.1f} m")
    ax.set_xlabel(f"x [{axis_unit}]")
    ax.set_ylabel(f"y [{axis_unit}]")
    if show_prevailing_wind_arrow:
        add_prevailing_wind_arrow(ax, boundary, wind_arrow_wd_deg)
        xmin, xmax, ymin, ymax = padded_limits_for_wind_arrow(boundary, wind_arrow_wd_deg)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="best")


def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def format_plot_title_line(wake_model, pipeline, seed, opt_bin_label, aep_value=None):
    aep_label = "Fine AEP" if use_highres_aep_for_comparison else "AEP"
    seed_label = f"Seed {seed}" if seed is not None else "All seeds"

    title = f"{wake_model} - {pipeline} - {seed_label} - {opt_bin_label}"
    if aep_value is not None and pd.notna(aep_value):
        title += f" - {float(aep_value):.1f} GWh {aep_label}"
    else:
        title += f" - {aep_label}"

    return title


# ==================================================
# LOAD CSV FILES
# ==================================================

df_summary = pd.concat(
    [
        load_latest_csv_per_bin("results_ss_summary_*.csv"),
        load_latest_csv_per_bin("results_rs_summary_*.csv"),
    ],
    ignore_index=True
)

df_layouts = pd.concat(
    [
        load_latest_csv_per_bin("results_ss_layouts_*.csv"),
        load_latest_csv_per_bin("results_rs_layouts_*.csv"),
    ],
    ignore_index=True
)

df_history = pd.concat(
    [
        load_latest_csv_per_bin("results_ss_history_*.csv"),
        load_latest_csv_per_bin("results_rs_history_*.csv"),
    ],
    ignore_index=True
)


# ==================================================
# PREPARE DATA
# ==================================================

df_summary = prepare_aep_column(df_summary)
df_layouts = prepare_aep_column(df_layouts)

df_summary = apply_ss_history_aep(df_summary, df_history)
df_layouts = apply_ss_history_aep(df_layouts, df_history)
df_history = add_interpolated_smartstart_history(df_history, df_summary)


# ==================================================
# LOAD SITE AND WIND TURBINE
# ==================================================

wt, _ = set_wt("HornsRev1")

boundary, x_ref, y_ref, site, n_wt, *_ = get_site(
    site_name="HornsRev1",
    wt=wt,
    boundary_pad=boundary_pad,
    x_points=20,
    y_points=20
)

plot_wind_arrow_wd_deg = prevailing_wd_deg
if plot_wind_arrow_wd_deg is None:
    plot_wind_arrow_wd_deg = get_prevailing_wd_deg(site)

if wind_arrow_plot_wd_deg is not None:
    plot_wind_arrow_wd_deg = wind_arrow_plot_wd_deg

if show_prevailing_wind_arrow and plot_wind_arrow_wd_deg is not None:
    print(f"Layout wind arrow: prevailing wind from {plot_wind_arrow_wd_deg:.0f} deg")


# ==================================================
# HIGH-RESOLUTION FINAL AEP EVALUATION
# ==================================================

df_highres_path = csv_dir / "final_layouts_highres_evaluation.csv"
if df_highres_path.exists():
    df_existing_highres = pd.read_csv(df_highres_path)
else:
    df_existing_highres = pd.DataFrame()

df_highres = evaluate_saved_layouts_highres(
    df_layouts,
    site,
    wt,
    existing_highres=df_existing_highres
)

df_highres.to_csv(df_highres_path, index=False)

print(f"Saved high-resolution AEP evaluation to:")
print(df_highres_path)
print()

merge_cols = ["script", "wake_model", "seed", "opt_bin_label", "method"]

# Save the original optimization/surrogate AEP before overwriting plot_AEP
df_summary["Optimization AEP [GWh]"] = df_summary["plot_AEP [GWh]"]
df_layouts["Optimization AEP [GWh]"] = df_layouts["plot_AEP [GWh]"]

df_summary = df_summary.merge(
    df_highres,
    on=merge_cols,
    how="left"
)

df_layouts = df_layouts.merge(
    df_highres,
    on=merge_cols,
    how="left"
)

if use_highres_aep_for_comparison:
    df_summary["plot_AEP [GWh]"] = np.where(
        df_summary["Fine AEP [GWh]"].notna(),
        df_summary["Fine AEP [GWh]"],
        df_summary["plot_AEP [GWh]"]
    )

    df_layouts["plot_AEP [GWh]"] = np.where(
        df_layouts["Fine AEP [GWh]"].notna(),
        df_layouts["Fine AEP [GWh]"],
        df_layouts["plot_AEP [GWh]"]
    )


# ==================================================
# LAYOUTS: BEST SEED PER PIPELINE
# ==================================================

layout_output_dir = base_dir / "Plots" / "PL_layouts"
if show_layout_plots or show_movement_plots:
    layout_output_dir.mkdir(exist_ok=True)

df_methods = df_summary[df_summary["method"].isin(method_order)].copy()

best_rows = (
    df_methods.sort_values("plot_AEP [GWh]", ascending=False)
    .groupby(["wake_model", "opt_bin_label", "method"], as_index=False)
    .first()
)

for (wake_model, opt_bin_label), df_wake in best_rows.groupby(["wake_model", "opt_bin_label"]):
    methods = [
        method for method in method_order
        if method in df_wake["method"].values
    ]

    for method in methods:
        row = df_wake[df_wake["method"] == method].iloc[0]
        seed = row["seed"]

        df_layout = df_layouts[
            (df_layouts["wake_model"] == wake_model) &
            (df_layouts["opt_bin_label"] == opt_bin_label) &
            (df_layouts["method"] == method) &
            (df_layouts["seed"] == seed)
        ].sort_values("turbine_id")

        layout_xy = df_layout[["x", "y"]].to_numpy()

        title = format_plot_title_line(
            wake_model=wake_model,
            pipeline=method,
            seed=seed,
            opt_bin_label=opt_bin_label,
            aep_value=row["plot_AEP [GWh]"],
        )

        fig, ax = plt.subplots(figsize=(8, 7))
        plot_layout_on_axis(
            ax,
            layout_xy,
            boundary,
            wt,
            title,
            wind_arrow_wd_deg=plot_wind_arrow_wd_deg
        )

        output_name = safe_filename(
            f"layout_{wake_model}_{opt_bin_label}_{method}_seed{seed}.png"
        )
        output_path = layout_output_dir / output_name
        plt.tight_layout()
        if show_layout_plots:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved layout figure: {output_path}")
            plt.show()
        else:
            plt.close(fig)


# ==================================================
# GB MOVEMENT LAYOUTS: 10 DEGREE BINS
# ==================================================

gb_movement_pairs = {
    "SS--GB": "SS",
    "SS--2S--GB": "SS--2S",
    "RS--GB": "RS",
    "RS--2S--GB": "RS--2S",
}

for (wake_model, opt_bin_label), df_wake in best_rows.groupby(["wake_model", "opt_bin_label"]):
    if opt_bin_label != "10 deg bins":
        continue

    for gb_method, base_method in gb_movement_pairs.items():
        if gb_method not in df_wake["method"].values:
            continue

        gb_row = df_wake[df_wake["method"] == gb_method].iloc[0]
        seed = gb_row["seed"]

        df_old = df_layouts[
            (df_layouts["wake_model"] == wake_model) &
            (df_layouts["opt_bin_label"] == opt_bin_label) &
            (df_layouts["method"] == base_method) &
            (df_layouts["seed"] == seed)
        ].sort_values("turbine_id")

        df_new = df_layouts[
            (df_layouts["wake_model"] == wake_model) &
            (df_layouts["opt_bin_label"] == opt_bin_label) &
            (df_layouts["method"] == gb_method) &
            (df_layouts["seed"] == seed)
        ].sort_values("turbine_id")

        if df_old.empty or df_new.empty:
            print(f"Skipping movement plot for {gb_method}, seed {seed}: missing layout data")
            continue

        old_layout = df_old[["x", "y"]].to_numpy()
        new_layout = df_new[["x", "y"]].to_numpy()

        fig, ax = plt.subplots(figsize=(8, 7))
        title = format_plot_title_line(
            wake_model=wake_model,
            pipeline=f"{base_method} to {gb_method}",
            seed=seed,
            opt_bin_label=opt_bin_label,
            aep_value=gb_row["plot_AEP [GWh]"],
        )
        plot_layout_movement_on_axis(
            ax,
            old_layout,
            new_layout,
            boundary,
            wt,
            title,
            wind_arrow_wd_deg=plot_wind_arrow_wd_deg
        )

        output_name = safe_filename(
            f"movement_{wake_model}_{opt_bin_label}_{base_method}_to_{gb_method}_seed{seed}.png"
        )
        output_path = layout_output_dir / output_name
        plt.tight_layout()
        if show_movement_plots:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved movement layout figure: {output_path}")
            plt.show()
        else:
            plt.close(fig)


# ==================================================
# FINAL COMPARISON TABLE
# ==================================================

df_table_source = df_summary[
    df_summary["method"].isin(["Current Horns Rev", *method_order])
].copy()

table_rows = []

for opt_bin_label, df_bin in df_table_source.groupby("opt_bin_label"):
    df_reference = df_bin[
        (df_bin["method"] == "Current Horns Rev") &
        (df_bin["script"] == "ResultsRS")
    ]

    if df_reference.empty:
        df_reference = df_bin[df_bin["method"] == "Current Horns Rev"]

    reference_aep = df_reference["plot_AEP [GWh]"].mean()

    table_rows.append({
        "Optimization bins": opt_bin_label,
        "Pipeline": "Reference",
        "Mean AEP [GWh]": reference_aep,
        "Best AEP [GWh]": np.nan,
        "Std. Dev. [GWh]": np.nan,
        "Delta AEP [GWh]": np.nan,
        "Delta AEP [%]": np.nan,
        "Runtime [min]": np.nan,
    })

    for method in method_order:
        df_method = df_bin[df_bin["method"] == method]

        if df_method.empty:
            continue

        mean_aep = df_method["plot_AEP [GWh]"].mean()
        best_aep = df_method["plot_AEP [GWh]"].max()
        std_aep = df_method["plot_AEP [GWh]"].std(ddof=1)
        delta_aep = mean_aep - reference_aep

        table_rows.append({
            "Optimization bins": opt_bin_label,
            "Pipeline": method,
            "Mean AEP [GWh]": mean_aep,
            "Best AEP [GWh]": best_aep,
            "Std. Dev. [GWh]": std_aep,
            "Delta AEP [GWh]": delta_aep,
            "Delta AEP [%]": 100.0 * delta_aep / reference_aep,
            "Runtime [min]": df_method["runtime_sec"].mean() / 60.0,
        })

df_table = pd.DataFrame(table_rows)

print()
if use_highres_aep_for_comparison:
    print("Final PL comparison table based on high-resolution AEP evaluation:")
else:
    print("Final PL comparison table based on optimization/surrogate AEP:")

print(df_table.round(3).to_string(index=False))
print()

print(df_table.to_latex(index=False, float_format="%.3f", na_rep="--"))


# ==================================================
# OPTIONAL: SAVE FINAL TABLE AS CSV
# ==================================================

df_table_path = csv_dir / "final_pipeline_comparison_table_highres.csv"
df_table.to_csv(df_table_path, index=False)

print()
print(f"Saved final comparison table to:")
print(df_table_path)
print()


# ==================================================
# AEP DEVELOPMENT OVER TIME - INDIVIDUAL PIPELINE RUNS
# ==================================================
# This still uses the original history AEP, because history is the optimization process.
# It should not be replaced with high-resolution AEP.

dev_x_min = -250.0
dev_x_max_by_script = {
    "ResultsSS": 30000,
    "ResultsRS": 11000,
}
dev_y_min = 70
dev_y_max = 73.5
dev_y_pad = max((dev_y_max - dev_y_min) * 0.05, 0.05)

if show_development_plots:
    development_groups = df_history.groupby(["script", "wake_model", "opt_bin_label"])
else:
    development_groups = []

stage_markers = {
    "SmartStart interpolated": ("D", "SmartStart start"),
    "RandomSearch": ("o", "RandomSearch start"),
    "Two-step": ("s", "Two-step start"),
    "Gradient": ("^", "Gradient start"),
}

for (script, wake_model, opt_bin_label), df_wake in development_groups:
    plt.figure(figsize=(10, 6))

    for seed, df_seed in df_wake.groupby("seed"):
        df_seed = df_seed.sort_values("elapsed_sec")

        line, = plt.plot(
            df_seed["elapsed_sec"],
            df_seed["AEP [GWh]"],
            linewidth=1.5,
            alpha=0.85,
            label=f"Seed {seed}"
        )

        line_color = line.get_color()
        for phase, (marker, _) in stage_markers.items():
            df_phase = df_seed[df_seed["phase"].eq(phase)]
            if df_phase.empty:
                continue

            stage_start = df_phase.sort_values("elapsed_sec").iloc[0]
            plt.scatter(
                stage_start["elapsed_sec"],
                stage_start["AEP [GWh]"],
                marker=marker,
                s=70,
                facecolors="white",
                edgecolors=line_color,
                linewidths=1.7,
                zorder=5
            )

    pipeline_name = "SmartStart" if script == "ResultsSS" else "RandomSearch"
    layout_style_title = format_plot_title_line(
        wake_model=wake_model,
        pipeline=f"{pipeline_name} pipeline runs",
        seed=None,
        opt_bin_label=opt_bin_label,
        aep_value=None,
    )
    development_title = (
        f"AEP development over time - {wake_model} "
        f"({pipeline_name} pipeline runs, {opt_bin_label})"
    )

    plt.title(
        f"{layout_style_title}\n{development_title}"
    )
    plt.xlabel("Elapsed time [s]")
    plt.ylabel("Optimization AEP [GWh]")
    plt.xlim(dev_x_min, dev_x_max_by_script.get(script, df_wake["elapsed_sec"].max()))
    plt.ylim(dev_y_min - dev_y_pad, dev_y_max + dev_y_pad)
    plt.grid(True, alpha=0.3)

    if show_prevailing_wind_arrow and plot_wind_arrow_wd_deg is not None:
        plt.gca().text(
            0.98,
            0.96,
            f"Prevailing wind: {plot_wind_arrow_wd_deg:.0f} deg",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="tab:blue",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "tab:blue",
                "alpha": 0.85,
            },
        )

    seed_legend = plt.legend(fontsize=8, loc="lower right")
    plt.gca().add_artist(seed_legend)

    active_phases = set(df_wake["phase"].unique())
    stage_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=7,
            linestyle="None",
            label=label
        )
        for phase, (marker, label) in stage_markers.items()
        if phase in active_phases
    ]
    if stage_handles:
        plt.legend(handles=stage_handles, fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.show()


# ==================================================
# AEP VS RUNTIME
# ==================================================

df_plot = df_summary[df_summary["method"].isin(method_order)].copy()

df_mean = (
    df_plot.groupby(["wake_model", "opt_bin_label", "method"], as_index=False)
    .agg(
        aep_mean=("plot_AEP [GWh]", "mean"),
        aep_std=("plot_AEP [GWh]", "std"),
        runtime_mean=("runtime_sec", "mean"),
        runtime_std=("runtime_sec", "std"),
        n_seeds=("seed", "nunique")
    )
    .fillna({"aep_std": 0.0, "runtime_std": 0.0})
)

plt.figure(figsize=(10, 6))

label_offsets = {
    "SS": (7, 5),
    "SS--2S": (8, 12),
    "SS--GB": (7, -14),
    "SS--2S--GB": (8, -18),
    "RS": (7, 5),
    "RS--2S": (8, 12),
    "RS--GB": (7, -14),
    "RS--2S--GB": (8, -18),
}

for method in method_order:
    df_method = df_mean[df_mean["method"] == method]

    if df_method.empty:
        continue

    for _, row in df_method.iterrows():
        if len(df_mean["wake_model"].unique()) == 1:
            label = f"{method} ({row['opt_bin_label']})"
        else:
            label = f"{row['wake_model']} - {method} ({row['opt_bin_label']})"

        plt.errorbar(
            row["runtime_mean"] / 60.0,
            row["aep_mean"],
            xerr=row["runtime_std"] / 60.0,
            yerr=row["aep_std"],
            fmt="o",
            markersize=8,
            capsize=5,
            color=method_colors.get(method, None),
            label=label
        )

        plt.annotate(
            method,
            (row["runtime_mean"] / 60.0, row["aep_mean"]),
            textcoords="offset points",
            xytext=label_offsets.get(method, (7, 5)),
            fontsize=9
        )

for method in method_order:
    df_seed = df_plot[df_plot["method"] == method]

    if df_seed.empty:
        continue

    plt.scatter(
        df_seed["runtime_sec"] / 60.0,
        df_seed["plot_AEP [GWh]"],
        s=28,
        alpha=0.35,
        color=method_colors.get(method, None)
    )

if use_highres_aep_for_comparison:
    plt.title("High-resolution AEP vs runtime - pipeline comparison")
    plt.ylabel("Fine AEP [GWh]")
else:
    plt.title("Optimization AEP vs runtime - pipeline comparison")
    plt.ylabel("Optimization AEP [GWh]")

plt.xlabel("Runtime [min]")
plt.grid(True, alpha=0.3)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout(rect=[0, 0, 0.82, 1])
if show_runtime_plot:
    plt.show()
else:
    plt.close()


# ==================================================
# OPTIONAL: COMPARE OPTIMIZATION AEP AND FINE AEP
# ==================================================

if "Fine AEP [GWh]" in df_summary.columns:
    df_compare = df_summary[
        df_summary["method"].isin(["Current Horns Rev", *method_order])
    ].copy()

    df_compare = df_compare[
        df_compare["Fine AEP [GWh]"].notna()
    ].copy()

    if not df_compare.empty:
        df_compare["Fine minus Optimization AEP [GWh]"] = (
            df_compare["Fine AEP [GWh]"] -
            df_compare["Optimization AEP [GWh]"]
        )

        compare_path = csv_dir / "optimization_vs_highres_aep_comparison.csv"
        df_compare.to_csv(compare_path, index=False)

        print()
        print("Saved optimization vs high-resolution AEP comparison to:")
        print(compare_path)
        print()

        print("Optimization vs high-resolution AEP comparison:")
        cols_to_print = [
            "script",
            "wake_model",
            "seed",
            "opt_bin_label",
            "method",
            "Optimization AEP [GWh]",
            "Fine AEP [GWh]",
            "Fine minus Optimization AEP [GWh]",
        ]

        print(
            df_compare[cols_to_print]
            .round(3)
            .to_string(index=False)
        )
