import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_arrow_points(idx: int, xs: np.ndarray, ys: np.ndarray):
    """Finds start and end points for drawing a directional arrow starting near the given index.

    Ensures that the points are distinct and point chronologically along the curve.
    """
    n = len(xs)
    if idx < 0 or idx >= n - 1:
        return None, None

    # Try to find a downstream point (next_idx > idx) that is different from idx
    next_idx = idx + 1
    while next_idx < n and xs[next_idx] == xs[idx] and ys[next_idx] == ys[idx]:
        next_idx += 1

    if next_idx < n:
        return (xs[idx], ys[idx]), (xs[next_idx], ys[next_idx])

    # If no downstream point is different, try to find an upstream point (prev_idx < idx)
    prev_idx = idx - 1
    while prev_idx >= 0 and xs[prev_idx] == xs[idx] and ys[prev_idx] == ys[idx]:
        prev_idx -= 1

    if prev_idx >= 0:
        return (xs[prev_idx], ys[prev_idx]), (xs[idx], ys[idx])

    return None, None


# Identify the CSV path relative to this file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "merged_data_L303.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "data", "merged_data_L303.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "..", "data", "merged_data_L303.csv")),
]

csv_path = None
for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    csv_path = "data/merged_data_L303.csv"


def plot_time_trend(run_id: int, output_filename: str) -> str:
    """Filters the merged aerodynamic CSV data for the given RunNo and plots Cl, Cdp, and Cm over time.

    The plot consists of 3 vertically stacked subplots sharing the X-axis ('Time (sec)').
    Saves the interactive figure to plot.json using Plotly.

    Args:
        run_id: The RunNo integer to filter the data.
        output_filename: The file path parameter (preserved for signature compatibility).

    Returns:
        A success message string.
    """
    if not os.path.exists(csv_path):
        return f"Error: CSV data file not found at {csv_path}."

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV file: {e}"

    # Ensure RunNo is treated numerically to match against run_id
    df["RunNo"] = pd.to_numeric(df["RunNo"], errors="coerce")
    df_filtered = df[df["RunNo"] == run_id]

    if df_filtered.empty:
        return f"No data found for RunNo {run_id}."

    # Sort by Time (sec) to ensure the line plot is ordered correctly
    df_filtered = df_filtered.sort_values(by="Time (sec)")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"Run {run_id} - Lift Coefficient (Cl)",
            f"Run {run_id} - Parasitic Drag Coefficient (Cdp)",
            f"Run {run_id} - Pitching Moment Coefficient (Cm)",
        ),
    )

    # Plot Cl
    fig.add_trace(
        go.Scatter(
            x=df_filtered["Time (sec)"],
            y=df_filtered["Cl"],
            mode="lines",
            line=dict(color="royalblue", width=2),
            name="Cl",
            hovertemplate=f"<b>Run {run_id}</b><br>Time: %{{x:.2f}} s<br>Cl: %{{y:.4f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Plot Cdp
    fig.add_trace(
        go.Scatter(
            x=df_filtered["Time (sec)"],
            y=df_filtered["Cdp"],
            mode="lines",
            line=dict(color="darkorange", width=2),
            name="Cdp",
            hovertemplate=f"<b>Run {run_id}</b><br>Time: %{{x:.2f}} s<br>Cdp: %{{y:.4f}}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Plot Cm
    fig.add_trace(
        go.Scatter(
            x=df_filtered["Time (sec)"],
            y=df_filtered["Cm"],
            mode="lines",
            line=dict(color="forestgreen", width=2),
            name="Cm",
            hovertemplate=f"<b>Run {run_id}</b><br>Time: %{{x:.2f}} s<br>Cm: %{{y:.4f}}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Aerodynamic Coefficients Time Trend: Run {run_id}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=60, r=40, t=80, b=60),
    )

    fig.update_yaxes(title_text="Cl", row=1, col=1, showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig.update_yaxes(title_text="Cdp", row=2, col=1, showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig.update_yaxes(title_text="Cm", row=3, col=1, showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig.update_xaxes(title_text="Time (sec)", row=3, col=1)

    fig.write_json("plot.json")
    return "Plot successfully generated using Plotly and saved as plot.json."


def plot_runs_comparison(run_ids: list[int], output_filename: str) -> str:
    """Filters the merged aerodynamic CSV data for the provided RunNo list and plots AOA vs Cl, Cdp, and Cm.

    The plot consists of 3 rows of subplots ('AOA (deg)' vs 'Cl', 'Cdp', 'Cm' respectively),
    plotting each run as a scatter plot with distinct colors, up-sweep/down-sweep hysteresis separation,
    directional markers, and interactive hover tooltips. Saves the figure to plot.json.

    Args:
        run_ids: A list of RunNo integers to filter the data.
        output_filename: The file path parameter (preserved for signature compatibility).

    Returns:
        A success message string.
    """
    if not os.path.exists(csv_path):
        return f"Error: CSV data file not found at {csv_path}."

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV file: {e}"

    # Ensure RunNo is treated numerically
    df["RunNo"] = pd.to_numeric(df["RunNo"], errors="coerce")

    # Filter for the provided RunNos
    df_filtered = df[df["RunNo"].isin(run_ids)]
    if df_filtered.empty:
        return f"No data found for the specified runs: {run_ids}."

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "AOA vs Lift Coefficient (Cl)",
            "AOA vs Drag Coefficient (Cdp)",
            "AOA vs Pitching Moment Coefficient (Cm)",
        ),
    )

    # A collection of distinct premium colors
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for idx, run_id in enumerate(run_ids):
        run_df = df_filtered[df_filtered["RunNo"] == run_id]
        if run_df.empty:
            continue
        color = colors[idx % len(colors)]

        # Extract values from the first row of this run to calculate M and Re
        first_row = run_df.iloc[0]
        temp = first_row.get("Temperature")
        airspeed = first_row.get("Airspeed")
        rey = first_row.get("Rey")

        import math
        mach_str = "Unknown"
        if pd.notna(temp) and pd.notna(airspeed) and temp > 0:
            a = math.sqrt(1.4 * 287.05 * float(temp))
            mach = float(airspeed) / a
            mach_str = f"{mach:.1f}"

        re_str = "Unknown"
        if pd.notna(rey):
            re_val = float(rey) * 1e6
            re_str = f"{re_val:.1e}".replace("+0", "").replace("+", "")

        label = f"Run {run_id} (M={mach_str}, Re={re_str})"

        # Sort chronologically by Time (sec)
        run_df = run_df.sort_values(by="Time (sec)")

        # Find local minimums of the AOA array to isolate a single pitching cycle
        aoa_all = run_df["AOA (deg)"].values
        minima_indices = []
        window = 10
        for i in range(len(aoa_all)):
            start = max(0, i - window)
            end = min(len(aoa_all), i + window + 1)
            if aoa_all[i] == np.min(aoa_all[start:end]):
                if not minima_indices or (i - minima_indices[-1] > window):
                    minima_indices.append(i)

        if len(minima_indices) >= 2:
            run_df = run_df.iloc[minima_indices[0] : minima_indices[1] + 1]

        aoa = run_df["AOA (deg)"].values
        cl = run_df["Cl"].values
        cdp = run_df["Cdp"].values
        cm = run_df["Cm"].values

        if len(aoa) > 0:
            # 1. Data Splitting (Chronological separation)
            max_aoa_idx = np.argmax(aoa)

            # Up-sweep segment (increasing AOA)
            up_aoa = aoa[:max_aoa_idx + 1]
            up_cl = cl[:max_aoa_idx + 1]
            up_cdp = cdp[:max_aoa_idx + 1]
            up_cm = cm[:max_aoa_idx + 1]

            # Down-sweep segment (decreasing AOA)
            down_aoa = aoa[max_aoa_idx:]
            down_cl = cl[max_aoa_idx:]
            down_cdp = cdp[max_aoa_idx:]
            down_cm = cm[max_aoa_idx:]

            # 2. Plotting (Color-coding)
            label_up = f"Run {run_id} Up-sweep"
            label_down = f"Run {run_id} Down-sweep"

            hover_up = (
                f"<b>Run {run_id} (Up-sweep)</b><br>"
                f"M: {mach_str}, Re: {re_str}<br>"
                "AOA: %{x:.2f}°<br>"
                "Value: %{y:.4f}<extra></extra>"
            )
            hover_down = (
                f"<b>Run {run_id} (Down-sweep)</b><br>"
                f"M: {mach_str}, Re: {re_str}<br>"
                "AOA: %{x:.2f}°<br>"
                "Value: %{y:.4f}<extra></extra>"
            )

            # Plot Cl (row 1)
            fig.add_trace(
                go.Scatter(
                    x=up_aoa,
                    y=up_cl,
                    mode="lines+markers",
                    name=label_up,
                    legendgroup=f"run_{run_id}_up",
                    line=dict(color=color, width=2),
                    marker=dict(size=5, color=color),
                    hovertemplate=hover_up,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=down_aoa,
                    y=down_cl,
                    mode="lines+markers",
                    name=label_down,
                    legendgroup=f"run_{run_id}_down",
                    line=dict(color=color, width=2, dash="dash"),
                    marker=dict(size=5, color=color),
                    hovertemplate=hover_down,
                ),
                row=1,
                col=1,
            )

            # Plot Cdp (row 2)
            fig.add_trace(
                go.Scatter(
                    x=up_aoa,
                    y=up_cdp,
                    mode="lines+markers",
                    name=label_up,
                    legendgroup=f"run_{run_id}_up",
                    showlegend=False,
                    line=dict(color=color, width=2),
                    marker=dict(size=5, color=color),
                    hovertemplate=hover_up,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=down_aoa,
                    y=down_cdp,
                    mode="lines+markers",
                    name=label_down,
                    legendgroup=f"run_{run_id}_down",
                    showlegend=False,
                    line=dict(color=color, width=2, dash="dash"),
                    marker=dict(size=5, color=color),
                    hovertemplate=hover_down,
                ),
                row=2,
                col=1,
            )

            # Plot Cm (row 3)
            fig.add_trace(
                go.Scatter(
                    x=up_aoa,
                    y=up_cm,
                    mode="lines+markers",
                    name=label_up,
                    legendgroup=f"run_{run_id}_up",
                    showlegend=False,
                    line=dict(color=color, width=2),
                    marker=dict(size=5, color=color),
                    hovertemplate=hover_up,
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=down_aoa,
                    y=down_cm,
                    mode="lines+markers",
                    name=label_down,
                    legendgroup=f"run_{run_id}_down",
                    showlegend=False,
                    line=dict(color=color, width=2, dash="dash"),
                    marker=dict(size=5, color=color),
                    hovertemplate=hover_down,
                ),
                row=3,
                col=1,
            )

            # 3. Directional Arrow Indicators (On-path indicators)
            idx_up = np.argmin(np.abs(up_aoa - 15.0))
            idx_down = np.argmin(np.abs(down_aoa - 25.0))

            def annotate_direction(row_num, xs, ys, start_idx, color_arrow):
                n = len(xs)
                if start_idx < 0 or start_idx >= n - 1:
                    return
                next_idx = start_idx + 1
                while next_idx < n and xs[next_idx] == xs[start_idx] and ys[next_idx] == ys[start_idx]:
                    next_idx += 1
                if next_idx < n:
                    xref = "x" if row_num == 1 else f"x{row_num}"
                    yref = "y" if row_num == 1 else f"y{row_num}"
                    fig.add_annotation(
                        x=xs[next_idx],
                        y=ys[next_idx],
                        ax=xs[start_idx],
                        ay=ys[start_idx],
                        xref=xref,
                        yref=yref,
                        axref=xref,
                        ayref=yref,
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor=color_arrow,
                    )

            # Annotate Up-sweep arrows
            annotate_direction(1, up_aoa, up_cl, idx_up, color)
            annotate_direction(2, up_aoa, up_cdp, idx_up, color)
            annotate_direction(3, up_aoa, up_cm, idx_up, color)

            # Annotate Down-sweep arrows
            annotate_direction(1, down_aoa, down_cl, idx_down, color)
            annotate_direction(2, down_aoa, down_cdp, idx_down, color)
            annotate_direction(3, down_aoa, down_cm, idx_down, color)

    # Configure axes details & Figure Formatting
    fig.update_yaxes(title_text="Cl (Lift)", row=1, col=1, showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig.update_yaxes(title_text="Cdp (Drag)", row=2, col=1, showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig.update_yaxes(title_text="Cm (Pitching Moment)", row=3, col=1, showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig.update_xaxes(title_text="AOA (deg)", row=3, col=1)

    if len(run_ids) == 1:
        main_title = f"<b>Dynamic Lift Hysteresis Comparison (Run {run_ids[0]})</b>"
    else:
        main_title = f"<b>Dynamic Lift Hysteresis Comparison (Runs {', '.join(map(str, run_ids))})</b>"

    fig.update_layout(
        title=dict(text=main_title, x=0.5, xanchor="center", font=dict(size=15)),
        template="plotly_white",
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
        hovermode="closest",
        margin=dict(l=60, r=40, t=80, b=60),
    )

    fig.write_json("plot.json")
    return "Plot successfully generated using Plotly and saved as plot.json."
