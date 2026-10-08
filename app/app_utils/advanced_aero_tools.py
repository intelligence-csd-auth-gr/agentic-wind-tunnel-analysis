import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pcolors

# Identify the file paths relative to this file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_csv_paths = [
    os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "merged_data_L303.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "data", "merged_data_L303.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "..", "data", "merged_data_L303.csv")),
]
possible_xc_paths = [
    os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "xc_locations.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "data", "xc_locations.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "..", "data", "xc_locations.csv")),
]

csv_path = None
for path in possible_csv_paths:
    if os.path.exists(path):
        csv_path = path
        break
if csv_path is None:
    csv_path = "data/merged_data_L303.csv"

xc_path = None
for path in possible_xc_paths:
    if os.path.exists(path):
        xc_path = path
        break
if xc_path is None:
    xc_path = "data/xc_locations.csv"


def plot_pressure_distribution(run_ids: list[int], target_alphas: list[float], output_filename: str = "") -> str:
    """Finds the row with AOA closest to each target_alpha for each run and plots Cp vs x/c geometry using Plotly.

    Inverts the Y-axis (standard aerodynamics convention), adds interactive hover tooltips,
    and saves the figure to plot.json.

    Args:
        run_ids: A list of RunNo integers to filter the data.
        target_alphas: A list of target angles of attack (AOA) in degrees.
        output_filename: The file path parameter (preserved for signature compatibility).

    Returns:
        A success message string.
    """
    if not os.path.exists(csv_path):
        return f"Error: CSV data file not found at {csv_path}."
    if not os.path.exists(xc_path):
        return f"Error: Geometry file not found at {xc_path}."

    if isinstance(target_alphas, (int, float)):
        target_alphas = [float(target_alphas)]
    if isinstance(run_ids, (int, str)):
        run_ids = [int(run_ids)]

    try:
        df = pd.read_csv(csv_path)
        xc_df = pd.read_csv(xc_path)
    except Exception as e:
        return f"Error reading data files: {e}"

    x = xc_df["x/c"].values
    if len(x) != 63:
        return f"Error: Geometry coordinates length is {len(x)}, expected 63."

    # Ensure RunNo and AOA are treated numerically
    df["RunNo"] = pd.to_numeric(df["RunNo"], errors="coerce")
    df["AOA (deg)"] = pd.to_numeric(df["AOA (deg)"], errors="coerce")

    fig = go.Figure()
    plotted_curves = 0
    sep_count = 0
    cp_cols = [f"Cp_{i}" for i in range(1, 64)]

    # A collection of distinct premium colors
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#3366cc", "#dc3912", "#ff9900", "#109618", "#990099"
    ]

    for run_id in run_ids:
        df_run = df[df["RunNo"] == run_id]
        if df_run.empty:
            continue

        for target_alpha in target_alphas:
            # Find row closest to target_alpha
            closest_idx = (df_run["AOA (deg)"] - target_alpha).abs().idxmin()
            row = df_run.loc[closest_idx]
            actual_aoa = row["AOA (deg)"]

            cp = row[cp_cols].values.astype(float)
            color = colors[plotted_curves % len(colors)]
            cl = row["Cl"]
            cdp = row["Cdp"]
            cm = row["Cm"]
            label = f"Run {run_id} - AOA: {actual_aoa:.2f}° (C_L: {cl:.2f}, C_D: {cdp:.4f}, C_m: {cm:.4f})"

            hovertemplate = (
                f"<b>Run {run_id} - AOA: {actual_aoa:.2f}°</b><br>"
                "x/c: %{x:.4f}<br>"
                "Cp: %{y:.4f}<br>"
                f"C<sub>L</sub>: {cl:.2f}<br>"
                f"C<sub>D</sub>: {cdp:.4f}<br>"
                f"C<sub>m</sub>: {cm:.4f}<extra></extra>"
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=cp,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    hovertemplate=hovertemplate,
                )
            )

            # Isolate suction side data (upper surface: index 0 to 31)
            x_suction = x[:32]
            cp_suction = cp[:32]

            # Detect flow separation
            sep_point = detect_flow_separation(x_suction, cp_suction)
            if sep_point is not None:
                y_pos = 0.95 - 0.06 * (sep_count % 10)
                fig.add_vline(
                    x=sep_point,
                    line_width=1.5,
                    line_dash="dash",
                    line_color=color,
                    opacity=0.8,
                )
                fig.add_annotation(
                    x=sep_point,
                    y=y_pos,
                    yref="paper",
                    text=f"Sep (Run {run_id}, AOA {actual_aoa:.1f}°): x/c = {sep_point:.2f}",
                    showarrow=False,
                    xanchor="left",
                    xshift=5,
                    font=dict(color=color, size=10),
                )
                sep_count += 1

            plotted_curves += 1

    if plotted_curves == 0:
        return f"Error: No data found for the specified runs: {run_ids}."

    if len(target_alphas) == 1:
        title_text = f"<b>Pressure Distribution (Target AOA: {target_alphas[0]:.2f}°)</b>"
    else:
        aoa_strs = [f"{a:.1f}°" for a in target_alphas]
        aoa_summary = ", ".join(aoa_strs[:4]) + ("..." if len(aoa_strs) > 4 else "")
        title_text = f"<b>Pressure Distribution Comparison (Target AOAs: {aoa_summary})</b>"

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        xaxis=dict(
            title="x/c (Chordwise Position)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            zeroline=True,
            zerolinecolor="rgba(150, 150, 150, 0.5)",
        ),
        yaxis=dict(
            title="Cp (Pressure Coefficient)",
            autorange="reversed",  # Invert Y-axis: negative Cp (suction) upwards
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            zeroline=True,
            zerolinecolor="rgba(150, 150, 150, 0.5)",
        ),
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
        margin=dict(l=60, r=40, t=60, b=60),
    )

    fig.write_json("plot.json")
    return "Plot successfully generated using Plotly and saved as plot.json."


def compare_integrated_vs_reported(run_id: int, output_filename: str) -> str:
    """Calculates integrated Cl, Cdp, and Cm from the 63 Cp columns and compares them with reported values.

    Generates a 3-subplot comparison figure with a y=x reference line and saves the figure to plot.json.

    Args:
        run_id: The RunNo integer to filter the data.
        output_filename: The file path parameter (preserved for signature compatibility).

    Returns:
        A success message string.
    """
    if not os.path.exists(csv_path):
        return f"Error: CSV data file not found at {csv_path}."
    if not os.path.exists(xc_path):
        return f"Error: Geometry file not found at {xc_path}."

    try:
        df = pd.read_csv(csv_path)
        xc_df = pd.read_csv(xc_path)
    except Exception as e:
        return f"Error reading data files: {e}"

    x = xc_df["x/c"].values
    if len(x) != 63:
        return f"Error: Geometry coordinates length is {len(x)}, expected 63."

    # Ensure RunNo is treated numerically
    df["RunNo"] = pd.to_numeric(df["RunNo"], errors="coerce")
    df_filtered = df[df["RunNo"] == run_id]

    if df_filtered.empty:
        return f"Error: No data found for RunNo {run_id}."

    dx = np.diff(x)
    x_mid = (x[:-1] + x[1:]) / 2

    cl_rep = df_filtered["Cl"].values
    cdp_rep = df_filtered["Cdp"].values
    cm_rep = df_filtered["Cm"].values
    alpha = np.radians(df_filtered["AOA (deg)"].values)

    cp_cols = [f"Cp_{i}" for i in range(1, 64)]
    cps = df_filtered[cp_cols].values

    cl_calc = []
    cdp_calc = []
    cm_calc = []

    for idx, cp in enumerate(cps):
        cp_mid = (cp[:-1] + cp[1:]) / 2
        cn = -np.sum(cp_mid * dx)
        cl = cn
        cdp = cn * np.sin(alpha[idx])
        cm = np.sum(cp_mid * (x_mid - 0.25) * dx)
        cl_calc.append(cl)
        cdp_calc.append(cdp)
        cm_calc.append(cm)

    cl_calc = np.array(cl_calc)
    cdp_calc = np.array(cdp_calc)
    cm_calc = np.array(cm_calc)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "<b>Cl Comparison</b>",
            "<b>Cdp Comparison</b>",
            "<b>Cm Comparison</b>",
        ),
    )

    # Subplot 1: Cl
    all_cl = np.concatenate([cl_rep, cl_calc])
    min_cl, max_cl = float(all_cl.min()), float(all_cl.max())
    fig.add_trace(
        go.Scatter(
            x=cl_rep,
            y=cl_calc,
            mode="markers",
            marker=dict(color="royalblue", size=6, opacity=0.7),
            name="Cl Data",
            hovertemplate="<b>Cl Comparison</b><br>Reported Cl: %{x:.4f}<br>Calculated Cl: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[min_cl, max_cl],
            y=[min_cl, max_cl],
            mode="lines",
            line=dict(color="red", dash="dash", width=1.5),
            name="y=x",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Cdp
    all_cdp = np.concatenate([cdp_rep, cdp_calc])
    min_cdp, max_cdp = float(all_cdp.min()), float(all_cdp.max())
    fig.add_trace(
        go.Scatter(
            x=cdp_rep,
            y=cdp_calc,
            mode="markers",
            marker=dict(color="darkorange", size=6, opacity=0.7),
            name="Cdp Data",
            hovertemplate="<b>Cdp Comparison</b><br>Reported Cdp: %{x:.4f}<br>Calculated Cdp: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[min_cdp, max_cdp],
            y=[min_cdp, max_cdp],
            mode="lines",
            line=dict(color="red", dash="dash", width=1.5),
            name="y=x",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    # Subplot 3: Cm
    all_cm = np.concatenate([cm_rep, cm_calc])
    min_cm, max_cm = float(all_cm.min()), float(all_cm.max())
    fig.add_trace(
        go.Scatter(
            x=cm_rep,
            y=cm_calc,
            mode="markers",
            marker=dict(color="forestgreen", size=6, opacity=0.7),
            name="Cm Data",
            hovertemplate="<b>Cm Comparison</b><br>Reported Cm: %{x:.4f}<br>Calculated Cm: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=[min_cm, max_cm],
            y=[min_cm, max_cm],
            mode="lines",
            line=dict(color="red", dash="dash", width=1.5),
            name="y=x",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=3,
    )

    # Extract Reynolds, Temperature, and Airspeed from merged data
    first_row = df_filtered.iloc[0]
    temp_val = first_row.get("Temperature")
    airspeed_val = first_row.get("Airspeed")
    rey_val = first_row.get("Rey")

    temp_str = f"{float(temp_val):.2f} K" if pd.notna(temp_val) else "Unknown"
    airspeed_str = f"{float(airspeed_val):.2f} m/s" if pd.notna(airspeed_val) else "Unknown"
    if pd.notna(rey_val):
        try:
            re_val = float(rey_val) * 1e6
            rey_str = f"{re_val:.2e} (Rey: {float(rey_val):.2f})"
        except (ValueError, TypeError):
            rey_str = str(rey_val)
    else:
        rey_str = "Unknown"

    info_text = (
        f"<b>Reynolds (Re):</b> {rey_str}<br>"
        f"<b>Temperature:</b> {temp_str}<br>"
        f"<b>Airspeed:</b> {airspeed_str}"
    )

    fig.update_xaxes(title_text="Reported Cl", row=1, col=1, showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig.update_yaxes(title_text="Calculated Cl", row=1, col=1, showgrid=True, gridcolor="rgba(200,200,200,0.3)")

    fig.update_xaxes(title_text="Reported Cdp", row=1, col=2, showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig.update_yaxes(title_text="Calculated Cdp", row=1, col=2, showgrid=True, gridcolor="rgba(200,200,200,0.3)")

    fig.update_xaxes(title_text="Reported Cm", row=1, col=3, showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig.update_yaxes(title_text="Calculated Cm", row=1, col=3, showgrid=True, gridcolor="rgba(200,200,200,0.3)")

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.18,
        text=info_text,
        showarrow=False,
        align="right",
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
        font=dict(size=9),
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Run {run_id} - Integrated vs Reported Coefficients</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=60, r=40, t=100, b=60),
    )

    fig.write_json("plot.json")
    return "Plot successfully generated using Plotly and saved as plot.json."


def detect_flow_separation(xc: np.ndarray, cp_suction: np.ndarray, threshold: float = 0.5) -> float | None:
    """Detects the onset of flow separation on the upper surface of the airfoil using the Cp gradient plateau method.

    Aerodynamic reasoning:
    On the suction (upper) surface of an airfoil, the flow typically accelerates past the leading edge, reaching a low-pressure
    peak (suction peak). Downstream, it enters an adverse pressure gradient region where velocity decreases and pressure
    increases (dCp/dx > 0). If this adverse pressure gradient is sufficiently strong, the boundary layer flow loses momentum,
    decelerates to a standstill, and separates from the surface. In the separated flow region, the pressure coefficient (Cp)
    tends to flatten out, forming a characteristic plateau where the spatial derivative dCp/dx is close to zero.
    By finding the first location downstream of the leading edge region (x/c > 0.4) where the absolute value of the gradient
    falls below a threshold (e.g., 0.5) and remains low for at least 3 consecutive points, we can identify the start of this
    Cp plateau, which serves as a proxy for the onset of flow separation. The leading edge region is ignored because flow
    acceleration and transition near the suction peak can also exhibit low or zero pressure gradients that do not indicate
    separation.

    Args:
        xc: A 1D numpy array of chordwise coordinates (x/c).
        cp_suction: A 1D numpy array of pressure coefficients (Cp) on the upper/suction surface.
        threshold: The gradient magnitude threshold below which Cp is considered to be plateaued. Default is 0.5.

    Returns:
        The x/c coordinate where the separation plateau begins, or None if no separation is detected.
    """
    xc = np.asarray(xc)
    cp_suction = np.asarray(cp_suction)

    if len(xc) != len(cp_suction):
        raise ValueError("xc and cp_suction must be of the same length.")

    # Filter out consecutive duplicate coordinate points to prevent division by zero in np.gradient
    if len(xc) > 1:
        keep = np.ones(len(xc), dtype=bool)
        for i in range(1, len(xc)):
            if xc[i] == xc[i - 1]:
                keep[i] = False
        xc_filtered = xc[keep]
        cp_filtered = cp_suction[keep]
    else:
        xc_filtered = xc
        cp_filtered = cp_suction

    # Calculate spatial derivative (gradient) of pressure coefficient
    dcp_dxc = np.gradient(cp_filtered, xc_filtered)

    # Find the separation point
    n = len(xc_filtered)
    for i in range(n - 2):
        if xc_filtered[i] > 0.4:
            # Check if absolute value of the gradient remains below the threshold for at least 3 consecutive points
            if (abs(dcp_dxc[i]) < threshold and 
                abs(dcp_dxc[i+1]) < threshold and 
                abs(dcp_dxc[i+2]) < threshold):
                return float(xc_filtered[i])

    return None


def plot_center_of_pressure(df: pd.DataFrame) -> str:
    """Calculates the non-dimensional Center of Pressure (x_cp/c) and generates a stability plot using Plotly.

    Formula:
        $x_{cp}/c = 1/4 - (C_{M, c/4} / C_L)$
    
    This function adds a column 'Center_of_Pressure' to the DataFrame, plots Cl vs Cm,
    explicitly configures the x-axis limits/direction to ensure positive values are on
    the right-hand side and negative values are on the left-hand side, and saves the
    plot as 'plot.json'.

    Args:
        df: A pandas DataFrame containing lift ('Cl' or 'C_L') and pitching moment ('Cm' or 'C_M') columns.

    Returns:
        A success message string.
    """
    # Identify lift and moment coefficient columns
    cm_col = 'Cm' if 'Cm' in df.columns else ('C_M' if 'C_M' in df.columns else None)
    cl_col = 'Cl' if 'Cl' in df.columns else ('C_L' if 'C_L' in df.columns else None)

    if cm_col is None or cl_col is None:
        raise KeyError("DataFrame must contain lift ('Cl' or 'C_L') and moment ('Cm' or 'C_M') columns.")

    # Calculate Center of Pressure
    df['Center_of_Pressure'] = 0.25 - (df[cm_col] / df[cl_col])

    # Extract run number
    run_number = "Unknown"
    if "RunNo" in df.columns:
        non_null_runs = df["RunNo"].dropna()
        if not non_null_runs.empty:
            try:
                run_number = int(float(non_null_runs.iloc[0]))
            except ValueError:
                run_number = str(non_null_runs.iloc[0])

    # Calculate Mean Mach and Reynolds
    mean_mach = 0.0
    mean_re = 0.0

    if "Temperature" in df.columns and "Airspeed" in df.columns:
        temp_col = pd.to_numeric(df["Temperature"], errors="coerce")
        airspeed_col = pd.to_numeric(df["Airspeed"], errors="coerce")
        valid_mask = (temp_col > 0) & temp_col.notna() & airspeed_col.notna()
        if valid_mask.any():
            a = np.sqrt(1.4 * 287.05 * temp_col[valid_mask])
            mach_series = airspeed_col[valid_mask] / a
            mean_mach = float(mach_series.mean())

    if "Rey" in df.columns:
        rey_col = pd.to_numeric(df["Rey"], errors="coerce")
        valid_re = rey_col.dropna()
        if not valid_re.empty:
            mean_re = float((valid_re * 1e6).mean())

    # Generate Stability Plot: Lift Coefficient (C_L) on y-axis, Moment Coefficient (C_M) on x-axis
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[cm_col],
            y=df[cl_col],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6, color="#1f77b4"),
            name=f"Mean Mach: {mean_mach:.2f}, Mean Re: {mean_re:.2e}",
            customdata=df['Center_of_Pressure'],
            hovertemplate=(
                f"<b>Run {run_number}</b><br>"
                "Moment Coefficient (C<sub>M</sub>): %{x:.4f}<br>"
                "Lift Coefficient (C<sub>L</sub>): %{y:.4f}<br>"
                "Center of Pressure (x<sub>cp</sub>/c): %{customdata:.4f}<br>"
                f"Mean Mach: {mean_mach:.2f}<br>"
                f"Mean Re: {mean_re:.2e}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Stability Plot: Lift Coefficient vs. Moment Coefficient (Run {run_number})</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Moment Coefficient (C<sub>M</sub>)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            zeroline=True,
            zerolinecolor="rgba(150, 150, 150, 0.5)",
            autorange=True,  # Standard direction: positive on the right, negative on the left
        ),
        yaxis=dict(
            title="Lift Coefficient (C<sub>L</sub>)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            zeroline=True,
            zerolinecolor="rgba(150, 150, 150, 0.5)",
        ),
        template="plotly_white",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
        hovermode="closest",
        margin=dict(l=60, r=40, t=60, b=60),
    )

    fig.write_json("plot.json")
    return "Plot successfully generated using Plotly and saved as plot.json."


def plot_aerodynamic_efficiency(df: pd.DataFrame) -> str:
    """Calculates the aerodynamic efficiency (Cl / Cd) and generates an efficiency plot using Plotly.

    This function adds a column 'Aerodynamic_Efficiency' to the DataFrame, plots
    Aerodynamic Efficiency vs Lift Coefficient, and saves the plot as 'plot.json'.

    Args:
        df: A pandas DataFrame containing lift and drag coefficient columns.

    Returns:
        A success message string.
    """
    # Identify lift and drag coefficient columns dynamically
    cl_col = next((col for col in ['Cl', 'C_L'] if col in df.columns), None)
    cd_col = next((col for col in ['Cd', 'C_D', 'Cdp'] if col in df.columns), None)

    if cl_col is None or cd_col is None:
        raise KeyError("DataFrame must contain lift ('Cl' or 'C_L') and drag ('Cd', 'C_D', or 'Cdp') columns.")

    # Calculate Aerodynamic Efficiency (Cl / Cd). Handle division by zero safely (Cd == 0 -> NaN)
    df['Aerodynamic_Efficiency'] = df[cl_col] / df[cd_col]
    df.loc[df[cd_col] == 0, 'Aerodynamic_Efficiency'] = np.nan

    # Extract run number
    run_number = "Unknown"
    if "RunNo" in df.columns:
        non_null_runs = df["RunNo"].dropna()
        if not non_null_runs.empty:
            try:
                run_number = int(float(non_null_runs.iloc[0]))
            except ValueError:
                run_number = str(non_null_runs.iloc[0])

    # Calculate Mean Mach and Reynolds
    mean_mach = 0.0
    mean_re = 0.0

    if "Temperature" in df.columns and "Airspeed" in df.columns:
        temp_col = pd.to_numeric(df["Temperature"], errors="coerce")
        airspeed_col = pd.to_numeric(df["Airspeed"], errors="coerce")
        valid_mask = (temp_col > 0) & temp_col.notna() & airspeed_col.notna()
        if valid_mask.any():
            a = np.sqrt(1.4 * 287.05 * temp_col[valid_mask])
            mach_series = airspeed_col[valid_mask] / a
            mean_mach = float(mach_series.mean())

    if "Rey" in df.columns:
        rey_col = pd.to_numeric(df["Rey"], errors="coerce")
        valid_re = rey_col.dropna()
        if not valid_re.empty:
            mean_re = float((valid_re * 1e6).mean())

    # Generate Efficiency Plot: Aerodynamic Efficiency (Cl/Cd) on y-axis, Lift Coefficient (Cl) on x-axis
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[cl_col],
            y=df['Aerodynamic_Efficiency'],
            mode="lines+markers",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=6, color="#2ca02c"),
            name=f"Mean Mach: {mean_mach:.2f}, Mean Re: {mean_re:.2e}",
            hovertemplate=(
                f"<b>Run {run_number}</b><br>"
                "Lift Coefficient (C<sub>L</sub>): %{x:.4f}<br>"
                "Aerodynamic Efficiency (C<sub>L</sub>/C<sub>D</sub>): %{y:.4f}<br>"
                f"Mean Mach: {mean_mach:.2f}<br>"
                f"Mean Re: {mean_re:.2e}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Aerodynamic Efficiency vs. Lift Coefficient (Run {run_number})</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Lift Coefficient (C<sub>L</sub>)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            zeroline=True,
            zerolinecolor="rgba(150, 150, 150, 0.5)",
        ),
        yaxis=dict(
            title="Aerodynamic Efficiency (C<sub>L</sub> / C<sub>D</sub>)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            zeroline=True,
            zerolinecolor="rgba(150, 150, 150, 0.5)",
        ),
        template="plotly_white",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
        hovermode="closest",
        margin=dict(l=60, r=40, t=60, b=60),
    )

    fig.write_json("plot.json")
    return "Plot successfully generated using Plotly and saved as plot.json."


def plot_pressure_time_window(run_ids: list[int], time_start: float, time_end: float) -> str:
    """Visualizes the dynamic evolution of the Cp distribution over a specific time window, supporting multiple runs via subplots.

    Args:
        run_ids: A list of RunNo integers to filter the data.
        time_start: Start time in seconds.
        time_end: End time in seconds.

    Returns:
        A success message string.
    """
    if not os.path.exists(csv_path):
        return f"Error: CSV data file not found at {csv_path}."
    if not os.path.exists(xc_path):
        return f"Error: Geometry file not found at {xc_path}."

    if isinstance(run_ids, (int, str)):
        run_ids = [int(run_ids)]

    try:
        df = pd.read_csv(csv_path)
        xc_df = pd.read_csv(xc_path)
    except Exception as e:
        return f"Error reading data files: {e}"

    x = xc_df["x/c"].values
    if len(x) != 63:
        return f"Error: Geometry coordinates length is {len(x)}, expected 63."

    # Ensure numeric columns
    df["RunNo"] = pd.to_numeric(df["RunNo"], errors="coerce")
    df["Time (sec)"] = pd.to_numeric(df["Time (sec)"], errors="coerce")
    if "AOA (deg)" in df.columns:
        df["AOA (deg)"] = pd.to_numeric(df["AOA (deg)"], errors="coerce")

    t_min_bound = min(time_start, time_end)
    t_max_bound = max(time_start, time_end)

    n_runs = len(run_ids)
    if n_runs == 0:
        return "Error: No run IDs provided."

    subplot_titles = [f"Run: {r}" for r in run_ids]
    fig = make_subplots(
        rows=n_runs,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=max(0.05, 0.2 / n_runs) if n_runs > 1 else 0.2,
    )

    cp_cols = [f"Cp_{i}" for i in range(1, 64)]
    plotted_traces = 0

    for i, run_id in enumerate(run_ids, start=1):
        df_run = df[
            (df["RunNo"] == run_id)
            & (df["Time (sec)"] >= t_min_bound)
            & (df["Time (sec)"] <= t_max_bound)
        ].sort_values("Time (sec)")

        if df_run.empty:
            continue

        # If filtered time window has more than 10 rows, evenly sample exactly 10 rows
        if len(df_run) > 10:
            sample_indices = np.round(np.linspace(0, len(df_run) - 1, 10)).astype(int)
            sampled_df = df_run.iloc[sample_indices]
        else:
            sampled_df = df_run

        run_t_min = df_run["Time (sec)"].min()
        run_t_max = df_run["Time (sec)"].max()
        time_span = run_t_max - run_t_min

        for _, row in sampled_df.iterrows():
            t_val = float(row["Time (sec)"])
            aoa_val = float(row["AOA (deg)"]) if "AOA (deg)" in row and pd.notna(row["AOA (deg)"]) else 0.0
            cp_values = row[cp_cols].values.astype(float)

            # Continuous color scale mapped to Time (sec)
            norm_t = (t_val - run_t_min) / time_span if time_span > 0 else 0.5
            norm_t = max(0.0, min(1.0, norm_t))
            color = pcolors.sample_colorscale("Plasma", [norm_t])[0]

            hovertemplate = (
                f"<b>Run {run_id}</b><br>"
                f"Time: {t_val:.3f} s<br>"
                f"AOA: {aoa_val:.2f}°<br>"
                "x/c: %{x:.4f}<br>"
                "Cp: %{y:.4f}<extra></extra>"
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=cp_values,
                    mode="lines+markers",
                    name=f"Run {run_id} (t={t_val:.3f}s)",
                    line=dict(color=color, width=2),
                    marker=dict(size=4, color=color),
                    hovertemplate=hovertemplate,
                ),
                row=i,
                col=1,
            )
            plotted_traces += 1

    if plotted_traces == 0:
        return f"Error: No data found for the specified runs ({run_ids}) within time window [{time_start}, {time_end}]."

    fig.update_yaxes(
        title_text="Cp",
        autorange="reversed",
        showgrid=True,
        gridcolor="rgba(200, 200, 200, 0.3)",
        zeroline=True,
        zerolinecolor="rgba(150, 150, 150, 0.5)",
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(200, 200, 200, 0.3)",
        zeroline=True,
        zerolinecolor="rgba(150, 150, 150, 0.5)",
    )
    fig.update_xaxes(title_text="x/c", row=n_runs, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>Pressure Coefficient Time Evolution ({t_min_bound:.2f}s - {t_max_bound:.2f}s)</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        template="plotly_white",
        height=max(450, 280 * n_runs),
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
        margin=dict(l=60, r=40, t=60, b=60),
    )

    fig.write_json("plot.json")
    return "Plot successfully generated using Plotly and saved as plot.json."
