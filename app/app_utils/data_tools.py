import os
import pandas as pd

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


def extract_run_data(run_id: int) -> str:
    """Filters the merged aerodynamic CSV data for the given RunNo and returns a text summary of its statistics.

    Args:
        run_id: The RunNo integer to filter the data.

    Returns:
        A string containing the summary statistics of the run, or an error message.
    """
    if not os.path.exists(csv_path):
        return f"Error: CSV data file not found at {csv_path}."

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV file: {e}"

    # Ensure RunNo is treated numerically
    df["RunNo"] = pd.to_numeric(df["RunNo"], errors="coerce")
    df_filtered = df[df["RunNo"] == run_id]

    if df_filtered.empty:
        return f"Error: No data found for RunNo {run_id}."

    # Extract statistics
    num_samples = len(df_filtered)
    aoa_min = df_filtered["AOA (deg)"].min()
    aoa_max = df_filtered["AOA (deg)"].max()
    avg_airspeed = df_filtered["Airspeed"].mean()
    avg_temp = df_filtered["Temperature"].mean()
    avg_press = df_filtered["Ambient Pressure"].mean()
    avg_cl = df_filtered["Cl"].mean()
    avg_cdp = df_filtered["Cdp"].mean()
    avg_cm = df_filtered["Cm"].mean()

    summary = (
        f"Summary for Run {run_id}:\n"
        f"- Number of Samples: {num_samples}\n"
        f"- Angle of Attack (AOA) Range: {aoa_min:.2f} to {aoa_max:.2f} deg\n"
        f"- Average Airspeed: {avg_airspeed:.2f} m/s\n"
        f"- Average Temperature: {avg_temp:.2f} K\n"
        f"- Average Ambient Pressure: {avg_press:.2f} Pa\n"
        f"- Average Lift Coefficient (Cl): {avg_cl:.4f}\n"
        f"- Average Pressure Drag Coefficient (Cdp): {avg_cdp:.4f}\n"
        f"- Average Pitching Moment Coefficient (Cm): {avg_cm:.4f}"
    )

    return summary


def get_run_metadata(run_id: int) -> str:
    """Filters the merged aerodynamic CSV data for the given RunNo and returns a text summary of the environmental conditions from the first row of that run.

    Args:
        run_id: The RunNo integer to filter the data.

    Returns:
        A string containing a summary of the environmental conditions, or an error message.
    """
    if not os.path.exists(csv_path):
        return f"Error: CSV data file not found at {csv_path}."

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV file: {e}"

    # Ensure RunNo is treated numerically
    df["RunNo"] = pd.to_numeric(df["RunNo"], errors="coerce")
    df_filtered = df[df["RunNo"] == run_id]

    if df_filtered.empty:
        return f"Error: No data found for RunNo {run_id}."

    # Extract environmental conditions from the first row
    first_row = df_filtered.iloc[0]

    profile = first_row.get("Profile_x", "Unknown")
    run_type = first_row.get("Type", "Unknown")
    degree = first_row.get("Degree", "Unknown")
    temp = first_row.get("Temperature")
    press = first_row.get("Ambient Pressure")
    airspeed = first_row.get("Airspeed")
    rey = first_row.get("Rey")
    freq = first_row.get("Frequency")

    # Format values if numerical, otherwise leave as string
    try:
        temp_str = f"{float(temp):.2f} K" if pd.notna(temp) else "Unknown"
    except (ValueError, TypeError):
        temp_str = str(temp) if pd.notna(temp) else "Unknown"

    try:
        press_str = f"{float(press):.4f} Pa" if pd.notna(press) else "Unknown"
    except (ValueError, TypeError):
        press_str = str(press) if pd.notna(press) else "Unknown"

    try:
        airspeed_str = f"{float(airspeed):.2f} m/s" if pd.notna(airspeed) else "Unknown"
    except (ValueError, TypeError):
        airspeed_str = str(airspeed) if pd.notna(airspeed) else "Unknown"

    try:
        rey_str = f"{float(rey):.2f}" if pd.notna(rey) else "Unknown"
    except (ValueError, TypeError):
        rey_str = str(rey) if pd.notna(rey) else "Unknown"

    try:
        freq_str = f"{float(freq):.2f} Hz" if pd.notna(freq) else "Unknown"
    except (ValueError, TypeError):
        freq_str = str(freq) if pd.notna(freq) else "Unknown"

    degree_str = str(degree).strip() if pd.notna(degree) else "Unknown"
    profile_str = str(profile).strip() if pd.notna(profile) else "Unknown"
    type_str = str(run_type).strip() if pd.notna(run_type) else "Unknown"

    summary = (
        f"Metadata for Run {run_id}:\n"
        f"- Profile: {profile_str}\n"
        f"- Type: {type_str}\n"
        f"- Degree: {degree_str}\n"
        f"- Temperature: {temp_str}\n"
        f"- Ambient Pressure: {press_str}\n"
        f"- Airspeed: {airspeed_str}\n"
        f"- Reynolds Number (Rey): {rey_str}\n"
        f"- Frequency: {freq_str}"
    )

    return summary

