import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import numpy as np
from scipy.signal import savgol_filter

# --- CONFIGURATION ---
PLANT_IDENTIFIER = "25072203-1"
BASE_DIR = Path("PATH")

# Define output locations
OUTPUT_CSV = BASE_DIR / PLANT_IDENTIFIER / "processed_training_data.csv"
OUTPUT_HTML = BASE_DIR / PLANT_IDENTIFIER / "interactive_plot.html"

# --- PARAMETERS ---
SPIKE_THRESHOLD_STD = 4.0  # Z-score threshold for spike detection
SAVGOL_WINDOW = 201  # Window size for Savitzky-Golay filter (must be odd) 201
SAVGOL_POLY = 1  # Polynomial order (1 = linear smoothing)
PLOT_DAYS_SPAN = 2.0  # Visualization span


# ============ FUNCTIONS ============

def remove_spikes(signal, threshold_std=SPIKE_THRESHOLD_STD):
    """
    Identifies spikes using Z-score and replaces them via linear interpolation.
    """
    mean_val = signal.mean()
    std_val = signal.std()

    # Avoid division by zero
    if std_val == 0:
        return signal

    z_scores = np.abs((signal - mean_val) / std_val)
    cleaned = signal.copy()
    spike_mask = z_scores > threshold_std

    if spike_mask.any():
        x_indices = np.arange(len(signal))
        cleaned[spike_mask] = np.interp(
            x_indices[spike_mask],
            x_indices[~spike_mask],
            signal[~spike_mask]
        )
        print(f"STATUS: Removed {spike_mask.sum()} spikes using Z-score threshold {threshold_std}.")
    else:
        print("STATUS: No spikes detected above threshold.")

    return cleaned


def filter_signal(signal):
    """
    Applies spike removal followed by Savitzky-Golay smoothing.
    """
    # 1. Despike
    despiked = remove_spikes(signal)

    # 2. Smooth (Savitzky-Golay)
    if len(despiked) > SAVGOL_WINDOW:
        print(f"STATUS: Applying Savitzky-Golay filter (Window={SAVGOL_WINDOW}, Poly={SAVGOL_POLY})...")
        smooth = savgol_filter(despiked, SAVGOL_WINDOW, SAVGOL_POLY)
    else:
        print("WARNING: Signal too short for Savitzky-Golay window. Skipping smoothing.")
        smooth = despiked

    return smooth


def get_file_paths(base_dir, plant_id):
    """Automatically finds all CSV files in the plant's directory."""
    plant_dir = base_dir / plant_id

    if not plant_dir.exists():
        print(f"ERROR: Directory not found: {plant_dir}")
        return []

    files = sorted([f for f in plant_dir.glob("*chunk*.csv")])
    print(f"STATUS: Found {len(files)} chunk files in {plant_dir.name}")
    return files


def process_data(file_paths):
    if not file_paths:
        print("FATAL: No file paths provided.")
        return None, None

    print(f"STATUS: Loading {len(file_paths)} raw files...")

    all_dfs = []
    for f in file_paths:
        try:
            df = pd.read_csv(f, usecols=['timestamp', 'metric_value'])
            all_dfs.append(df)
        except Exception as e:
            print(f"ERROR: Failed to read {f.name}: {e}")

    if not all_dfs:
        print("FATAL: No dataframes could be loaded.")
        return None, None

    # Merge
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Data Cleaning
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
    full_df['metric_value'] = pd.to_numeric(full_df['metric_value'], errors='coerce')

    # Initial NaN removal
    full_df = full_df.dropna(subset=['metric_value', 'timestamp'])
    full_df = full_df.sort_values('timestamp').drop_duplicates('timestamp')

    raw_count = len(full_df)
    print(f"STATUS: Raw data count: {raw_count} rows.")

    # --- PIPELINE APPLICATION ---

    signal_raw = full_df['metric_value'].values
    signal_processed = filter_signal(signal_raw)

    processed_df = full_df.copy()
    processed_df['metric_value'] = signal_processed

    return full_df, processed_df


def plot_data_span(raw_df, processed_df):
    """Generates an interactive Plotly chart and saves it as an HTML file."""
    print(f"STATUS: Generating interactive plot showing the last {PLOT_DAYS_SPAN} days...")

    end_time = processed_df['timestamp'].max()
    start_time = end_time - pd.Timedelta(days=PLOT_DAYS_SPAN)

    plot_processed = processed_df[processed_df['timestamp'] >= start_time]
    plot_raw = raw_df[raw_df['timestamp'] >= start_time]

    fig = go.Figure()

    # Trace 1: Raw Data (Gray)
    fig.add_trace(go.Scattergl(
        x=plot_raw['timestamp'],
        y=plot_raw['metric_value'],
        mode='lines',
        name='Raw Signal',
        line=dict(color='gray', width=1),
        opacity=0.5
    ))

    # Trace 2: Processed Data (Blue)
    fig.add_trace(go.Scattergl(
        x=plot_processed['timestamp'],
        y=plot_processed['metric_value'],
        mode='lines',
        name='Processed (Despiked + SavGol)',
        line=dict(color='blue', width=2),
        opacity=0.9
    ))

    # Layout Configuration
    fig.update_layout(
        title=f"Signal Processing: Z-Score Despiking & Savitzky-Golay (Last {PLOT_DAYS_SPAN} Days)",
        xaxis_title="Timestamp",
        yaxis_title="EP Reading",
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode="x unified"
    )

    # Save to HTML file
    print(f"STATUS: Saving interactive plot to {OUTPUT_HTML}...")
    fig.write_html(str(OUTPUT_HTML))

    # Show in browser/notebook
    fig.show()


# --- EXECUTION ---

input_files = get_file_paths(BASE_DIR, PLANT_IDENTIFIER)

if not input_files:
    print("STOP: No files found.")
    sys.exit(1)

raw_data, processed_data = process_data(input_files)

if processed_data is not None:
    # Ensure directory exists
    (BASE_DIR / PLANT_IDENTIFIER).mkdir(parents=True, exist_ok=True)

    # Save CSV
    processed_data.to_csv(OUTPUT_CSV, index=False)
    print(f"SUCCESS: Processed data saved to: {OUTPUT_CSV}")

    # Generate and Save HTML Plot
    plot_data_span(raw_data, processed_data)
else:
    print("FAILURE: Processing returned None.")