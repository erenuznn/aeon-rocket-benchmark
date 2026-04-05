import pandas as pd
import numpy as np
import os


def generate_unilateral_ts(csv_path, output_dir, dataset_name, state_label, points_per_window=3600):
    """
    Ingests continuous CSV biosignals and outputs a unilateral .ts file.
    Designed for master pipeline integration.

    Parameters:
    - csv_path: String path to the raw input CSV.
    - output_dir: String path to the target directory for the .ts file.
    - dataset_name: String identifier for the @problemName header.
    - state_label: Integer classification target (e.g., 0 or 1).
    - points_per_window: Integer defining temporal sequence length.

    Returns:
    - output_path: String path of the generated .ts file for downstream processing.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}.ts")

    # Data Ingestion
    df = pd.read_csv(csv_path)
    signal_array = df.iloc[:, 1].values

    # Temporal Segmentation
    total_valid_chunks = len(signal_array) // points_per_window

    if total_valid_chunks == 0:
        raise ValueError("Execution halted: Insufficient data to form a single temporal window.")

    truncated_signal = signal_array[:total_valid_chunks * points_per_window]
    feature_matrix = truncated_signal.reshape(total_valid_chunks, points_per_window)

    # File Generation Sequence
    with open(output_path, 'w') as file:
        # Write Metadata Header Block
        file.write(f"@problemName {dataset_name}\n")
        file.write("@timeStamps false\n")
        file.write("@missing false\n")
        file.write("@univariate true\n")
        file.write("@equalLength true\n")
        file.write("@classLabel true 0 1\n")
        file.write("@data\n")

        # Write Data Matrix
        for row in feature_matrix:
            row_str = ",".join([f"{val:.6f}" for val in row])
            file.write(f"{row_str}:{state_label}\n")

    print(f"Unilateral .ts transformation complete. Generated {total_valid_chunks} instances.")
    return output_path