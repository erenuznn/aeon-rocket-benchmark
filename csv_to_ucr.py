import pandas as pd
import numpy as np
import os


def generate_unified_unilateral_ts(file_label_mapping, output_path, dataset_name, points_per_window=3600):
    """
    Ingests multiple processed CSV files and outputs a single aggregated unilateral .ts file.

    Parameters:
    - file_label_mapping: Dictionary. Keys = CSV file paths (strings). Values = state labels (integers).
    - output_path: String path for the final .ts file.
    - dataset_name: String identifier for the @problemName header.
    - points_per_window: Integer defining temporal sequence length.
    """

    all_instances = []
    total_processed_files = 0

    for csv_path, state_label in file_label_mapping.items():
        if not os.path.exists(csv_path):
            print(f"Warning: File not found {csv_path}. Skipping.")
            continue

        # Data Ingestion
        df = pd.read_csv(csv_path)
        signal_array = df.iloc[:, 1].values

        # Temporal Segmentation
        total_valid_chunks = len(signal_array) // points_per_window

        if total_valid_chunks > 0:
            truncated_signal = signal_array[:total_valid_chunks * points_per_window]
            feature_matrix = truncated_signal.reshape(total_valid_chunks, points_per_window)

            # Format rows and append to master list
            for row in feature_matrix:
                row_str = ",".join([f"{val:.6f}" for val in row])
                all_instances.append(f"{row_str}:{state_label}\n")

            total_processed_files += 1

    if not all_instances:
        raise ValueError("Execution halted: No valid instances generated from provided files.")

    # File Generation Sequence
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as file:
        # Write Metadata Header Block
        file.write(f"@problemName {dataset_name}\n")
        file.write("@timeStamps false\n")
        file.write("@missing false\n")
        file.write("@univariate true\n")
        file.write("@equalLength true\n")

        # Extract unique labels dynamically for header
        unique_labels = sorted(list(set(file_label_mapping.values())))
        label_str = " ".join(map(str, unique_labels))
        file.write(f"@classLabel true {label_str}\n")

        file.write("@data\n")

        # Write Unified Data Matrix
        for instance in all_instances:
            file.write(instance)

    print(f"Unified unilateral .ts transformation complete.")
    print(f"Processed {total_processed_files} files.")
    print(f"Total aggregated instances: {len(all_instances)}.")
    return output_path