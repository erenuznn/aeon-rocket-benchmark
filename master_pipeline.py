import parquet_to_csv
import Preprocessing_and_Comparison
import csv_to_ucr

from aeon.datasets import load_from_ts_file
from sklearn.model_selection import train_test_split
from aeon.classification.convolution_based import RocketClassifier
from sklearn.metrics import accuracy_score


def execute_rocket_classification(ts_file_path, test_fraction=0.2):
    """
    Executes ingestion, training, and evaluation of ROCKET on a unilateral .ts file.
    """
    print("Initiating unilateral data ingestion sequence.")
    X, y = load_from_ts_file(ts_file_path)

    print(f"Total unilateral instances extracted: {X.shape[0]}")
    print(f"Temporal dimension per instance: {X.shape[2]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=42,
        stratify=y
    )

    print("Initializing unilateral ROCKET classifier.")
    classifier = RocketClassifier(n_kernels=10000, random_state=42)

    print("Execution of training protocol.")
    classifier.fit(X_train, y_train)

    print("Execution of testing protocol.")
    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Unilateral classification accuracy: {accuracy:.4f}")

    return classifier


# ==========================================
# --- CONFIGURATION PARAMETERS ---
# ==========================================
# Mapping configuration for the unilateral plant channels
# 0 = unstressed, 1 = stressed
PLANT_DICTIONARY = {
    #"25072203-1": 0,  # Waterlogged
    "25072205-1": 1,  # Stress
    "25072219-1": 0,    #Adequate
    "25072221-1": 1,    #Stress
    "25072233-1": 0,    #Adequate
    #"25072235-1": 0,    #Waterlogged
    #"25072236-1": 0,    #Adequate
    #"25072237-1": 0,    #Waterlogged
    #"25072238-1": 0,    #Waterlogged
    #"25072240-1": 0,    #Adequate
    #"25072245-1": 0,    #Waterlogged
    #"25072247-1": 0,    #Waterlogged
    #"25072249-1": 0,    #Adequate
    #"25072252-1": 0,    #Waterlogged
    #"25072269-1": 0,    #Adequate
    #"25072277-1": 0,    #Waterlogged
    #"25072279-1": 1,    #Stress
    #"25072280-1": 1,    #Stress
    #"25072283-1": 1,    #Stress
    #"25072288-1": 0,    #Adequate
    #"25072290-1": 1,    #Stress
    #"25072294-1": 1,    #Stress
    #"25072300-1": 1,    #Stress
    #"25072361-1": 0,    #Adequate
}

RAW_DATA_ROOT = "/Users/erenuzun/Desktop/Thesis/ML/DATA/test_data/vivent46/1sec"
DAYS_PER_CHUNK = 12
SPIKE_THRESHOLD_STD = 4.0
SAVGOL_WINDOW = 201
SAVGOL_POLY = 1
PLOT_DAYS_SPAN = 2.0

TS_OUTPUT_FILE = "/Users/erenuzun/Desktop/Thesis/ML/DATA/processed_ts/Vivent_Master_Unilateral.ts"
DATASET_IDENTIFIER = "Vivent_Master_Unilateral"
POINTS_PER_WINDOW = 3600

file_label_mapping = {}

# ==========================================
# --- BATCH EXECUTION SEQUENCE ---
# ==========================================
for plant_id, label in PLANT_DICTIONARY.items():
    print(f"Processing unilateral sequence for plant: {plant_id}")

    generated_csv_dir = parquet_to_csv.execute_chunking(
        raw_data_path=f"{RAW_DATA_ROOT}",
        days_per_chunk=DAYS_PER_CHUNK
    )

    Preprocessing_and_Comparison.execute_processing(
        base_dir=generated_csv_dir,
        plant_identifier=plant_id,
        threshold_std=SPIKE_THRESHOLD_STD,
        savgol_window=SAVGOL_WINDOW,
        savgol_poly=SAVGOL_POLY,
        plot_days_span=PLOT_DAYS_SPAN
    )

    # Store the target output file and its corresponding label
    processed_csv_target = f"{generated_csv_dir}/plant_id={plant_id}/processed_training_data.csv"
    file_label_mapping[processed_csv_target] = label

print("-" * 50)
print("Initiating Master Unilateral TS Generation")

generated_ts_file = csv_to_ucr.generate_unified_unilateral_ts(
    file_label_mapping=file_label_mapping,
    output_path=TS_OUTPUT_FILE,
    dataset_name=DATASET_IDENTIFIER,
    points_per_window=POINTS_PER_WINDOW
)

print("-" * 50)
print("Initiating ROCKET Classification Sequence")

trained_model = execute_rocket_classification(
    ts_file_path=generated_ts_file,
    test_fraction=0.2
)

print("-" * 50)
print("Pipeline execution sequence terminated.")