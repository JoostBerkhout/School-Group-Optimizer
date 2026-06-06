import os

from code.models.ILP import run_ilp
from code.models.CP import run_cp
from code.evaluation.evaluate_results import run_evaluate

import sys

def save_results(results, timestamp):
    folder = 'data/results'
    results_folder = os.path.join(folder, school, method)

    solution_folder = os.path.join(results_folder, "solutions")
    print(f"Saving results to {solution_folder}")
    os.makedirs(solution_folder, exist_ok=True)

    output_file = os.path.join(solution_folder, f"{method}_{timestamp}.csv")
    results.to_csv(output_file, index=False)

def run_pipeline():
    print("Running pipeline for school: {}".format(school))

    results = None
    timestamp = None

    # Run ILP algorithm
    if run_baseline_ilp:
        results, timestamp = run_ilp(school, processed_data_folder, timelimit, min_prefs_per_kid, deviation)

    # Run CP algorithm
    if run_cp_model:
        results, timestamp = run_cp(school, processed_data_folder, timelimit, min_prefs_per_kid, deviation)

    if results is not None:
        # Save results
        save_results(results, timestamp)
        # Evaluate results
        run_evaluate(school, processed_data_folder, method, results, timestamp)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 main.py <school> <method: cp|ilp> [timelimit] [min_prefs_per_kid] [deviation]")
        sys.exit(1)

    school = sys.argv[1]
    method = sys.argv[2].upper()
    random_seed = 42

    # Set which model to run
    run_baseline_ilp = method == "ILP"
    run_cp_model = method == "CP"

    # Set time limit for the solver (default 30 minutes)
    timelimit = 30 * 60
    if len(sys.argv) > 3 and sys.argv[3] != "-":
        timelimit = int(sys.argv[3])

    # Set minimum preferences per kid (default 1)
    min_prefs_per_kid = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    # Set deviation to 10% (default 0.1)
    deviation = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1

    # Define paths
    processed_data_folder = 'data/processed_data'

    # Run pipeline
    print(f"school {school}, method {method}")
    run_pipeline()
