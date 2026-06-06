# Getting Started (uv)

Assuming `uv` is already installed:

1. Open a terminal in the project root (`School-Group-Optimizer`).
2. Sync the environment and dependencies:

```bash
uv sync
```

3. Run the pipeline (example):

```bash
uv run python main.py <school> <method: cp|ilp> [timelimit] [min_prefs_per_kid] [deviation]
```

Use the sections below for preprocessing and evaluation details.

## Instructions
### Adding excel data
1. Add a folder with the name of the school to the `data/raw_data` folder
2. Add a single excel file to the folder

### Preprocessing data
To preprocess the data, follow these steps:
1. Open the `data/preprocess.py` file
2. Make sure the paths to the raw data and processed data are correct
   - `raw_data_path = "data/raw_data"`
   - `processed_data_path = "data/processed_data"`
3. Run `python3 code/preprocessing/preprocess.py` to preprocess the data for all schools

### Running optimization models
1. Open the `main.py` file
2. Make sure the paths to the processed data are correct
   - `processed_data_path = "data/processed_data"`
3. Run `python3 main.py <school> <method: cp|ilp> [timelimit] [min_prefs_per_kid] [deviation]`
   - `<school>`: The name of the school folder (e.g. `school1`)
   - `<method>`: The optimization method to use (e.g. `cp`, `ilp`)
   - `[random_seed]`: Optional random seed for reproducibility (default is 42)

### Running evaluation
Evaluation is run directly after running the optimization models. They can be run separately as well.
1. Open the `code/evaluation/evaluate_results.py` file
2. Make sure the paths to the processed data and results are correct
   - `processed_data_path = "data/processed_data"`
   - `results_path = "data/results"`
3. For now, change school name and results file path to what you want to evaluate
4. Run `python3 code/evaluation/evaluate_results.py <method: cp|ilp >` to evaluate the results
   - `<method>`: The optimization method to evaluate (e.g. `cp`, `ilp`)
5. Results will be saved in `data/results/<school>/<method>`
