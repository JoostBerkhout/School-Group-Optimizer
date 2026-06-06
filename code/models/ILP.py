from pyscipopt import Model
from pyscipopt import quicksum
from pyscipopt import Eventhdlr, SCIP_EVENTTYPE
import pandas as pd
import time
import csv
import os
import math
from datetime import datetime
from helpers import create_preference_matrix, read_dfs, read_variables, estimated_max_balance_penalty, estimated_max_fairness

def create_initial_model(students, teachers, data, variables):
    model = Model("ilp")

    # Decision variables
    # x[s][t] = 1 if student s assigned to teacher t
    x = {}
    for s in students:
        for t in teachers:
            x[s, t] = model.addVar(vtype="BINARY", name=f"x_{s}_{t}")

    model = add_objective(model, students, teachers, x, data, variables)

    return model, x

def add_objective(model, students, teachers, x, data, variables):
    attributes_to_balance = ['Gender', 'Grade', 'Extra Care']
    if 'Behavior' in data.info_students.columns:
        attributes_to_balance.append('Behavior')

    balance_penalty_terms = add_balance(model, x, attributes_to_balance, teachers, data)

    preferences = create_preference_matrix(data, variables)
    fairness_layers = add_fairness_layers(model, x, students, teachers, preferences)
    fairness_terms = []
    # Get highest number of preferences given by any student
    max_k = max(k for k, _ in fairness_layers) if fairness_layers else 1
    for k, met_k in fairness_layers:
        # Each layer is weighted exponentially based on how many preferences are met
        # Higher k means more preferences met, so weight is lower to focus more on
        # improving fairness for students with fewer preferences met first
        weight = 10 ** (max_k - k)
        fairness_terms.append(weight * met_k)

    print("Fairness terms (ILP):", fairness_terms[:5])

    # Scale each objective by its estimated max value to normalize
    balance_scale = 1 / max(1, estimated_max_balance_penalty(data, attributes_to_balance, teachers))
    fairness_scale = 1 / max(1, estimated_max_fairness(fairness_layers))

    # Apply scaling to weights
    balance_weight = 2 * balance_scale
    fairness_weight = 1 * fairness_scale

    # Set objective
    model.setObjective(
        fairness_weight * quicksum(fairness_terms)
        - balance_weight * quicksum(balance_penalty_terms),
        "maximize"
    )

    return model

# SOFT CONSTRAINTS
def add_balance(model, x, attributes, teachers, data):
    balance_penalty_terms = []
    max_students = len(data.info_students)

    for attribute in attributes:
        # Get unique values for the attribute
        categories = data.info_students[attribute].unique()
        # Get target per teacher for each category
        category_students = {cat: data.info_students[data.info_students[attribute] == cat]['Student'].tolist() for cat in categories}
        target_per_teacher = {cat: len(category_students[cat]) / len(teachers) for cat in categories}

        for t in teachers:
            for cat in categories:
                # Get the list of students in this category
                assigned_count = sum(x[s, t] for s in category_students[cat])
                target = target_per_teacher[cat]

                # Calculate over and under deviation
                over_dev = model.addVar(vtype="INTEGER", lb=0, ub=max_students, name=f"over_dev_{cat}_{t}")
                under_dev = model.addVar(vtype="INTEGER", lb=0, ub=max_students, name=f"under_dev_{cat}_{t}")

                # Measures deviation from target by splitting into over- and under-assignment penalties
                model.addCons(assigned_count - int(target) == over_dev - under_dev)
                balance_penalty_terms.append(over_dev)
                balance_penalty_terms.append(under_dev)

    return balance_penalty_terms

def add_fairness_layers(model, x, students, teachers, preferences):
    all_layer_vars = []

    for s1 in students:
        preferred_students = [s2 for s2 in students if s1 != s2 and preferences.loc[s1, s2] == 1]
        num_prefs = len(preferred_students)

        # Skip students with no preferences
        if num_prefs == 0:
            continue

        satisfied_bools = []
        for s2 in preferred_students:
            # Create var that is 1 if s1 and s2 assigned to same teacher
            together = model.addVar(vtype="BINARY", name=f"together_{s1}_{s2}")
            model.addCons(together <= quicksum(x[s1, t] * x[s2, t] for t in teachers))
            model.addCons(together >= quicksum(x[s1, t] + x[s2, t] - 1 for t in teachers))
            satisfied_bools.append(together)

        # Count the number of satisfied preferences for this student
        num_satisfied = model.addVar(vtype="INTEGER", lb=0, ub=num_prefs, name=f"satisfied_count_{s1}")
        model.addCons(num_satisfied == quicksum(satisfied_bools))

        for k in range(1, num_prefs + 1):
            # Create var for each possible k to track if at least k preferences are met
            met_k = model.addVar(vtype="BINARY", name=f"{s1}_at_least_{k}_prefs")
            # If met_k is 1 then at least k preferences should be met
            model.addCons(num_satisfied >= k - (1 - met_k) * num_prefs)
            # If met_k is 0 then less than k preferences should be met
            model.addCons(num_satisfied <= num_prefs - (1 - met_k))
            all_layer_vars.append((k, met_k))

    return all_layer_vars

# HARD CONSTRAINTS
def add_balance_constraints(model, attribute, deviation, x, teachers, data):
    # Get unique categories for the attribute
    categories = data.info_students[attribute].unique()
    # Get target per teacher for each category
    category_students = {cat: data.info_students[data.info_students[attribute] == cat]['Student'].tolist() for cat in categories}
    target_per_teacher = {cat: len(category_students[cat]) / len(teachers) for cat in categories}

    for t in teachers:
        for cat in categories:
            # Calculate the lower and upper bounds for the number of students in this category
            lower_bound = math.floor((1 - deviation) * target_per_teacher[cat])
            upper_bound = math.ceil((1 + deviation) * target_per_teacher[cat])

            # Get the list of students in this category
            students_in_cat = category_students[cat]

            # Add the balancing constraints for this category and teacher
            model.addCons(quicksum(x[s, t] for s in students_in_cat) >= lower_bound,
                name=f"{attribute}_{cat}_{t}_min")
            model.addCons(quicksum(x[s, t] for s in students_in_cat) <= upper_bound,
                name=f"{attribute}_{cat}_{t}_max")

    return model

def add_fairness_constraints(model, x, students, teachers, preferences, min_prefs_per_kid):
    for s1 in students:
        # Only add constraints if the minimum preference is set greater than 0
        if min_prefs_per_kid > 0:
            preferred_students = [s2 for s2 in students if s1 != s2 and preferences.loc[s1, s2] == 1]

            # Continue if s1 has any preferred students
            if preferred_students:
                together_vars = []

                for s2 in preferred_students:
                    # Add var together = 1 if both students are assigned to the same teacher
                    together = model.addVar(vtype="BINARY", name=f"together_{s1}_{s2}")
                    model.addCons(together <= quicksum(x[s1, t] * x[s2, t] for t in teachers))
                    model.addCons(together >= quicksum(x[s1, t] + x[s2, t] - 1 for t in teachers))
                    together_vars.append(together)

                # Require that the sum of 'together' variables is at least min_prefs_per_kid for student s1
                model.addCons(quicksum(together_vars) >= min_prefs_per_kid, name=f"{s1}_at_least_one_pref")

    return model

def add_hard_constraints(model, x, students, teachers, data, variables, preferences, min_prefs_per_kid, deviation):
    for s1 in students:
        # Each student is assigned to exactly one teacher
        model.addCons(quicksum(x[s1, t] for t in teachers) == 1, name=f"Student_{s1}_assigned_once")

    # Assignment constraints
    for _, (s1, s2, together) in data.constraints_students.iterrows():
        for t in teachers:
            if together == "Yes":
                # Students must be together
                model.addCons(x[s1, t] == x[s2, t])
            elif together == "No":
                # Students must not be together
                model.addCons(x[s1, t] + x[s2, t] <= 1)

    for _, (s, t, together) in data.constraints_teachers.iterrows():
        if together == "Yes":
            # Student must be with the teacher
            model.addCons(x[s, t] == 1)
        elif together == "No":
            # Student must not be with the teacher
            model.addCons(x[s, t] == 0)

    for t in teachers:
        # Group size constraints
        model.addCons(quicksum(x[s, t] for s in students) >= variables.min_group_size, name=f"Teacher_{t}_min_size")
        model.addCons(quicksum(x[s, t] for s in students) <= variables.max_group_size, name=f"Teacher_{t}_max_size")

    # Max extra care constraints
    extra_care_values = dict(zip(data.info_students['Student'], data.info_students['Extra Care'].map({'Yes': 1, 'No': 0})))
    for t in teachers:
        model.addCons(quicksum(x[s, t] * extra_care_values[s] for s in students) <= variables.max_extra_care,
            name=f"max_extra_care_{t}")

    # Add fairness constraints
    model = add_fairness_constraints(model, x, students, teachers, preferences, min_prefs_per_kid)

    # Balancing constraints
    model = add_balance_constraints(model, 'Gender', deviation, x, teachers, data)
    model = add_balance_constraints(model, 'Grade', deviation, x, teachers, data)
    model = add_balance_constraints(model, 'Extra Care', deviation, x, teachers, data)

    # Add balance constraints for behavior if specified
    if 'Behavior' in data.info_students.columns:
        model = add_balance_constraints(model, 'Behavior', deviation, x, teachers, data)
    else:
        print("No 'Behavior' attribute found in the data. Skipping balancing constraints for behavior.")

    return model

# FINAL MODEL CREATION
def create_model(school, processed_data_folder, min_prefs_per_kid, deviation):
    data = read_dfs(school, processed_data_folder)
    variables = read_variables(data)

    students = data.info_students['Student'].tolist()
    teachers = data.info_teachers['Teacher'].tolist()

    # Initialize model
    model, x = create_initial_model(students, teachers, data, variables)

    # Hard constraints
    preference_matrix = create_preference_matrix(data, variables)
    model = add_hard_constraints(model, x, students, teachers, data, variables, preference_matrix, min_prefs_per_kid, deviation)

    return model, x

# RUNNING THE MODEL
class ILPObjectiveLogger:
    def __init__(self, results_folder, timestamp, timelimit, min_prefs_per_kid, deviation):
        # (model, results_folder, timestamp, timelimit, min_prefs_per_kid, deviation
        self.start_time = time.time()
        self.best_objective = None
        self.solution_count = 0
        self.results_folder = results_folder
        self.timestamp = timestamp
        self.school =  os.path.basename(os.path.dirname(results_folder))

        # Setup paths
        log_folder = os.path.join(results_folder, "logs")
        os.makedirs(log_folder, exist_ok=True)
        self.log_file_path = os.path.join(log_folder, f"ILP_{self.timestamp}.csv")

        # Open the CSV file and write headers if it doesn't exist
        with open(self.log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Add metadata to the CSV file
            writer.writerow(["Run Config"])
            writer.writerow(["School", self.school])
            writer.writerow(["Method", "CP"])
            writer.writerow(["Min Prefs Per Kid", min_prefs_per_kid])
            writer.writerow(["Deviation", deviation])
            writer.writerow(["Time Limit (s)", timelimit])
            writer.writerow([])
            writer.writerow(["Timestamp", "Solution #", "Elapsed Time (s)", "Objective Value"])

    def log_solution(self, model):
        if model.getNSols() == 0:
            return  # No solution to log

        try:
            current_objective = model.getSolObjVal(model.getBestSol())
        except Exception as e:
            print(f"Warning: Unable to retrieve objective value: {e}")
            return

        elapsed = time.time() - self.start_time
        self.solution_count += 1

        if self.best_objective is None or current_objective >= self.best_objective:
            self.best_objective = current_objective
            print(f"[{elapsed:.1f}s] New best solution #{self.solution_count}, objective = {current_objective}")
            self.save_to_csv(elapsed, current_objective)

    def save_to_csv(self, elapsed, current_objective):
        timestamp = datetime.now().strftime("%d-%m_%H:%M:%S")
        with open(self.log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, self.solution_count, round(elapsed, 3), current_objective])

    def end_search(self, status_str):
        elapsed = time.time() - self.start_time
        if self.best_objective is not None:
            print(f"[{elapsed:.1f}s] Search ended. Best solution #{self.solution_count}, objective = {self.best_objective}")
            self.save_to_csv(elapsed, self.best_objective)

            # Write final status to logging
            with open(self.log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Status", status_str])

class BestSolutionLogger(Eventhdlr):
    def __init__(self, logger):
        self.logger = logger

    def eventinit(self):
        # Catch events when a new best solution is found
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        self.logger.log_solution(self.model)
        return {"result": None}

def solve_model(model, results_folder, timestamp, timelimit, min_prefs_per_kid, deviation):
    model.setParam("limits/time", timelimit)

    # Set seed and settings to ensure reproducibility and enable single-threaded search
    model.setParam("randomization/randomseedshift", 42)
    model.setParam("randomization/permutationseed", 42)
    model.setParam("randomization/permutevars", False)
    model.setParam("parallel/maxnthreads", 1)

    # Set up and attach the logger callback
    logger = ILPObjectiveLogger(results_folder, timestamp, timelimit, min_prefs_per_kid, deviation)
    model.includeEventhdlr(BestSolutionLogger(logger), "BestSolutionLogger", "Logs when a better solution is found")

    # Solve the model
    model.optimize()

    # Log best solution
    logger.log_solution(model)

    # Final log at the end
    status_str = model.getStatus()
    logger.end_search(status_str)
    return model

def format_solution(model, x):
    assignments = []
    for (s, t), var in x.items():
        val = model.getVal(var)
        if val > 0.5:
            assignments.append((s, t))

    df = pd.DataFrame(assignments, columns=["Student", "Teacher"])
    df = df.sort_values(by="Teacher")

    return df

def run_ilp(school, processed_data_folder, timelimit, min_prefs_start, deviation):
    folder = 'data/results'
    timestamp = datetime.now().strftime("%d-%m_%H_%M")
    results_folder = os.path.join(folder, school, "ILP")

    # 1. Try decreasing min_prefs from 5 to 0 with normal deviation
    for min_prefs in reversed(range(min_prefs_start +1)):
        print(f"Phase 1: Trying min_prefs_per_kid={min_prefs}, deviation={deviation}")
        model, x = create_model(school, processed_data_folder, min_prefs, deviation)
        model = solve_model(model, results_folder, timestamp, timelimit, min_prefs, deviation)

        if model.getNSols() > 0:
            df = format_solution(model, x)
            return df, timestamp

    # 2. Try again with no balance constraint (deviation=1.0)
    for min_prefs in reversed(range(min_prefs_start + 1)):
        print(f"Phase 2: Trying min_prefs_per_kid={min_prefs}, deviation=1.0")
        model, x = create_model(school, processed_data_folder, min_prefs, 1.0)
        model = solve_model(model, results_folder, timestamp, timelimit, min_prefs, 1.0)
        if model.getNSols() > 0:
            df = format_solution(model, x)
            return df, timestamp

    print("No solution found in any configuration.")
    return None, timestamp
