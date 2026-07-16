from ortools.sat.python import cp_model
import math
import os
import csv
import time
from datetime import datetime
import pandas as pd
from helpers import create_preference_matrix, read_dfs, read_variables, estimated_max_balance_penalty, estimated_max_fairness

def create_initial_model(students, teachers, data, variables):
    model = cp_model.CpModel()

    # Decision variables
    # x[s, t] = 1 if student s assigned to teacher t, 0 else
    x = {}
    for s in students:
        for t in teachers:
            x[s, t] = model.NewBoolVar(f'x_{s}_{t}')

    model = add_objective(model, x, students, teachers, data, variables)

    return model, x

def add_objective(model, x, students, teachers, data, variables):
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
        # TODO: The sum does not match how it is written down in (13) of the thesis
        #   I have the feeling you want to have a counter for number of matches of preferences
        #   and then just give that a weight. So sum_k num_matches[k] * weights[k]

    print("Fairness terms (CP):", fairness_terms[:5])

    # Scale each objective by its estimated max value to normalize
    balance_scale = 1 / max(1, estimated_max_balance_penalty(data, attributes_to_balance, teachers))
    fairness_scale = 1 / max(1, estimated_max_fairness(fairness_layers))

    # Apply scaling to weights
    balance_weight = 2 * balance_scale
    fairness_weight = 1 * fairness_scale

    model.Maximize(
        fairness_weight * sum(fairness_terms)
        - balance_weight * sum(balance_penalty_terms)
    )

    return model

# SOFT CONSTRAINTS
def add_balance(model, x, attributes, teachers, data):
    balance_penalty_terms = []
    max_students = len(data.info_students)

    for attribute in attributes:
        # Get the unique categories for the attribute
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
                over_dev = model.NewIntVar(0, max_students, f"over_dev_{t}_{attribute}_{cat}")
                under_dev = model.NewIntVar(0, max_students, f"under_dev_{t}_{attribute}_{cat}")

                # Measures deviation from target by splitting into over- and under-assignment penalties
                model.Add(assigned_count - int(target) == over_dev - under_dev)
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
            # Create var that is 1 if both students are assigned to the same teacher
            both_assigned = model.NewBoolVar(f"satisfied_{s1}_{s2}")
            both_assigned_per_teacher = [model.NewBoolVar(f"{s1}_{s2}_with_{t}") for t in teachers]
            for i, t in enumerate(teachers):
                # Check if both students are assigned to a teacher
                model.AddBoolAnd([x[s1, t], x[s2, t]]).OnlyEnforceIf(both_assigned_per_teacher[i])
                model.AddBoolOr([x[s1, t].Not(), x[s2, t].Not()]).OnlyEnforceIf(both_assigned_per_teacher[i].Not())

            # Var is only 1 if at least one of the together_per_teacher vars is 1
            model.AddBoolOr(both_assigned_per_teacher).OnlyEnforceIf(both_assigned)
            model.AddBoolAnd([v.Not() for v in both_assigned_per_teacher]).OnlyEnforceIf(both_assigned.Not())
            satisfied_bools.append(both_assigned)

        # Count the number of satisfied preferences for this student
        num_satisfied = model.NewIntVar(0, num_prefs, f"num_satisfied_{s1}")
        model.Add(num_satisfied == sum(satisfied_bools))

        # Preference layers: has at least k prefs satisfied?
        for k in range(1, num_prefs + 1):
            # Create var for each possible k to track if at least k preferences are met
            met_k = model.NewBoolVar(f"{s1}_at_least_{k}_prefs")
            # If met_k is 1 then at least k preferences should be met
            model.Add(num_satisfied >= k).OnlyEnforceIf(met_k)
            # If met_k is 0 then less than k preferences should be met
            model.Add(num_satisfied < k).OnlyEnforceIf(met_k.Not())
            # TODO: Why not just model.Add(num_satisfied == k).OnlyEnforceIf(met_k), or something similar?
            all_layer_vars.append((k, met_k))

    return all_layer_vars

# HARD CONSTRAINTS
def add_balance_constraints(model, attribute, deviation, x, teachers, data):
    # Get the unique categories for the attribute
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
            model.Add(sum(x[s, t] for s in students_in_cat) >= lower_bound)
            model.Add(sum(x[s, t] for s in students_in_cat) <= upper_bound)

    return model

def add_fairness_constraints(model, x, students, teachers, preferences, min_prefs_per_kid):
    if min_prefs_per_kid == 0:
        return model

    for s1 in students:
        preferred_students = [s2 for s2 in students if s1 != s2 and preferences.loc[s1, s2] == 1]

        if not preferred_students:
            continue  # Skip if no preferred students

        together_vars = []

        for s2 in preferred_students:
            # Create var that is 1 if both students are assigned to the same teacher
            both_assigned = model.NewBoolVar(f"satisfied_{s1}_{s2}")
            both_assigned_per_teacher = [model.NewBoolVar(f"{s1}_{s2}_with_{t}") for t in teachers]
            for i, t in enumerate(teachers):
                # Check if both students are assigned to a teacher
                model.AddBoolAnd([x[s1, t], x[s2, t]]).OnlyEnforceIf(both_assigned_per_teacher[i])
                model.AddBoolOr([x[s1, t].Not(), x[s2, t].Not()]).OnlyEnforceIf(both_assigned_per_teacher[i].Not())

            # Var is only 1 if at least one of the together_per_teacher vars is 1
            model.AddBoolOr(both_assigned_per_teacher).OnlyEnforceIf(both_assigned)
            model.AddBoolAnd([v.Not() for v in both_assigned_per_teacher]).OnlyEnforceIf(both_assigned.Not())
            together_vars.append(both_assigned)

        # Require that the sum of 'together' variables is at least min_prefs_per_kid for student s1
        model.Add(sum(together_vars) >= min_prefs_per_kid)

    return model

def add_hard_constraints(model, x, students, teachers, data, variables, preferences, min_prefs_per_kid, deviation):
    for s1 in students:
        # Each student must be assigned to exactly one teacher
        model.AddExactlyOne(x[s1, t] for t in teachers)

    # Assignment constraints
    for _, (s1, s2, together) in data.constraints_students.iterrows():
        for t in teachers:
            if together == "Yes":
                # Students must be together
                model.Add(x[s1, t] == x[s2, t])
            elif together == "No":
                # Students must not be together
                model.Add(x[s1, t] + x[s2, t] <= 1)

    for _, (s, t, together) in data.constraints_teachers.iterrows():
        if together == "Yes":
            # Student must be with the teacher
            model.Add(x[s, t] == 1)
        elif together == "No":
            # Student must not be with the teacher
            model.Add(x[s, t] == 0)

    # Maximum group size constraint
    for t in teachers:
        model.Add(sum(x[s, t] for s in students) >= variables.min_group_size)
        model.Add(sum(x[s, t] for s in students) <= variables.max_group_size)

    # Maximum extra care constraints
    extra_care_values = dict(zip(data.info_students['Student'], data.info_students['Extra Care'].map({'Yes': 1, 'No': 0})))
    for t in teachers:
        model.Add(sum(x[s, t] * extra_care_values[s] for s in students) <= variables.max_extra_care)

    # Add fairness constraints
    model = add_fairness_constraints(model, x, students, teachers, preferences, min_prefs_per_kid)

    # Add balance constraints
    model = add_balance_constraints(model, 'Gender', deviation, x, teachers, data)
    # model = add_balance_constraints(model, 'Grade', deviation, x, teachers, data)
    model = add_balance_constraints(model, 'Extra Care', deviation, x, teachers, data)

    # Add balance constraints for behavior if specified
    if 'Behavior' in data.info_students.columns:
        model = add_balance_constraints(model, 'Behavior', deviation, x, teachers, data)
    if 'Learning' in data.info_students.columns:
        model = add_balance_constraints(model, 'Learning', deviation, x, teachers, data)
    if 'Combination' in data.info_students.columns:
        model = add_balance_constraints(model, 'Learning', deviation, x, teachers, data)
    else:
        print("No extra attributes found in the data.")

    return model

# FINAL MODEL CREATION
def create_model(school, processed_data_folder, min_prefs_per_kid, deviation):
    data = read_dfs(school, processed_data_folder)
    variables = read_variables(data)

    students = data.info_students['Student'].tolist()
    teachers = data.info_teachers['Teacher'].tolist()

    # Initialize model
    model, x = create_initial_model(students, teachers, data, variables)

    # Add hard constraints
    preference_matrix = create_preference_matrix(data, variables)
    model = add_hard_constraints(model, x, students, teachers, data, variables, preference_matrix, min_prefs_per_kid, deviation)

    return model, x

# RUNNING THE MODEL
class ObjectiveLogger(cp_model.CpSolverSolutionCallback):
    def __init__(self, results_folder, timestamp, timelimit, min_prefs_per_kid, deviation):
        super().__init__()
        self.start_time = time.time()
        self.best_objective = None
        self.solution_count = 0
        self.timestamp = timestamp
        self.school =  os.path.basename(os.path.dirname(results_folder))

        # Create a subfolder for logs
        log_folder = os.path.join(results_folder, "logs")
        os.makedirs(log_folder, exist_ok=True)
        self.results_folder = log_folder

        # Set up the CSV file with a timestamp-based filename
        self.file_path = os.path.join(self.results_folder, f"CP_{self.timestamp}.csv")

        # Open the CSV file and write headers if it doesn't exist
        with open(self.file_path, mode='w', newline='') as file:
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

    def on_solution_callback(self):
        self.solution_count += 1
        current_objective = self.ObjectiveValue()
        elapsed = time.time() - self.start_time

        # Log every new best solution
        if (self.best_objective is None) or (current_objective > self.best_objective):
            self.best_objective = current_objective
            print(f"[{elapsed:.1f}s] New best solution #{self.solution_count}, objective = {current_objective}")
            self.save_to_csv(elapsed, current_objective, self.timestamp)

    def save_to_csv(self, elapsed, current_objective, timestamp):
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, self.solution_count, round(elapsed, 3), current_objective])

    def EndSearch(self, status_str):
        if self.best_objective is not None:
            elapsed = time.time() - self.start_time
            print(f"[{elapsed:.1f}s] Search ended. Best solution #{self.solution_count}, objective = {self.best_objective}")
            self.save_to_csv(elapsed, self.best_objective, self.timestamp)

            with open(self.file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Status", status_str])

def solve_model(model, x, results_folder, timestamp, timelimit, min_prefs_per_kid, deviation):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timelimit
    solver.parameters.log_search_progress = True

    # Set seed to ensure reproducibility and enable single-threaded search
    solver.parameters.random_seed = 42
    solver.parameters.num_search_workers = 16

    # Set up and attach the logger callback
    logger = ObjectiveLogger(results_folder, timestamp, timelimit, min_prefs_per_kid, deviation)

    # OR-Tools API changed method names in newer versions.
    if hasattr(solver, "SolveWithSolutionCallback"):
        status = solver.SolveWithSolutionCallback(model, logger)
    else:
        status = solver.solve(model, logger)

    if hasattr(solver, "StatusName"):
        status_str = solver.StatusName(status)
    elif hasattr(solver, "status_name"):
        status_str = solver.status_name(status)
    else:
        status_str = str(status)

    logger.EndSearch(status_str)

    # Check if a solution was found
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        solution = {key: solver.Value(var) for key, var in x.items()}
        return solution

def format_solution(solution):
    assignments = [(student, teacher) for (student, teacher), assigned in solution.items() if assigned == 1]
    df = pd.DataFrame(assignments, columns=['Student', 'Teacher'])
    df = df.sort_values(by='Teacher')
    return df

def run_cp(school, processed_data_folder, timelimit, min_prefs_start, deviation):
    folder = 'data/results'
    timestamp = datetime.now().strftime("%d-%m_%H_%M")
    results_folder = os.path.join(folder, school, "CP")

    # 1. Try decreasing min_prefs from 5 to 0 with normal deviation
    for min_prefs in reversed(range(min_prefs_start + 1)):
        print(f"Phase 1: Trying min_prefs_per_kid={min_prefs}, deviation={deviation}")
        model, x = create_model(school, processed_data_folder, min_prefs, deviation)
        solution = solve_model(model, x, results_folder, timestamp, timelimit, min_prefs, deviation)
        if solution:
            df = format_solution(solution)
            return df, timestamp

    # 2. Try again with no balance constraint (deviation = 1.0)
    for min_prefs in reversed(range(min_prefs_start + 1)):
        print(f"Phase 2: Trying min_prefs_per_kid={min_prefs}, deviation=1.0 (no balance constraint)")
        model, x = create_model(school, processed_data_folder, min_prefs, 1.0)
        solution = solve_model(model, x, results_folder, timestamp, timelimit, min_prefs, 1.0)
        if solution:
            df = format_solution(solution)
            return df, timestamp

    print("No solution found in any configuration.")
    return None, timestamp
