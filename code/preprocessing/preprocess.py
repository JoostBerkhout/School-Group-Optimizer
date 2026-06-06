import pandas as pd
import os
import warnings
from validate_data import validate_grouping_data
from helpers import InputData, read_variables

warnings.filterwarnings('ignore', category=UserWarning, message='.*Data Validation extension is not supported.*')

def read_data(excel_path):
    # Define sheet names
    sheets = pd.read_excel(excel_path, sheet_name=['Info Docenten', 'Info Leerlingen', 'Groepswensen', 'Eigen Indelingen'], skiprows=1)

    # Read the sheets as dataframes
    info_teachers = sheets['Info Docenten']
    info_students = sheets['Info Leerlingen']
    group_preferences = sheets['Groepswensen']
    current_groups = sheets['Eigen Indelingen']

    # Find the starting indices for each table
    start_table_1 = group_preferences.index[0] + 1
    start_table_2 = group_preferences[group_preferences.iloc[:, 0] == 'Naam Leerling 1'].index[0] + 2
    start_table_3 = group_preferences[group_preferences.iloc[:, 0] == 'Naam Leerling'].index[0] + 2

    # Find the end indices for each table based on the next table's start
    end_table_1 = start_table_2 - 2
    end_table_2 = start_table_3 - 2
    end_table_3 = len(group_preferences) + 1

    # Read each table into a separate dataframe using skiprows and nrows
    group_preferences = pd.read_excel(excel_path, sheet_name='Groepswensen', skiprows=start_table_1, nrows=end_table_1 - start_table_1, header=None)
    constraints_students = pd.read_excel(excel_path, sheet_name='Groepswensen', skiprows=start_table_2, nrows=end_table_2 - start_table_2)
    constraints_teachers = pd.read_excel(excel_path, sheet_name='Groepswensen', skiprows=start_table_3, nrows=end_table_3 - start_table_3)

    # Format the group preferences table
    group_preferences = group_preferences.iloc[1:].reset_index(drop=True).T
    group_preferences.columns = group_preferences.iloc[0]
    group_preferences = group_preferences.drop(0).reset_index(drop=True)
    group_preferences = group_preferences.astype(int)

    return info_teachers, info_students, group_preferences, constraints_students, constraints_teachers, current_groups

def translate_dfs(info_teachers, info_students, group_preferences, constraints_students, constraints_teachers, current_groups):
    # Translate the column names to English
    info_teachers.columns = ['Teacher']
    group_preferences.columns = ['Number of Students', 'Number of Groups', 'Minimum Group Size', 'Maximum Number Extra Care']
    constraints_students.columns = ['Student 1', 'Student 2', 'Together']
    constraints_teachers.columns = ['Student', 'Teacher', 'Together']
    current_groups.columns = ['Student', 'Teacher']

    # Only translate Behavior column if it exists: I think here it allows for
    # different formats of the template data with extra care columns
    info_columns = ['Student', 'Grade', 'Gender', 'Extra Care', 'Preference 1', 'Preference 2', 'Preference 3', 'Preference 4', 'Preference 5']
    if info_students.shape[1] == 10:
        info_columns.insert(4, 'Behavior')
    if info_students.shape[1] == 12:
        info_columns[4:4] = ["Behavior", "Learning", "Combination"]

    info_students.columns = info_columns

    # Translate values in dataframes to English
    info_students['Gender'] = info_students['Gender'].replace({'Jongen': 'Boy', 'Meisje': 'Girl'})
    info_students['Extra Care'] = info_students['Extra Care'].replace({'Ja': 'Yes', 'Nee': 'No'})
    if 'Behavior' in info_students.columns:
        info_students['Behavior'] = info_students['Behavior'].replace({'Ja': 'Yes', 'Nee': 'No'})
    if 'Learning' in info_students.columns:
        info_students['Learning'] = info_students['Learning'].replace({'Ja': 'Yes', 'Nee': 'No'})
    if 'Combination' in info_students.columns:
        info_students['Combination'] = info_students['Combination'].replace({'Ja': 'Yes', 'Nee': 'No'})


    constraints_students['Together'] = constraints_students['Together'].replace({'Ja': 'Yes', 'Nee': 'No'})
    constraints_teachers['Together'] = constraints_teachers['Together'].replace({'Ja': 'Yes', 'Nee': 'No'})

    return info_teachers, info_students, group_preferences, constraints_students, constraints_teachers, current_groups

def strip_whitespace(df):
    for col in df.columns:
        # Only string/mixed type columns
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: x.strip().capitalize() if isinstance(x, str) else x)
    return df



def save_dataframes_to_csv(school, data, processed_data_folder):
    # Check if school folder exists
    school_processed_folder = os.path.join(processed_data_folder, school)
    os.makedirs(school_processed_folder, exist_ok=True)

    dfs = {
        'group_preferences': data.group_preferences,
        'info_students': data.info_students,
        'info_teachers': data.info_teachers,
        'constraints_students': data.constraints_students,
        'constraints_teachers': data.constraints_teachers,
        'current_groups': data.current_groups
    }

    # Save each dataframe to a CSV file
    for df_name, df in dfs.items():
        file_path = os.path.join(school_processed_folder, f'{df_name}.csv')
        df.to_csv(file_path, index=False, na_rep='')


def preprocess(school, raw_data_folder, processed_data_folder):
    # Get file path
    school_path = os.path.join(raw_data_folder, school)
    excel_file = [f for f in os.listdir(school_path) if f.endswith('.xlsx')][0]
    excel_path = os.path.join(school_path, excel_file)

    # Read in data
    info_teachers, info_students, group_preferences, constraints_students, constraints_teachers, current_groups = read_data(excel_path)

    # Translate data
    info_teachers, info_students, group_preferences, constraints_students, constraints_teachers, current_groups = translate_dfs(info_teachers, info_students, group_preferences, constraints_students, constraints_teachers, current_groups)

    # Strip whitespace from all dataframes
    info_teachers = strip_whitespace(info_teachers)
    info_students = strip_whitespace(info_students)
    group_preferences = strip_whitespace(group_preferences)
    constraints_students = strip_whitespace(constraints_students)
    constraints_teachers = strip_whitespace(constraints_teachers)
    current_groups = strip_whitespace(current_groups)

    data = InputData(
        group_preferences,
        info_students,
        info_teachers,
        constraints_students,
        constraints_teachers,
        current_groups
    )

    variables = read_variables(data)

    is_valid = validate_grouping_data(data, variables)
    if is_valid == False:
        exit("Grouping data for {} is invalid. Please check the errors above.".format(school))
    else:
        print("Grouping data for {} is valid.".format(school))

    # Save the processed data
    save_dataframes_to_csv(school, data, processed_data_folder)
    print("Data preprocessing for {} completed.".format(school))


def run_preprocess(raw_data_folder, processed_data_folder):
    for school in os.listdir(raw_data_folder):
        if school == '.DS_Store' or school == 'school_1':
            print("Skipping file: {}".format(school))
            continue

        # Only run preprocessing for school if it is not already processed
        school_processed_folder = os.path.join(processed_data_folder, school)
        if os.path.exists(school_processed_folder):
            print("Data for {} already processed.".format(school))
            continue

        print("Running preprocessing for {}".format(school))

        preprocess(school, raw_data_folder, processed_data_folder)


if __name__ == "__main__":
    # Define paths
    raw_data_folder = 'data/raw_data'
    processed_data_folder = 'data/processed_data'

    # Run preprocessing
    run_preprocess(raw_data_folder, processed_data_folder)













