import csv
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def process_data(input_data_file_path, pct_cutoff, train_size, val_size, test_size, output_file_path):
    # Read the CSV file and store data
    data = []
    with open(input_data_file_path, 'r', encoding='unicode_escape') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert diff_score_with_a_agg to float and include only rows with valid diff_score_with_a_agg
            try:
                row['diff_score_with_a_agg'] = float(row['diff_score_with_a_agg'])
                # Ensure 'message_id', 'userid', 'ifpid', and 'year' are numeric
                numeric_fields = ['message_id', 'userid', 'ifpid', 'year']
                for field in numeric_fields:
                    row[field] = pd.to_numeric(row[field], errors='coerce')

                # Check if any of the numeric fields are NaN after conversion
                if any(pd.isna(row[field]) for field in numeric_fields):
                    continue  # Skip rows with invalid numeric fields

                # Add valid rows to the dataset
                data.append(row)
            except (ValueError, KeyError):
                # Skip rows with invalid 'diff_score_with_a_agg' or missing keys
                continue
    data = pd.DataFrame(data)
    # split data pairwise into labels 0 vs 1 
    data['classification_label'] = None  # Initialize the label column with default values
    grouped_by_question = data.groupby('ifpid')
    x = pct_cutoff
    for ifpid, group in grouped_by_question:
        # group into subgoups by date_posted
        grouped_by_question_and_date = group.groupby('date_posted')
        for date, subgroup in grouped_by_question_and_date:

        # smaller diff_score_with_a_agg = better
        # get the top x% of grouped_by_question_and_date sorted by diff_score_with_a_agg and set its label to 0 and the bottom x% label to 1

            # Sort the subgroup by 'diff_score_with_a_agg'
            subgroup = subgroup.sort_values(by='diff_score_with_a_agg', ascending=True)

            # Calculate the number of rows corresponding to x%
            num_rows = len(subgroup)
            top_x_count = int(num_rows * x)
            bottom_x_count = int(num_rows * x)
            
            # Set labels for the top x% (smallest diff_score_with_a_agg)
            if top_x_count > 0:
                subgroup.iloc[:top_x_count, subgroup.columns.get_loc('classification_label')] = 1
            
            # Set labels for the bottom x% (largest diff_score_with_a_agg)
            if bottom_x_count > 0:
                subgroup.iloc[-bottom_x_count:, subgroup.columns.get_loc('classification_label')] = 0

            # save the modified subgroup back to the original dataframe
            data.loc[subgroup.index, 'classification_label'] = subgroup['classification_label']

    # print(data[["ifpid", "date_posted", "diff_score_with_a_agg", "classification_label"]].sort_values(by=['ifpid', 'date_posted', 'diff_score_with_a_agg']).head(50))
    data = data[data['classification_label'].notna()]
    print(f"total data size: {len(data)}")
    print(f"total ones: {data['classification_label'].sum()}")
    print(f"total zeros: {len(data) - data['classification_label'].sum()}")


    # Split the data into training, validation, and test sets
    df_train, df_temp = train_test_split(data, test_size=val_size+test_size, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=test_size/(val_size+test_size), random_state=42)

    # # Get unique `ifpid` values
    # unique_ifpids = data['ifpid'].unique()

    # # Split the unique ifpids into train, validation, and test groups
    # train_ifpids, temp_ifpids = train_test_split(unique_ifpids, test_size=val_size+test_size, random_state=42)
    # val_ifpids, test_ifpids = train_test_split(temp_ifpids, test_size=test_size/(val_size+test_size), random_state=42)

    # # Assign data rows based on the split
    # df_train = data[data['ifpid'].isin(train_ifpids)]
    # df_val = data[data['ifpid'].isin(val_ifpids)]
    # df_test = data[data['ifpid'].isin(test_ifpids)]

    # Print label counts for training data
    print(f"training -- total data size: {len(df_train)}, num ones: {df_train['classification_label'].sum()}, num_zeros: {len(df_train) - df_train['classification_label'].sum()}")

    # Print label counts for validation data
    print(f"validation -- total data size: {len(df_val)}, num ones: {df_val['classification_label'].sum()}, num_zeros: {len(df_val) - df_val['classification_label'].sum()}")
    
    # Print label counts for testing data
    print(f"test -- total data size: {len(df_test)}, num ones: {df_test['classification_label'].sum()}, num_zeros: {len(df_test) - df_test['classification_label'].sum()}")
    

    # Create dictionaries for JSON output
    data_dict_train = {
        "texts": [row['body'] for _, row in df_train.iterrows()],
        "labels": [row['classification_label'] for _, row in df_train.iterrows()],
        "diff_score_with_a_aggs": [row['diff_score_with_a_agg'] for _, row in df_train.iterrows()],
    }

    data_dict_val = {
        "texts": [row['body'] for _, row in df_val.iterrows()],
        "labels": [row['classification_label'] for _, row in df_val.iterrows()],
        "diff_score_with_a_aggs": [row['diff_score_with_a_agg'] for _, row in df_val.iterrows()],
    }

    data_dict_test = {
        "texts": [row['body'] for _, row in df_test.iterrows()],
        "labels": [row['classification_label'] for _, row in df_test.iterrows()],
        "diff_score_with_a_aggs": [row['diff_score_with_a_agg'] for _, row in df_test.iterrows()],
    }

    # Save training data to JSON
    train_output_file = f"{output_file_path}_train.json"
    with open(train_output_file, "w") as json_file:
        json.dump(data_dict_train, json_file, indent=4)
    print(f"Training data saved to {train_output_file}")

    # Save validation data to JSON
    val_output_file = f"{output_file_path}_val.json"
    with open(val_output_file, "w") as json_file:
        json.dump(data_dict_val, json_file, indent=4)
    print(f"Validation data saved to {val_output_file}")

    # Save testing data to JSON
    test_output_file = f"{output_file_path}_test.json"
    with open(test_output_file, "w") as json_file:
        json.dump(data_dict_test, json_file, indent=4)
    print(f"Testing data saved to {test_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV data for rationale analysis.")
    parser.add_argument("input_data_file_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("pct_cutoff", type=float, help="Percentage of top and bottom forecastes per question to set as 0 and 1.")
    parser.add_argument("train_size", type=float, help="Percentage of samples for training")
    parser.add_argument("val_size", type=float, help="Percentage of samples for validation")
    parser.add_argument("test_size", type=float, help="Percentage of samples for testing")
    parser.add_argument("output_file_path", type=str, help="Base output file path for saving JSON files.")
    args = parser.parse_args()

    process_data(
        input_data_file_path=args.input_data_file_path,
        pct_cutoff=args.pct_cutoff,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        output_file_path=args.output_file_path
    )