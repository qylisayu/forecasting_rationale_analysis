import csv
import json
import argparse

def process_data(input_data_file_path, upper_cutoff, lower_cutoff, train_N, output_file_path):
    # Read the CSV file and store data
    data = []
    with open(input_data_file_path, 'r', encoding='unicode_escape') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert norm_score to float and include only rows with valid norm_score
            try:
                row['norm_score'] = float(row['norm_score'])
                data.append(row)
            except ValueError:
                continue  # Skip rows with invalid norm_score

    # Create the classification labels
    processed_data = []
    for row in data:
        norm_score = row['norm_score']
        if norm_score > upper_cutoff:
            row['classification_label'] = 0
        elif norm_score < lower_cutoff:
            row['classification_label'] = 1
        else:
            continue  # Skip rows that don't meet the cutoff criteria
        processed_data.append(row)

    # Split the data into training and testing sets
    df_train = processed_data[:train_N]
    df_test = processed_data[train_N:]

    # Print label counts for training data
    train_label_counts = {0: 0, 1: 0}
    for row in df_train:
        train_label_counts[row['classification_label']] += 1
    print("Training data label counts:")
    print(train_label_counts)

    # Print label counts for testing data
    test_label_counts = {0: 0, 1: 0}
    for row in df_test:
        test_label_counts[row['classification_label']] += 1
    print("Testing data label counts:")
    print(test_label_counts)

    # Create dictionaries for JSON output
    data_dict_train = {
        "texts": [row['body'] for row in df_train],
        "labels": [row['classification_label'] for row in df_train],
    }

    data_dict_test = {
        "texts": [row['body'] for row in df_test],
        "labels": [row['classification_label'] for row in df_test],
    }

    # Save training data to JSON
    train_output_file = f"{output_file_path}_train.json"
    with open(train_output_file, "w") as json_file:
        json.dump(data_dict_train, json_file, indent=4)
    print(f"Training data saved to {train_output_file}")

    # Save testing data to JSON
    test_output_file = f"{output_file_path}_test.json"
    with open(test_output_file, "w") as json_file:
        json.dump(data_dict_test, json_file, indent=4)
    print(f"Testing data saved to {test_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV data for rationale analysis.")
    parser.add_argument("input_data_file_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("upper_cutoff", type=float, help="Upper cutoff value for classification.")
    parser.add_argument("lower_cutoff", type=float, help="Lower cutoff value for classification.")
    parser.add_argument("train_N", type=int, help="Number of samples for training.")
    parser.add_argument("output_file_path", type=str, help="Base output file path for saving JSON files.")
    args = parser.parse_args()

    process_data(
        input_data_file_path=args.input_data_file_path,
        upper_cutoff=args.upper_cutoff,
        lower_cutoff=args.lower_cutoff,
        train_N=args.train_N,
        output_file_path=args.output_file_path
    )