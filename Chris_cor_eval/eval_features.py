import argparse
import json
import os
import pandas as pd
from openai import OpenAI
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def load_api_key(key_path):
    """Load the OpenAI API key from a CSV file."""
    chatgpt_key = pd.read_csv(key_path, encoding="UTF-8")
    return chatgpt_key.loc[0, 'Key']


def prepare_prompts(forecast_file, feature_file, batch_size):
    """Prepare prompts for each row in the input data."""
    with open(forecast_file, 'rb') as f:
        content = f.read().decode('unicode_escape', errors='replace')

    binary_data = pd.read_csv(StringIO(content), on_bad_lines='skip', engine='python')

    # Clean data
    binary_data = binary_data[
        binary_data[['message_id', 'userid', 'ifpid', 'year']].apply(
            lambda x: pd.to_numeric(x, errors='coerce')
        ).notna().all(axis=1)
    ]
    binary_data[['message_id', 'userid', 'ifpid', 'year']] = binary_data[['message_id', 'userid', 'ifpid', 'year']].astype('int64')

    llm_models = pd.read_csv(feature_file, encoding="UTF-8")

    binary_data['GPT_generate_scores'] = binary_data.apply(
        lambda row: (
            f"Below we have the following data components of a geopolitical forecast: "
            f"(1) a forecasting question, (2) the specific answer options, "
            f"(3) the formal resolution criteria, (4) the date and probability forecast of a forecaster, "
            f"and (5) the forecaster's rationale. "
            f"We would like to score the rationale on a few dozen patterns, with scores for each pattern "
            f"being either 0 (no evidence of pattern present in rationale), 1 (some evidence, but incomplete or mixed), "
            f"or 2 (clear evidence of the pattern). "
            f"The pipe-delimited patterns include:\n\n"
            f"{' || '.join(llm_models['Model_name'] + ': ' + llm_models['Description'])}\n\n"
            f"Return scores for all {len(llm_models)} patterns, with exact titles/case from the prompt. "
            f"Also, provide a one-sentence description explaining the forecaster's reasoning, including specific details.\n\n"
            f"Here are the input data: the forecasting question (1) is '{row['q_text']}' with answer options (2) of {row['options']}. "
            f"The formal resolution criteria (3) are: '{row['q_text']}'. "
            f"On {row['date_posted']}, a forecaster gave a forecast of {row['a_option_fcast'] * 100:.0f}% (4) "
            f"and provided the following rationale (5): '{row['body']}'.\n"
        ),
        axis=1
    )

    binary_data = binary_data.head(batch_size)

    return binary_data


def create_jsonl_file(binary_data, file_name):
    with open(file_name, 'w') as f:
        for i, row in binary_data.iterrows():
            body_content = {
                "custom_id": f"request-{i}-{row['year']}-{row['userid']}-{row['message_id']}-{row['ifpid']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a forecast rationale scorer. Your task is to evaluate forecasts based on specified patterns and provide scores. "
                                "Provide the output strictly by calling the function 'forecast_scorecard' with the correct arguments."
                            )
                        },
                        {"role": "user", "content": row['GPT_generate_scores']}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "forecast_scorecard",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "measures_scores": {
                                        "type": "array",
                                        # "uniqueItems": True,
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "pattern": {
                                                    "type": "string",
                                                    "description": "Exact name of the pattern taken from prompt"
                                                },
                                                "score": {
                                                    "type": "integer",
                                                    "description": "A score of 0, 1, or 2",
                                                    "enum": [0, 1, 2]
                                                }
                                            },
                                            "required": ["pattern", "score"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "reasoning_explanation": {
                                        "type": "string",
                                        "description": "A one-sentence description that explains the forecaster's reasoning for the question, including specific details and not just broad reasoning patterns."
                                    }
                                },
                                "required": ["measures_scores", "reasoning_explanation"],
                                "additionalProperties": False
                            }
                        }
                    }
                }
            }
            f.write(json.dumps(body_content) + '\n')


def submit_batch(client, batch_input_file, feature_file, forecast_file, batch_size):
    """Submit a batch job to OpenAI."""
    uploaded_file = client.files.create(file=open(batch_input_file, "rb"), purpose="batch")
    batch_job = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "ACE LLM Analysis Batch Job",
                  "batch_input_file_id": uploaded_file.id,
                  "feature_file": feature_file,
                  "forecast_file": forecast_file,
                  "batch_size": str(batch_size)}
    )
    return batch_job.id


def retrieve_batch(client, batch_id):
    """Retrieve batch results by ID and save to a file."""
    batch = client.batches.retrieve(batch_id)

    if batch.status == "completed":
        output_file_id = batch.output_file_id
        file_response = client.files.content(output_file_id)

        batch_size = batch.metadata['batch_size']
        feature_file = batch.metadata['feature_file']
        forecast_file = batch.metadata['forecast_file']
        forecast_name = os.path.splitext(os.path.basename(forecast_file))[0]
        feature_name = os.path.splitext(os.path.basename(feature_file))[0]
        batch_output_file = f"{forecast_name}_{feature_name}_batch_output_raw_{batch_size}.jsonl"

        with open(batch_output_file, 'w') as f:
            f.write(file_response.text)
        print(f"Batch results saved to {batch_output_file}")

        return batch_output_file, batch
    else:
        print(f"Batch {batch_id} is not completed. Current status: {batch.status}")

        return None, batch


def create_df_from_batch_output(batch_output_file):
    data_list = []

    # Open and read the JSONL file
    with open(batch_output_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            custom_id = data['custom_id']
            response_content = data['response']['body']['choices'][0]['message']['content']

            # Split the custom_id into its components
            parts = custom_id.split('-')
            request_number = parts[1] if len(parts) > 1 else None
            year = parts[2] if len(parts) > 2 else None
            userid = parts[3] if len(parts) > 3 else None
            message_id = parts[4] if len(parts) > 4 else None
            ifpid = parts[5] if len(parts) > 5 else None

            # Append the parsed data as a dictionary
            data_list.append({
                'request_number': request_number,
                'year': year,
                'userid': userid,
                'message_id': message_id,
                'ifpid': ifpid,
                'response_content': response_content
            })

    # Create a pandas DataFrame
    df = pd.DataFrame(data_list)
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Processing Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a batch job")
    submit_parser.add_argument("--key", type=str, required=True, help="Path to the API key file")
    submit_parser.add_argument("--forecast_file", type=str, required=True, help="Path to the forecasts file")
    submit_parser.add_argument("--feature_file", type=str, required=True, help="Path to the features file")
    submit_parser.add_argument("--batch_size", type=int, required=True, help="Batch size")

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve batch results")
    retrieve_parser.add_argument("--key", type=str, required=True, help="Path to the API key file")
    retrieve_parser.add_argument("--batch_id", type=str, required=True, help="Batch ID to retrieve results for")

    args = parser.parse_args()

    # Load API key
    api_key = load_api_key(args.key)
    client = OpenAI(api_key=api_key)

    if args.command == "submit":
        forecast_name = os.path.splitext(os.path.basename(args.forecast_file))[0]
        feature_name = os.path.splitext(os.path.basename(args.feature_file))[0]
        raw_input_file = f"{forecast_name}_{feature_name}_batch_input_raw_{args.batch_size}.jsonl"

        data = prepare_prompts(args.forecast_file, args.feature_file, args.batch_size)
        create_jsonl_file(data, raw_input_file)
        batch_id = submit_batch(client, raw_input_file, args.feature_file, args.forecast_file, args.batch_size)
        print(f"Batch submitted successfully. Batch ID: {batch_id}")

    elif args.command == "retrieve":
        batch_output_file, batch = retrieve_batch(client, args.batch_id)
        if batch_output_file != None:
            batch_response_df = create_df_from_batch_output(batch_output_file)
            batch_size = batch.metadata['batch_size']
            feature_file = batch.metadata['feature_file']
            forecast_file = batch.metadata['forecast_file']
            forecast_name = os.path.splitext(os.path.basename(forecast_file))[0]
            feature_name = os.path.splitext(os.path.basename(feature_file))[0]

            # plot correlation

            with open(forecast_file, 'rb') as f:
                content = f.read().decode('unicode_escape', errors='replace')  # or use a different encoding

            binary_data = pd.read_csv(StringIO(content), on_bad_lines='skip', engine='python')

            # Convert 'message_id', 'userid', and 'ifpid' to numeric, and drop rows with NaN after conversion
            binary_data = binary_data[
                binary_data[['message_id', 'userid', 'ifpid', 'year']].apply(
                    lambda x: pd.to_numeric(x, errors='coerce')
                ).notna().all(axis=1)
            ]

            # Convert the columns to integer type after filtering
            binary_data[['message_id', 'userid', 'ifpid', 'year']] = binary_data[['message_id', 'userid', 'ifpid', 'year']].astype('int64')


            # Function to extract measures_scores and reasoning_explanation
            def extract_measures_and_reasoning(response_content):
                response = json.loads(response_content)
                measures = {item["pattern"]: item["score"] for item in response["measures_scores"]}
                measures["reasoning_explanation"] = response["reasoning_explanation"]
                return measures
            
            # Apply extraction and expand the DataFrame
            expanded_data = batch_response_df["response_content"].apply(extract_measures_and_reasoning)
            expanded_df = pd.concat([batch_response_df.drop(columns=["response_content"]), expanded_data.apply(pd.Series)], axis=1)
            expanded_df[['request_number', 'year', 'userid', 'message_id', 'ifpid']] = expanded_df[['request_number', 'year', 'userid', 'message_id', 'ifpid']].apply(pd.to_numeric, errors='coerce').astype('int64')
            merged_df = expanded_df.merge(binary_data[['year', 'userid', 'message_id', 'ifpid', 'norm_score', 'accuracy_score', 'score_diff_a_avg']], 
                              on=['year', 'userid', 'message_id', 'ifpid'], 
                              how='left')
            
            # Exclude specified columns from the DataFrame for correlation
            columns_to_exclude = ['request_number', 'year', 'userid', 'message_id', 'ifpid', 'reasoning_explanation']
            filtered_df = merged_df.drop(columns=columns_to_exclude, errors='ignore')

            # add perturbation to avoid NaN correlation

            # List of columns to exclude from perturbation
            exclude_columns = ['norm_score', 'accuracy_score', 'score_diff_a_avg']

            # Create a copy of the DataFrame to avoid modifying the original
            df_perturbed = filtered_df.copy()

            # Identify columns to perturb
            columns_to_perturb = [col for col in df_perturbed.columns if col not in exclude_columns]

            # Add small perturbation to the selected columns
            perturbation = np.random.normal(0, 1e-6, size=(df_perturbed[columns_to_perturb].shape))
            df_perturbed[columns_to_perturb] += perturbation

            # Compute the correlation matrix
            correlation_matrix = df_perturbed.corr()

            sns.set(rc={'figure.figsize':(50, 50)})
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Matrix of Selected Columns")
            plt.savefig(f"{forecast_name}_{feature_name}_{batch_size}_correlation_matrix.png", dpi=100)

            # smaller norm_score = better prediction

        else:
            print(f"Batch not completed. Current status: {batch.status}")