import json
from typing import List
from nlparam import run_classification
import subprocess
import argparse
import wandb


if __name__ == '__main__':
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description="Run NLParam classification with data processing.")
    parser.add_argument("data_file_name", type=str, help="Name of the input data file.")
    parser.add_argument("pct_cutoff", type=float, help="Percentage of top and bottom forecastes per question to set as 0 and 1.")
    parser.add_argument("train_size", type=float, help="Percentage of samples for training")
    parser.add_argument("val_size", type=float, help="Percentage of samples for validation")
    parser.add_argument("test_size", type=float, help="Percentage of samples for testing")
    parser.add_argument("K", type=int, help="Number of predicates")
    parser.add_argument("num_iterations", type=int, help="Number of iterations for model training.")
    parser.add_argument("lr", type=float, help="Learning rate for optimizing phi.")
    args = parser.parse_args()

    data_file_name = args.data_file_name
    pct_cutoff = args.pct_cutoff
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    K: int = args.K
    num_iterations = args.num_iterations
    goal: str = "Here are some forecasters' reationales for their predictions. I want to understand what features are important for accurate forecasts."
    lr: float = args.lr


    data_dir = "../data_processing/"
    input_data_file_path = data_dir + data_file_name
    output_file_path = data_dir + "rationale_analysis_nlparam_classification_input_data"
    
    # process data
    try:
        # Construct the command to call the script
        command = [
            "python", "../data_processing/data_process_nlparam.py",
            input_data_file_path, str(pct_cutoff),
            str(train_size), str(val_size), str(test_size), output_file_path
        ]
        
        # Execute the script as a subprocess
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Print output and error (optional)
        print("Output:", result.stdout)
        print("Errors:", result.stderr)
    
    except subprocess.CalledProcessError as e:
        # Handle errors during the subprocess call
        print("Error occurred while processing data:")
        print(e.stderr)
        raise
    
    # command line prompt user if they would like to continue to running the model. Only continue if user says yes
    # Prompt user for confirmation
    user_input = input("Would you like to continue running the model? (yes/no): ").strip().lower()

    # Check the user's response
    if user_input in ['yes', 'y']:
        print("Continuing to run the model...")
        # Place your model execution code here
    else:
        print("Exiting the program. Model will not run.")
        exit()  # Exit the program
        
    # run model
    # Initialize W&B
    wandb.init(
            project="nlparam",
            name="rationale_analysis_nlparam_classification",
            config={
                "data_file_name": data_file_name,
                "pct_cutoff": pct_cutoff,
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
                "K": K,
                "num_iterations": num_iterations,
                "goal": goal,
                "lr": lr
                },
            reinit=True # to allow multiple runs in the same script
    )

    train_data_path = output_file_path + "_train.json"
    val_data_path = output_file_path + "_val.json"
    test_data_path = output_file_path + "_test.json"

    with open(train_data_path, "r") as f:
        task_dict = json.load(f)

    texts: List[str] = task_dict["texts"]
    labels: List[int] = task_dict["labels"]
    scores: List[float] = task_dict["raw_scores"]

    with open(val_data_path, "r") as f:
        val_task_dict = json.load(f)

    val_texts: List[str] = val_task_dict["texts"]
    val_labels: List[int] = val_task_dict["labels"]
    val_scores: List[float] = val_task_dict["raw_scores"]

    with open(test_data_path, "r") as f:
        test_task_dict = json.load(f)

    test_texts: List[str] = test_task_dict["texts"]
    test_labels: List[int] = test_task_dict["labels"]
    test_scores: List[float] = test_task_dict["raw_scores"]
    
    # runnning the classification model
    clf_result = run_classification(texts, labels, scores, 
                                    val_texts, val_labels, val_scores, 
                                    test_texts, test_labels, test_scores,
                                    K, goal, num_iterations, lr)

    w = clf_result["w"]
    predicates = clf_result["predicates"]
    for i, predicate in enumerate(predicates):
        print(f"{predicate}: {w[0][i]}")
    
    print("done.")

    wandb.finish()