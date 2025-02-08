# forecasting_rationale_analysis


## **Usage Instructions**
nlparam/nlparam_classification.py -- for running entire workflow of data processing, getting train/val/test split, and running nlparam classification

nlparam/nlparam_get_correlation.py -- for running correlation using the nlparam way (using nlparam llm validation prompt rather than the Chris prompt (update variables at top of file before running

Chris_cor_eval/eval_features.py -- for running correlation using the Chris pipeline

data_processing/data_process_nlparam -- for running data processing and train/val/test split. Called in nlparam/nlparam_classification.py. (NOTE: newer copy of data_process_nlparam_new.py splits data using diff_score_with_a_agg instead of raw_score. But need to ensure the version of forecasts data file used has the diff_score_with_a_agg column)


workbook.ipynb -- has useful adhoc code for 
1. estimating cost of nlparam_classification runs (2 different methods of estimating, read code to understand more. I usually estimate using both methods as sanity check -- they should be approx similar.)
2. parsing nlparam's parameters (gotten at the bottom of the nlparam_classifican run's log file) from a chunck of text to csv file and reformat to be same format as Chris's pipeline.
3. plotting correlation charts from the Wandb logged correlation files (input files downloaded directly from wandb Training and Validation correlation tables)

