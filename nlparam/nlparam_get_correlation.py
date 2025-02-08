import numpy as np
import pandas as pd
from nlparam.llm.validate import validate_descriptions
from nlparam import get_validator_by_name
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt


DATA_NAME = "binary_first_two_weeks_9_24_24"
FEATURE_FILE_NAME = "natural_parameters_01-20"
DESCRIPTIONS_FILE_PATH = f"../data_files/features/{FEATURE_FILE_NAME}.csv"
TEXTS_FILE_PATH = f"../data_files/forecasts/{DATA_NAME}.csv"
VALIDATOR = "gpt-4o-mini"
BATCH_SIZE = 1000


# load descriptions
descriptions_df = pd.read_csv(DESCRIPTIONS_FILE_PATH)
descriptions = descriptions_df['explanation'].tolist()
description_summaries = descriptions_df["natural parameter"].tolist()

# # load texts
# with open(TEXTS_FILE_PATH, 'rb') as f:
#     content = f.read().decode('unicode_escape', errors='replace')  # or use a different encoding

# # Create a DataFrame
# binary_data = pd.read_csv(StringIO(content), on_bad_lines='skip', engine='python')

binary_data = pd.read_csv(TEXTS_FILE_PATH)
print(binary_data.head())


# Convert 'message_id', 'userid', and 'ifpid' to numeric, and drop rows with NaN after conversion
binary_data = binary_data[
    binary_data[['message_id', 'userid', 'ifpid', 'year']].apply(
        lambda x: pd.to_numeric(x, errors='coerce')
    ).notna().all(axis=1)
]

# Convert the columns to integer type after filtering
binary_data[['message_id', 'userid', 'ifpid', 'year']] = binary_data[['message_id', 'userid', 'ifpid', 'year']].astype('int64')

binary_data = binary_data[:BATCH_SIZE]

texts = binary_data["body"].tolist()


# initialized validator
validator = get_validator_by_name(VALIDATOR)

# get denotations
validation_predicate_denotation = validate_descriptions(
                descriptions=descriptions,  
                texts=texts,                    
                validator=validator,                
                progress_bar=True                       
            )
# np.ndarray; A matrix of scores, which the i-th row and j-th column is how well the i-th text satisfies the j-th description.

print(f"descriptions: {descriptions}")
print(f"texts (len = {len(texts)})")

# get correlation matrix between "raw_score", "norm_score", "accuracy_score", "score_diff_a_avg" and all the descriptions
scores = binary_data[["raw_score", "norm_score", "accuracy_score", "score_diff_a_avg"]]

# Create a DataFrame for the denotation matrix
denotation_df = pd.DataFrame(
    validation_predicate_denotation, 
    columns=description_summaries
)

perturbation = np.random.normal(loc=0, scale=1e-6, size=denotation_df.shape)  # Gaussian noise
denotation_df_perturbed = denotation_df + perturbation

combined_df = pd.concat([denotation_df_perturbed, scores.reset_index(drop=True)], axis=1)

correlation_matrix = combined_df.corr()

sns.set(rc={'figure.figsize':(50, 50)})
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Selected Columns")
plt.savefig(f"../data_files/outputs/{DATA_NAME}_{FEATURE_FILE_NAME}_{BATCH_SIZE}_correlation_matrix.png", dpi=100)



