import numpy as np
import pandas as pd
import os
import traceback

from lmfit import lm_fit
from ebayes import eBayes

# File paths for the dataset and design matrix
DATASET_FILE = 'synthetic_data.csv'
DESIGN_MATRIX_FILE = 'synthetic_design_matrix.csv'

# Function to generate synthetic data
def generate_synthetic_data(num_genes=50, num_samples=30):
    np.random.seed(42)  # For reproducibility
    exprs = np.random.rand(num_genes, num_samples) * 100  # Random expression values
    gene_ids = [f"probe_{i+1}" for i in range(num_genes)]
    df = pd.DataFrame(exprs.T, index=[f"sample_{i+1}" for i in range(num_samples)], columns=gene_ids)
    return df

# Function to create a design matrix
def create_design_matrix(num_samples=30, sample_names=None):
    intercept = np.ones(num_samples)
    # Create a balanced predictor column with 50% zeros and 50% ones
    half_samples = num_samples // 2
    random_column = np.array([0] * half_samples + [1] * (num_samples - half_samples))
    np.random.shuffle(random_column)  # Shuffle to randomize the order
    design_matrix = np.column_stack((intercept, random_column))
    design_df = pd.DataFrame(design_matrix, index=sample_names, columns=['intercept', 'predictor'])
    return design_df

# Main entry point for testing
if __name__ == "__main__":
    # Check if the dataset file exists
    if os.path.exists(DATASET_FILE) and os.path.exists(DESIGN_MATRIX_FILE):
        # Read the dataset and design matrix from files
        data = pd.read_csv(DATASET_FILE, index_col=0)
        design_matrix = pd.read_csv(DESIGN_MATRIX_FILE, index_col=0)
    else:
        # Generate synthetic data
        data = generate_synthetic_data()
        # Save the dataset to a CSV file
        data.to_csv(DATASET_FILE)
       
        # Create a design matrix with matching sample names
        design_matrix = create_design_matrix(data.shape[0], sample_names=data.index)
        # Save the design matrix to a CSV file
        design_matrix.to_csv(DESIGN_MATRIX_FILE)

    # Run the lm_fit function
    try:
        fit_results = lm_fit(data, design=design_matrix, method="ls")
        print("\nFitting Results:")
        #print(fit_results["df_residual"])
    except Exception as e:
        print("An error occurred:")
        print(f"Exception: {e}")
        print("Traceback:")
        traceback.print_exc()  # This will print the full traceback including line numbers

    if fit_results is not None:
        try:
            results = eBayes(fit_results)
            #print(results["df_residual"])
        except Exception as e:
            print(f"Application of ebayes failed: Exception {e}")
            traceback.print_exc()

