import numpy as np
from scipy.stats import bernoulli
import pandas as pd
from sqlalchemy import create_engine

class SyntheticDataset:
    def __init__(self, alpha_beta_pairs, n_records):
        self.alpha = alpha_beta_pairs[0]
        self.beta = alpha_beta_pairs[1]
        self.n_records = n_records

    # Proxy model generating scores from Beta distribution
    def A(self):
        return np.random.beta(self.alpha, self.beta, self.n_records)

    # Oracle model generating labels from Bernoulli distribution
    def O(self, scores):
        return bernoulli.rvs(scores)

    # Function to generate the dataset
    def generate_dataset(self):
        proxy_scores = self.A()
        oracle_labels = self.O(proxy_scores)
        return proxy_scores, oracle_labels

# Set the seed for reproducibility
np.random.seed(0)

# Define the alpha and beta parameters for the Beta distributions
alpha_beta_pairs = (0.01, 2)

# Define the number of records
n_records = 10**6

# Create the synthetic dataset generator
dataset_generator = SyntheticDataset(alpha_beta_pairs, n_records)

# Generate the dataset
proxy_scores, oracle_labels = dataset_generator.generate_dataset()

# Save the data in a DataFrame
df = pd.DataFrame({
    'proxy_scores': proxy_scores,
    'oracle_labels': oracle_labels
})

# Save the DataFrame to a SQLite database
engine = create_engine('sqlite:///dataset.db')
df.to_sql('Beta2', engine, if_exists='replace', index=False)
