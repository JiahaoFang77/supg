from selection_query import query_syntax, calculatePR
from Algorithm import supg_query
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import argparse
import numpy as np

def main(query):
    # Create a SQLAlchemy engine
    engine = create_engine('sqlite:///dataset.db')

    # Execute a SQL query to read the proxy scores and oracle labels
    df = pd.read_sql_query('SELECT * FROM Beta2', engine)

    # Convert the DataFrame columns to numpy arrays
    A = df['proxy_scores'].to_numpy()
    O = df['oracle_labels'].to_numpy()

    # Parse the command line argument
    query = query_syntax (query)
    print(query)

    n_records = 10**6

    # record = supg_query(n_records, A, O, query['ORACLE LIMIT'], query['target'], query['delta'], query['proxy_estimates'])
    # record.sort()
    # record_accurate = [i for i in range(n_records) if O[i] == 1]
    # calculatePR(record, record_accurate) 

    precisions = []
    recalls = []

    for i in range(40):
        record = supg_query(n_records, A, O, query['ORACLE LIMIT'], query['target'], query['delta'], query['proxy_estimates'])
        record.sort()
        record_accurate = [j for j in range(n_records) if O[j] == 1]
        precision, recall = calculatePR(record, record_accurate)
        precisions.append(precision)
        recalls.append(recall)

    x_values = list(range(1,41))
    
    if query['proxy_estimates'] == 'PRECISION':
        measures_name = 'Precision'
        measures_values = precisions
    else:  
        measures_name = 'Recall'
        measures_values = recalls
    
    # Calculate mean and standard deviation
    mean_value = np.mean(measures_values)
    std_dev_value = np.std(measures_values)
    print(f'Mean {measures_name}: {mean_value}')
    print(f'Standard Deviation of {measures_name}: {std_dev_value}')
    
    # Plotting
    fig, ax = plt.subplots()

    ax.bar(x_values, measures_values, align='center', alpha=0.5, capsize=10)
    ax.set_xlabel('Trial times')
    ax.set_ylabel(measures_name)
    ax.set_title(f'{measures_name} v.s. Trial Times with a {measures_name.lower()} target of 95%')
    ax.yaxis.grid(True)
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(k) for k in x_values])


    ax.set_ylim(bottom=0.98)
    ax.set_ylim(bottom=0.85)
    
    # Draw a red horizontal line that represents the mean
    ax.axhline(y=mean_value, color='b', linestyle='-', label='Mean')
    ax.legend() # Show legend

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create a ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument("query", help="SUPG query string")

    # Parse arguments
    args = parser.parse_args()

    # Run main function with query argument
    main(args.query)
