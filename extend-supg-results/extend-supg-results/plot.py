import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# list your csv files
target_recalls = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99] # [40, 45, 50, 55, 60, 65] #
data_path = "C:/6_5_aidb/aidb-private-main/examples/supg_example/extend-supg-results"
csv_files = []
for t in target_recalls:
  csv_files.append(f"{data_path}/results_trecall{t}_conf_95_bgt_2000.csv")

dataframes = []
for file in csv_files:
  # load the data
  df = pd.read_csv(file)
   
  # assuming your target recall is part of your filename as 'trecall50', 'trecall60', etc.
  target_recall = int(file.split('trecall')[1].split('_')[0])
   
   # create a new column for target recall
  df['target_recall'] = target_recall
   
  dataframes.append(df)

# concatenate all dataframes
df = pd.concat(dataframes)
# print(df)
# create a box plot
box_plot = sns.boxplot(x='target_recall', y='recall', data=df)

# draw horizontal red line within each box for corresponding target recall value
# for i, t_recall in enumerate(target_recalls):
#     box_plot.plot([i - 0.4, i + 0.4], [t_recall, t_recall], 'r--') 

# set labels and title
plt.xlabel('Target Recall')
plt.ylabel('Achieved Recall')
plt.title('TASTI Proxy, Confidence 95%, 100 trials')

# show the plot
plt.show()


convergence_dict = {}

# Iterate through all target recalls
for t_recall in target_recalls:
  # Filter dataframe for specific target recall
  df_target = df[df['target_recall'] == t_recall]
 
  # Count how many times achieved recall is greater than target recall
  converge_count = len(df_target[df_target['recall'] >= t_recall])
 
   # Calculate convergence rate
  convergence_rate = converge_count / len(df_target)
 
  # Store the convergence rate in dictionary
  convergence_dict[t_recall] = convergence_rate

# Convert dictionary to pandas DataFrame for easier viewing
df_convergence = pd.DataFrame(list(convergence_dict.items()), columns=['Target Recall', 'Convergence Rate'])

print(df_convergence)