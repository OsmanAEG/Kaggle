import matplotlib.pyplot as plt
import seaborn as sns
import os
#import warnings

#warnings.filterwarnings("ignore", category=UserWarning)

def visualize_data(data_in):
  # get categorical and numerical entries
  cat_entries = data_in.select_dtypes(include=['object']).columns
  num_entries = data_in.select_dtypes(include=['int64', 'float64']).columns

  # create directories for plots
  cat_plot_dir = 'cat_plots'
  num_plot_dir = 'num_plots'

  # create directories for plots
  if not os.path.exists(cat_plot_dir):
    os.makedirs(cat_plot_dir)

  if not os.path.exists(num_plot_dir):
    os.makedirs(num_plot_dir)

  # plotting categorical data
  for col in cat_entries:
    plt.figure(figsize=(8, 6))
    sns.countplot(data= data_in, x= col)
    plt.title(f'{col} count')
    plt.savefig(os.path.join(cat_plot_dir, str(col) + '.png'))
    plt.close()

  # plotting numerical data
  for col in num_entries:
    plt.figure(figsize=(8, 6))
    sns.displot(data_in[col])
    plt.title(f'{col} distribution')
    plt.savefig(os.path.join(num_plot_dir, str(col) + '.png'))
    plt.close()