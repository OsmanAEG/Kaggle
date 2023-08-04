import pandas as pd
import torch
import matplotlib.pyplot as plt

#investigate train data
def investigate_data(data):
  unique_counts = data.nunique()
  data_types    = data.dtypes

  print(pd.DataFrame({'Unique Count': unique_counts, 'Data Type': data_types}))

  # check how many rows have missing values
  num_rows_missing = 0

  for i, row in data.iterrows():
    for j, item in enumerate(row):
      if pd.isna(item):
        num_rows_missing += 1
        break

  print(f'Number of rows with missing values: {num_rows_missing}')

#describe data
def describe_data(data):
  print(data.describe(include='all'))
  print('-------------------------------------------------------------------')

#print data head
def print_data(data):
  print(data.head())

#print data line
def print_data_line(data, index):
  print(data.iloc[index])

#check data for empty elements
def check_empty(data):
  for row_idx, row in data.iterrows():
    for col_idx, item in enumerate(row):
      if pd.isna(item):
        print(f'Missing value found at row {row_idx}, column {col_idx}')

#check data for NaN or Inf values
def check_nan_inf(data):
  for i, row in enumerate(data):
    for j, elem in enumerate(row):
      if torch.isnan(elem) or torch.isinf(elem):
        print(f'NaN or Inf found at row {i}, column {j}')
