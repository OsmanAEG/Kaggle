import pandas as pd
import torch

#investigate train data
def investigate_data(data):
  unique_counts = data.nunique()
  data_types    = data.dtypes

  print(pd.DataFrame({'Unique Count': unique_counts, 'Data Type': data_types}))

  # check how many rows have missing values
  num_rows_missing = 0

  for row_idx, row in data.iterrows():
    for col_idx, item in enumerate(row):
      if pd.isna(item):
        num_rows_missing += 1
        break

  print(f'Number of rows with missing values: {num_rows_missing}')

'''
Result:
-------------------------
                           Unique Count Data Type
CryoSleep                             2   float64
Age                                  80   float64
VIP                                   2   float64
RoomService                        1273   float64
FoodCourt                          1507   float64
ShoppingMall                       1115   float64
Spa                                1327   float64
VRDeck                             1306   float64
Transported                           2     int64
HomePlanet_Earth                      2   float64
HomePlanet_Europa                     2   float64
HomePlanet_Mars                       2   float64
Destination_55 Cancri e               2   float64
Destination_PSO J318.5-22             2   float64
Destination_TRAPPIST-1e               2   float64

Number of rows with missing values: 2087 (too many to drop)

Description of data:
       PassengerId HomePlanet CryoSleep    Cabin  Destination    VIP            Name
count         8693       8492      8476     8494         8511   8490            8493
unique        8693          3         2     6560            3      2            8473
top        0001_01      Earth     False  G/734/S  TRAPPIST-1e  False  Gollux Reedall
freq             1       4602      5439        8         5915   8291               2

PassengerId = useless
Cabin = ignore room number
Name = useless
'''

#describe data
def describe_data(data):
  print(data.describe(include='all'))
  print('-------------------------------------------------------------------')

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
