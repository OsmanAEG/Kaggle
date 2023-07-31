import torch
import pandas as pd
from investigate import *
from sklearn.preprocessing import StandardScaler

# data processing class
class Process_Data():
  def __init__(self, data, device, train=True):
    self.is_processed = False
    self.data = data
    self.scalers = {}
    self.train = train
    self.X = None
    self.Y = None
    self.device = device

  def process(self):
    if self.is_processed:
      print('Data has already been processed!')
    else:
      # drop useless info
      self.data = self.data.drop(['PassengerId', 'Name'], axis=1)

      # edit cabin info to remove room number
      remove_room_num = lambda x: \
        x.split('/')[0] + '/' + x.split('/')[2] if isinstance(x, str) else x

      self.data['Cabin'] = self.data['Cabin'].apply(remove_room_num)

      # catergorical and numerical entries
      cat_entries = self.data.select_dtypes(include=['object']).columns
      num_entries = self.data.select_dtypes(include=['int64', 'float64']).columns

      # replace categorical missing entries with mode
      for entry in cat_entries:
        self.data[entry] = self.data[entry].fillna(self.data[entry].mode().iloc[0])

      # replace numerical missing entries with mean
      for entry in num_entries:
        self.data[entry] = self.data[entry].fillna(self.data[entry].mean())

      # convert boolean to int
      self.data['CryoSleep'] = self.data['CryoSleep'].map({True: 1., False: 0.})
      self.data['VIP'] = self.data['VIP'].map({True: 1., False: 0.})

      if self.train:
        self.data['Transported'] = self.data['Transported'].map({True: 1., False: 0.})

      # convert categorical to one-hot
      self.data = pd.get_dummies(self.data,
                                 columns=['HomePlanet', 'Destination', 'Cabin'],
                                 dtype=float)

      columns_to_scale = ['Age', 'RoomService', 'FoodCourt',
                          'ShoppingMall', 'Spa', 'VRDeck']

      # normalize data
      for column in columns_to_scale:
        scaler = StandardScaler()
        if self.train:
          self.data[[column]] = scaler.fit_transform(self.data[[column]])
          self.scalers[column] = scaler
        else:
          self.data[[column]] = self.scalers[column].transform(self.data[[column]])

      # split data into X and Y
      if self.train:
        self.X = self.data.drop(['Transported'], axis=1)
        self.Y = self.data['Transported']
      else:
        self.X = self.data

      self.is_processed = True

  def print_data(self):
    print(self.data.head())

  def print_data_line(self, index):
    print(self.data.iloc[index])

  def get_tensors(self):
    if self.is_processed:
      if self.train:
        return torch.tensor(self.X.values, dtype=torch.float32, device=self.device),\
               torch.tensor(self.Y.values, dtype=torch.float32, device=self.device)
      else:
        return torch.tensor(self.X.values, dtype=torch.float32, device=self.device),\
               None
    else:
      print('Data has not been processed yet!')