import torch
import pandas as pd

from investigate import *
from visualize import *

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    # check if data has already been processed
    if self.is_processed:
      print('Data has already been processed!')
    else:
      # forward fill missing entries for name and cabin
      for i, col in enumerate(['Name', 'Cabin']):
        self.data[col] = self.data[col].fillna(method='ffill')

      # replace these missing categorical entries with mode
      for col in ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']:
        self.data[col] = self.data[col].fillna(self.data[col].mode().iloc[0])

      # replace these missing numerical entries with mean
      for col in ['Age']:
        self.data[col] = self.data[col].fillna(self.data[col].mean())

      # replace these missing numerical entries with median
      for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        self.data[col] = self.data[col].fillna(self.data[col].median())

      # split cabin info into deck, room, and side
      for i, col in enumerate(['Deck', 'Room', 'Side']):
        split_func = lambda x: x.split('/')[i] if pd.notnull(x) else None
        self.data[col] = self.data['Cabin'].apply(split_func)

      # split passenger id into group and number
      for i, col, in enumerate(['Group', 'Number']):
        split_func = lambda x: x.split('_')[i] if pd.notnull(x) else None
        self.data[col] = self.data['PassengerId'].apply(split_func)

      # splitting last name from name
      split_func = lambda x: x.split()[1] if pd.notnull(x) else None
      self.data['LastName'] = self.data['Name'].apply(split_func)

      # checking if the passenger is traveling with family
      self.data['FamilyOnBoard'] = False

      for idx, group in self.data.groupby(['LastName', 'Group']):
        if len(group) > 1:
          self.data.loc[group.index, 'FamilyOnBoard'] = True

      # checking if the passenger is an adult
      self.data['Adult'] = False
      self.data.loc[self.data['Age'] >= 18, 'Adult'] = True

      # split cabin info into deck, room, and side
      for i, col in enumerate(['Deck', 'Room', 'Side']):
        split_func = lambda x: x.split('/')[i] if pd.notnull(x) else None
        self.data[col] = self.data['Cabin'].apply(split_func)

      # total money spent by each passenger
      self.data['MoneySpent'] = 0.0
      for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        self.data['MoneySpent'] += self.data[col]

      # convert bool to string
      for col in ['FamilyOnBoard', 'CryoSleep', 'VIP', 'Adult']:
        self.data[col] = self.data[col].astype(str)

      # make room numerical
      self.data['Room'] = pd.to_numeric(self.data['Room'], errors='coerce')

      # make room numerical
      self.data['Room'] = pd.to_numeric(self.data['Room'], errors='coerce')

      # highlighting categorical and numerical entries
      cat_entries = self.data.select_dtypes(include=['object']).columns
      num_entries = self.data.select_dtypes(include=['int64', 'float64']).columns

      # normalize numerical data
      for column in num_entries:
        scaler = StandardScaler()
        if self.train:
          self.data[[column]]  = scaler.fit_transform(self.data[[column]])
          self.scalers[column] = scaler
        else:
          self.data[[column]] = self.scalers[column].transform(self.data[[column]])

      #print_data_line(self.data, 2)
      #visualize_data(self.data)
      #investigate_data(self.data)
      #assert 1 == 2

      # data to drop
      data_to_drop = [
                      'PassengerId',
                      #'HomePlanet',
                      #'CryoSleep',
                      'Cabin',
                      #'Destination',
                      #'Age',
                      #'VIP',
                      #'RoomService',
                      #'FoodCourt',
                      #'ShoppingMall',
                      #'Spa',
                      #'VRDeck',
                      'Name',
                      #'Deck',
                      #'Room',
                      #'Side',
                      'Group',
                      'Number',
                      'LastName',
                      'FamilyOnBoard',
                      #'Adult',
                      #'MoneySpent'
                      ]

      self.data = self.data.drop(columns=data_to_drop, axis=1)

      # convert transport to 1 and 0
      if self.train:
        self.data['Transported'] = self.data['Transported'].map({True: 1., False: 0.})

      # updating categorical and numerical entries
      cat_entries = self.data.select_dtypes(include=['object']).columns
      num_entries = self.data.select_dtypes(include=['int64', 'float64']).columns

      # convert categorical to one-hot
      self.data = pd.get_dummies(self.data, columns=cat_entries, dtype=float)

      # split data into X and Y
      if self.train:
        self.X = self.data.drop(['Transported'], axis=1)
        self.Y = self.data['Transported']
      else:
        self.X = self.data

      self.is_processed = True

  def split_train_data(self, ts=0.2):
    if self.is_processed:
      data_for_split = pd.concat([self.X, self.Y], axis=1)
      train_data, test_data = train_test_split(data_for_split, test_size=ts)

      X_train_data = train_data.drop(['Transported'], axis=1)
      Y_train_data = train_data['Transported']

      X_test_data = test_data.drop(['Transported'], axis=1)
      Y_test_data = test_data['Transported']

      X_train = torch.tensor(X_train_data.values, dtype=torch.float32,
                             device=self.device)

      Y_train = torch.tensor(Y_train_data.values, dtype=torch.float32,
                             device=self.device)

      X_test = torch.tensor(X_test_data.values, dtype=torch.float32,
                            device=self.device)

      Y_test = torch.tensor(Y_test_data.values, dtype=torch.float32,
                            device=self.device)

      return X_train, Y_train, X_test, Y_test

    else:
      print('Data has not been processed yet!')

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