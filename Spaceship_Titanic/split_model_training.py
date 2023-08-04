import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from data_processor import Process_Data
from model import Model_Arch

# device selection
device = torch.device('cuda')

# setup data
train_set = Process_Data(pd.read_csv('train.csv'), device, train=True)
train_set.process()

# split data
X_train, Y_train, X_test, Y_test = train_set.split_train_data()

N_in = X_train.shape[1]
N_out = 1

# setup model
model = Model_Arch(N_in, N_out).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train model
epochs = 100

model.train()
for epoch in range(epochs):
  optimizer.zero_grad()
  outputs = model(X_train)
  loss = criterion(outputs.squeeze(), Y_train)
  loss.backward()
  optimizer.step()

  if (epoch + 1) % 10 == 0:
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

# predict
model.eval()

with torch.no_grad():
  predictions = model(X_test)

outputs = torch.round(predictions).squeeze()

accuracy = (outputs == Y_test).float().mean()
print(f'Accuracy: {accuracy.item() * 100:.2f}%')

