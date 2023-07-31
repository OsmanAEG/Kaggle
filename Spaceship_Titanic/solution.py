import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from data_processor import Process_Data
from model import Model_Arch

'''pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)'''

# Check if CUDA is available and print device information
if torch.cuda.is_available():
  device = torch.device('cuda')
  print('DEVICE: ' + torch.cuda.get_device_name(0))
else:
  device = torch.device('cpu')
  print('DEVICE: CPU')

# setup data
train_set = Process_Data(pd.read_csv('train.csv'), device, train=True)
test_set  = Process_Data(pd.read_csv('test.csv'), device, train=False)

# process data
train_set.process()

test_set.scalers = train_set.scalers
test_set.process()

# convert to tensors
X_train, Y_train = train_set.get_tensors()
X_test,  _       = test_set.get_tensors()

# setup model parameters
N_in = X_train.shape[1]
N_out = 1

# setup model
model = Model_Arch(N_in, N_out).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train model
epochs = 1000

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

outputs = torch.round(predictions)

# write to csv
id_test = pd.read_csv('test.csv')['PassengerId']

with open('submission.csv', 'w') as f:
    f.write('PassengerId,Transported\n')
    for i in range(len(outputs)):
        is_transported = True if outputs[i][0] == 1 else False
        f.write(str(id_test[i]) + ',' + str(is_transported) + '\n')