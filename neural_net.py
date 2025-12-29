import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np

############## DATA ##############

class census_data(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features.values, dtype=torch.float32)
        self.y = torch.tensor(targets.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def encode_train_test(ds, features):
    for x in features:
        vec = pd.get_dummies(ds[x])
        ds = ds.drop(x, axis = 1)
        ds = pd.concat([ds, vec], axis=1)
        # ds = ds.join(vec)
    return ds


data = pd.read_csv("applied_stats_proj/train_test_2025.csv").drop('education', axis=1)
data = encode_train_test(data, ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
data['over50k'] = pd.factorize(data['over50k'])[0]
data = data.astype(np.float64)
train, test = train_test_split(data, test_size=0.2)

train_df = census_data(train.drop(columns="over50k"), train["over50k"])
training_loader = DataLoader(train_df, batch_size=4, shuffle=True)

test_df = census_data(test.drop(columns="over50k"), test["over50k"])
test_loader = DataLoader(test_df, batch_size=4, shuffle=True)

#TODO: need validation loader from submit_2025


print(train.drop(columns="over50k").head)
print(test.drop(columns="over50k").head)



############## MODEL ##############

# class better_one_l_net(torch.nn.Module):
#     #Constructor
#     def __init__(self, args):
#         super(better_one_l_net, self).__init__()
#         self.args = args
#         self.loss_fn = nn.MSELoss() #TODO: this is only relevant in the training loop, also GraNd uses cross entropy
#         # hidden layer 
#         self.linear_one = torch.nn.Linear(args.input_size, args.hidden_size)
#         self.linear_two = torch.nn.Linear(args.hidden_size, args.output_size) 
#         # defining layers as attributes
#         self.layer_in = None
#         self.act = None
#         self.layer_out = None
#     # prediction function
#     def forward(self, x):
#         self.layer_in = self.linear_one(x)
#         self.act = torch.sigmoid(self.layer_in)
#         self.layer_out = self.linear_two(self.act)
#         y_pred = torch.sigmoid(self.linear_two(self.act))
#         return y_pred

# ############## TRAINING AND TESTING ##############

# def train_one_epoch(loss_fn, epoch_index, tb_writer):
#     running_loss = 0.
#     last_loss = 0.
#     #TODO: below is the architecture for batches
#     for i, data in enumerate(training_loader):
#         # Every data instance is an input + label pair
#         #print(data)
#         X, Y = data
#         optimizer.zero_grad()
#         output = model(X)
#         #Compute loss using loss fn
#         loss = loss_fn(output, Y)
#         #Update weights
#         loss.backward()
#         optimizer.step()

#         #data gathering
#         running_loss += loss.item()
#         if i % 1000 == 999:
#             last_loss = running_loss / 1000 # loss per batch
#             print('  batch {} loss: {}'.format(i + 1, last_loss))
#             tb_x = epoch_index * len(training_loader) + i + 1
#             tb_writer.add_scalar('Loss/train', last_loss, tb_x)
#         running_loss = 0.

#     return last_loss


# def train_full(model):
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     #TODO fix this
#     writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
#     epoch_number = 0

#     EPOCHS = 5 #TODO: use args

#     best_loss_test = 1_000_000.

#     for epoch in range(EPOCHS):
#         print('EPOCH {}:'.format(epoch_number + 1))

#         # Make sure gradient tracking is on, and do a pass over the data
#         model.train(True)
#         avg_loss = train_one_epoch(model.loss_fn, epoch_number, writer)

#         running_loss_test = 0.0
#         running_acc_test = 0.0 #NEW
#         # Set the model to evaluation mode, disabling dropout and using population
#         # statistics for batch normalization.
#         model.eval()

#         # Disable gradient computation and reduce memory consumption.
#         with torch.no_grad():
#             for i, vdata in enumerate(test_loader):
#                 X_test, Y_test = vdata
#                 outputs_test = model(X_test)
#                 loss_test = model.loss_fn(outputs_test, Y_test)
#                 outputs_test = (outputs_test > 0.5).float()
#                 acc_test = torch.mean((outputs_test == Y_test).float())
#                 running_loss_test += loss_test
#                 running_acc_test += acc_test

#         avg_loss_test = running_loss_test / (i + 1)
#         avg_acc_test = running_acc_test / (i + 1)
#         print('LOSS train {} valid {}'.format(avg_loss, avg_loss_test))
#         print('ACC valid {}'.format(avg_acc_test))

#         # Log the running loss averaged per batch
#         # for both training and validation
#         writer.add_scalars('Training vs. Validation Loss',
#                         { 'Training' : avg_loss, 'Validation' : avg_loss_test },
#                         epoch_number + 1)
#                         #TODO: might add accuracy to this
#         writer.flush()

#         # Track best performance, and save the model's state
#         if avg_loss_test < best_loss_test:
#             best_loss_test = avg_loss_test
#             model_path = 'model_{}_{}'.format(timestamp, epoch_number)
#             torch.save(model.state_dict(), model_path)

#         epoch_number += 1

# ############## Test ##############
# def test(model):
#     loss, acc = 0, 0
#     #N = X.shape[0]
#     model.eval()
#     with torch.no_grad():
#         for i, vdata in enumerate(test_loader):
#             X_test, Y_test = vdata
#             logits = model(X_test) #TODO: maybe pass data in instead of referenceing overall, or with some kind of load_data function
#             loss += model.loss_fn(logits, Y_test)
#             logits = (logits > 0.5).float()
#             acc += torch.mean((logits == Y_test).float())
#             #TODO: double check that accuracy is not being calculated wrong
#         loss, acc = loss.item()/(i+1), acc.item()/(i+1)
#     return loss, acc