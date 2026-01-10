import pandas as pd
import torch
import torch.nn as nn # type: ignore
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler # type: ignore
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nn_args import MlpArguments, TrainingArguments
import datasets
import matplotlib.pyplot as plt

############## DATA ##############

class census_data(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features.values, dtype=torch.float32)
        self.y = torch.tensor(targets.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def encode_train_test_onehot(ds, features):
    non_onehot =  list( set(list(ds)) - set(features) )
    for x in features:
        vec = pd.get_dummies(ds[x])
        ds = ds.drop(x, axis = 1)
        ds = pd.concat([ds, vec], axis=1)

    # Standardize the data
    scaler = StandardScaler()
    for x in non_onehot:
        ds[x] = scaler.fit_transform(pd.DataFrame(ds[x]))
    return ds

def encode_train_test_num(df, features):
    # Encode the data
    ds = datasets.Dataset.from_pandas(df)
    for x in features:
        ds = ds.class_encode_column(x)
        cast = ds.features[x]
        print("casted %s" % (x))
    
    # Standardize the data
    df = ds.to_pandas()
    headers = list(df)
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=headers)
    return df


data = pd.read_csv("train_test_2025.csv").drop('education', axis=1)
Ys = pd.factorize(data['over50k'])[0]
data['over50k'] = Ys
data = encode_train_test_onehot(data, ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
data['over50k'] = Ys
data = data.astype(np.float64)
train, test = train_test_split(data, test_size=0.2)

train, finetune = train_test_split(train, test_size=0.4)

train_df = census_data(train.drop(columns="over50k"), train["over50k"])
training_loader = DataLoader(train_df, batch_size=4, shuffle=True)

test_df = census_data(test.drop(columns="over50k"), test["over50k"])
test_loader = DataLoader(test_df, batch_size=4, shuffle=True)

finetune_df = census_data(finetune.drop(columns="over50k"), finetune["over50k"])
finetuning_loader = DataLoader(finetune_df, batch_size=4, shuffle=True)

#TODO: need validation loader from submit_2025
# valid = pd.read_csv("submit_2025.csv").drop('education', axis=1)
# valid = encode_train_test(valid, ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
# valid['over50k'] = pd.factorize(valid['over50k'])[0]
# valid = valid.astype(np.float64)
# valid_df = census_data(valid.drop(columns="over50k"), valiud["over50k"])
# valid_loader = DataLoader(valid_df, batch_size=4, shuffle=True)


print(train.head)
# print(test.drop(columns="over50k").head)
print(test_df)




############## MODEL ##############

class better_one_l_net(torch.nn.Module):
    #Constructor
    def __init__(self, args):
        super(better_one_l_net, self).__init__()
        self.args = args
        self.loss_fn = nn.MSELoss()
        # hidden layer 
        self.linear_one = torch.nn.Linear(args.input_size, args.hidden_size)
        self.linear_two = torch.nn.Linear(args.hidden_size, args.output_size) 
        # defining layers as attributes
        self.layer_in = None
        self.act = None
        self.layer_out = None
    # prediction function
    def forward(self, x):
        self.layer_in = self.linear_one(x)
        self.act = torch.sigmoid(self.layer_in)
        self.layer_out = self.linear_two(self.act)
        y_pred = torch.sigmoid(self.linear_two(self.act))
        return y_pred

# ############## TRAINING AND TESTING ##############

def train_one_epoch(loss_fn, epoch_index):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        X, Y = data
        optimizer.zero_grad()
        output = model(X)
        #Compute loss using loss fn
        loss = loss_fn(output, Y)
        #Update weights
        loss.backward()
        optimizer.step()

        #data gathering
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
        running_loss = 0.

    return last_loss

def train_full(model):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #TODO fix this
    epoch_number = 0

    EPOCHS = 5 #TODO: use args

    best_loss_test = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model.loss_fn, epoch_number)

        running_loss_test = 0.0
        running_acc_test = 0.0 #NEW
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
                X_test, Y_test = vdata
                outputs_test = model(X_test)
                loss_test = model.loss_fn(outputs_test, Y_test)
                outputs_test = (outputs_test > 0.5).float()
                acc_test = torch.mean((outputs_test == Y_test).float())
                running_loss_test += loss_test
                running_acc_test += acc_test

        avg_loss_test = running_loss_test / (i + 1)
        avg_acc_test = running_acc_test / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_loss_test))
        print('ACC valid {}'.format(avg_acc_test))

        # Track best performance, and save the model's state
        if avg_loss_test < best_loss_test:
            best_loss_test = avg_loss_test

        epoch_number += 1

# ############## Test ##############
def test(model, data_loader):
    loss, acc = 0, 0
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(data_loader):
            X_test, Y_test = vdata
            logits = model(X_test)
            loss += model.loss_fn(logits, Y_test)
            logits = (logits > 0.5).float()
            acc += torch.mean((logits == Y_test).float())
        loss, acc = loss.item()/(i+1), acc.item()/(i+1)
    return loss, acc
    
def roc(model, df):
    model.eval()
    with torch.no_grad():
        X_test = df.X
        Y_test = df.y
        probabilities = model(X_test)
        y_score = probabilities.squeeze(-1).detach().numpy() 
        fpr, tpr, threshold = roc_curve(Y_test, y_score)
    return fpr, tpr, threshold

############## MEZO FineTuning ##############

def zo_perturb_parameters(model, epsilon, random_seed=None, scaling_factor=1):
    """
    Perturb the parameters with random vector z.
    Input: 
    - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use model.zo_random_seed)
    - scaling_factor: theta = theta + scaling_factor * z * eps
    """

    # Set the random seed to ensure that we sample the same z for perturbation/update
    torch.manual_seed(random_seed if random_seed is not None else model.zo_random_seed)
    
    for param in model.parameters():
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        param.data = param.data + scaling_factor * z * epsilon

def zo_step(model, X, y, epsilon):
    """
    Estimate gradient by MeZO. Return the loss from f(theta + z)
    """

    # Sample the random seed for sampling z
    model.zo_random_seed = np.random.randint(1000000000)

    # First function evaluation
    zo_perturb_parameters(model, epsilon, scaling_factor=1)
    output1 = model.forward(X)
    #Compute loss using (Mean Squared Error)
    loss1 = model.loss_fn(output1, y)

    # Second function evaluation
    zo_perturb_parameters(model, epsilon, scaling_factor=-2)
    output2 = model.forward(X)
    #Compute loss using (Mean Squared Error)
    loss2 = model.loss_fn(output2, y)

    model.projected_grad = ((loss1 - loss2) / (2 * epsilon)).item()

    # Reset model back to its parameters at start of step
    zo_perturb_parameters(model, epsilon, scaling_factor=1)
    
    return loss1

def zo_update(model, lr):
    """
    Update the parameters with the estimated gradients.
    """
    # Reset the random seed for sampling zs
    torch.manual_seed(model.zo_random_seed)     

    for param in model.parameters():
        # Resample z
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        newparam = param.data - lr * (model.projected_grad * z)
        param.data = newparam

def finetune(model, X, y, args):
    """
    almost the same as train().
    """
    losses = []
    for epoch in range(args.zo_epochs):
        loss = zo_step(model, X, y, args.epsilon)
        losses.append(loss)
        zo_update(model, args.zo_lr)
    return losses


############## RUN ##############

## Make Model and Args
args = MlpArguments(
    input_size=87, #TODO: assign this programatically again
    hidden_size = 50,#4,
    output_size = 1, #classes
    loss_fn = nn.MSELoss()#nn.BCELoss()
)
model = better_one_l_net(args)

## Train Model
trainArgs = TrainingArguments
optimizer = torch.optim.SGD(model.parameters(), trainArgs.lr)
train_full(model)

## Test Model
loss, acc = test(model, test_loader)
print('loss: {}, acc: {}'.format(loss, acc))

## Finetune Model and Retest
tunelosses = finetune(model, finetune_df.X, finetune_df.y, trainArgs)

loss, acc = test(model, test_loader)
print('loss: {}, acc: {}'.format(loss, acc))

## ROC and AUC
fpr, tpr, threshold = roc(model, test_df)
# print(fpr)

plt.plot(fpr,tpr,marker='.')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate' )
plt.show()

print(auc(fpr, tpr))