import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
STEP 1: LOADING DATASET
'''
# Create test and train datasets    
dataset = pd.read_csv('Dataset.csv', encoding="Latin-1") #ISO-8859-1 used in basic latin. UTF-8 for anything else
dataset_labels = dataset.pop("Diagnosis")
dataset.pop("DoctorInCharge")
dataset.pop("PatientID") 

train_portion_dataset = dataset[:int(len(dataset) * (7/8))]
test_portion_dataset = dataset[int(len(dataset) * (7/8)):]
train_portion_dataset_labels = dataset_labels[:int(len(train_portion_dataset))]
test_portion_dataset_labels = dataset_labels[int(len(train_portion_dataset)):]
actual = np.array(test_portion_dataset_labels)

print("Testing dataset:\t {}".format(len(test_portion_dataset)))
print("Testing datalabels:\t {}".format(len(test_portion_dataset_labels)))
print("Training dataset:\t {}".format(len(train_portion_dataset)))
print("Training datalabels:\t {}".format(len(train_portion_dataset_labels)))

train_dataset_features = train_portion_dataset.to_numpy().astype(np.float32)
test_dataset_features = test_portion_dataset.to_numpy().astype(np.float32)

mean_train = train_dataset_features.mean(axis=0)
std_train = train_dataset_features.std(axis=0)
mean_test = test_dataset_features.mean(axis=0)
std_test = test_dataset_features.std(axis=0)

train_dataset_features = (train_dataset_features - mean_train) / std_train
test_dataset_features = (test_dataset_features - mean_test) / std_test

'''
STEP 2: MAKING DATASET ITERABLE
'''

class CSVDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

train_dataset = CSVDataset(train_dataset_features, train_portion_dataset_labels)
test_dataset = CSVDataset(test_dataset_features, test_portion_dataset_labels)

batch_size = 8
dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=4, shuffle=True)

val_split = 0.2
val_size = int(len(train_portion_dataset) * val_split)
train_size = len(train_portion_dataset) - val_size
print("Validation_size:\t{}".format(val_size))
print("Training_size:\t\t{}".format(train_size))
 
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,[train_size, val_size])
print("Validation_size:\t{}".format(len(val_dataset)))
print("Training_size:\t\t{}".format(len(train_dataset)))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)


n_iters = 1880*5
num_epochs = int(n_iters / (len(train_dataset) / batch_size)) # ((n_iters)/ (1504/32)) = n_iters / 47... n_iters = epochs * 47
print("Epochs: {}".format(num_epochs))
input_dim = int(len(dataset.columns)) #32 as of now
output_dim = 1
hidden_dim = 24

'''
STEP 3: CREATE MODEL CLASS
'''
class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)     
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()                     
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
model = FNN(input_dim, hidden_dim,output_dim)

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.BCELoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
 
'''
STEP 7: TRAIN THE MODEL
''' 
accuracies = []
iterations = []
iter = 0
for epoch in range(num_epochs):
    train_loss = train_acc = total_train_acc = 0
    val_loss = val_acc = total_val_acc = 0
    train_total = 0
    for i, (fields, labels) in enumerate(train_loader):
        model.train()
        output = model(fields).squeeze(1)
        loss = criterion(output, labels) 
        train_loss += loss.item()
        train_acc = (output.round() == labels).sum().item()
        total_train_acc += train_acc
        train_total += labels.size(0)

        train_acc = 100 * total_train_acc / train_total
        train_loss = train_loss / train_total
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        iter += 1
        if((iter % (188*5)) == 0):
            val_total = 0
            with torch.no_grad():
                for fields, labels in val_loader:     
                    output = model(fields).squeeze(1)
                    loss = criterion((output), labels) 
                    val_loss += loss.item()
                    val_acc = (output.round() == labels).sum().item()
                    total_val_acc += val_acc
                    val_total += labels.size(0)
            val_acc = 100 * total_val_acc / val_total
            val_loss = val_loss / val_total
            print("Iteration: {}, Training Loss: {}, Training accuracy: {}".format(iter, train_loss, train_acc))
            print("Iteration: {}, Validation Loss: {}, Validation accuracy: {}".format(iter, val_loss, val_acc))

'''
STEP 8: TEST THE MODEL
'''
test_loss = test_acc = total_test_acc = test_total = 0
with torch.no_grad():  
    for fields, labels in test_loader:
        output = model(fields).squeeze(1)
        loss = criterion((output), labels) 
        test_loss += loss.item()
        test_acc = (output.round() == labels).sum().item()
        total_test_acc += test_acc
        test_total += labels.size(0)
test_acc = 100 * total_test_acc / test_total
test_loss = test_loss / test_total
print("Testing Loss: {}, Testing accuracy: {}".format(test_loss, test_acc))

'''
Feature Scaling
-   Read
https://medium.com/@punya8147_26846/understanding-feature-scaling-in-machine-learning-fe2ea8933b66
https://datascience.stackexchange.com/questions/27615/should-we-apply-normalization-to-test-data-as-well
-   To Read
https://builtin.com/data-science/when-and-why-standardize-your-data
https://www.geeksforgeeks.org/logistic-regression-and-the-feature-scaling-ensemble/

https://medium.com/@rsvmukhesh/determining-the-number-of-epochs-d8b3526d8d06
'''