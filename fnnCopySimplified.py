import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, output_dim)  
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class CSVDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

# Create test and train datasets    
dataset = pd.read_csv('Dataset.csv', encoding="ISO-8859-1")
dataset_labels = dataset.pop("Diagnosis")
dataset.pop("DoctorInCharge")

dataset_labels_array = np.array(dataset_labels)
# 35.36% positive 64.64% negative

test_portion_dataset = dataset[int((len(dataset) * (7/8))):]
test_portion_dataset_labels = dataset_labels[int((len(dataset) * (7/8))):]
actual = np.array(test_portion_dataset_labels)
train_portion_dataset = dataset[:int((len(dataset) * (7/8)))]
train_portion_dataset_labels = dataset_labels[:int((len(dataset) * (7/8)))]

dataset_features = train_portion_dataset.to_numpy().astype(np.float32)
test_dataset_features = test_portion_dataset.to_numpy().astype(np.float32)

scaler = StandardScaler()
train_portion_dataset = scaler.fit_transform(train_portion_dataset)
test_portion_dataset = scaler.fit_transform(test_portion_dataset)

batch_size = 100
n_iters = 2000
num_epochs = int(n_iters / (len(dataset_features) / batch_size))
input_dim = 33
output_dim = 2
hidden_dim = 512
layer_dim = 2

train_dataset = CSVDataset(dataset_features, train_portion_dataset_labels)
test_dataset = CSVDataset(test_dataset_features, test_portion_dataset_labels)
dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size = batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size = batch_size, shuffle=True)


model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

accuracies = []
iterations = []
iter = 0
for epoch in range(num_epochs):
    for i, (fields, labels) in enumerate(dataset_loader):
        optimizer.zero_grad()
        outputs = model(fields)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        iter += 1
        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for fields, labels in test_loader:

                # Forward pass only to get logits/output
                outputs = model(fields)
                
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                
                # Total number of labels
                total += labels.size(0)
                
                # Total correct predictions
                correct += (predicted == labels).sum().item()
        
            accuracy = 100 * correct / total
            accuracies.append(accuracy)
            iterations.append(iter)
            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

print(accuracies)

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy}%') 