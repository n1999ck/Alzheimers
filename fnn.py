import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)  
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, output_dim)  
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.sigmoid(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.sigmoid(out)
        out = self.dropout(out)

        out = self.fc3(out)
        return out

class RecurrentNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RecurrentNeuralNetModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):

        #Init hidden state with 0s
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        #Detach hidden state to avoid exploding gradients
        out, _ = self.lstm(x, (h0, c0))

        #index hidden state of last time step
        #out.size() => 100, 20, 10
        # We just want last time step hidden states
        out = self.fc(out[:, -1, :])
        return out

class CSVDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create test and train datasets    
dataset = pd.read_csv('Dataset.csv', encoding="ISO-8859-1")
dataset_labels = dataset.pop("Diagnosis")
dataset.pop("DoctorInCharge")

test_portion_dataset = dataset[int((len(dataset) * (7/8))):]
test_portion_dataset_labels = dataset_labels[int((len(dataset) * (7/8))):]
train_portion_dataset = dataset[:int((len(dataset) * (7/8)))]
train_portion_dataset_labels = dataset_labels[:int((len(dataset) * (7/8)))]


print(len(test_portion_dataset))
print(len(test_portion_dataset_labels))
print(len(train_portion_dataset))
print(len(train_portion_dataset_labels))

dataset_features = np.vstack(train_portion_dataset.values).astype(np.float32)
test_dataset_features = np.vstack(test_portion_dataset.values).astype(np.float32)


mean = dataset_features.mean(axis=0)
std = dataset_features.std(axis=0)
dataset_features = (dataset_features - mean) / std
test_dataset_features = (test_dataset_features - mean) / std

batch_size = 100
n_iters = 30000
num_epochs = n_iters / (len(dataset_features) / batch_size)
num_epochs = int(num_epochs)
print(num_epochs)
input_dim = 33
output_dim = 2
hidden_dim = 500
layer_dim = 2

train_dataset = CSVDataset(dataset_features, train_portion_dataset_labels)
test_dataset = CSVDataset(test_dataset_features, test_portion_dataset_labels)
dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size = batch_size,)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size = batch_size)


model = RecurrentNeuralNetModel(input_dim, hidden_dim, layer_dim, output_dim)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

 
val_split = 0.2
val_size = int(len(train_portion_dataset) * val_split)
train_size = len(train_portion_dataset) - val_size
print(val_size)
print(train_size)
 
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,[train_size, val_size])
print("Length of train_dataset:" + str(len(train_dataset)))
print("Length of val_dataset:" + str(len(val_dataset)))
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

accuracies = []
iterations = []
iter = 0
for epoch in range(num_epochs):
    for i, (fields, labels) in enumerate(dataset_loader):
        if len(fields.shape) == 2:
            fields = fields.unsqueeze(1)  # Add sequence dimension if missing
        
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
            model.eval()
            with torch.no_grad():
                for fields, labels in val_loader:
                    if len(fields.shape) == 2:
                        fields = fields.unsqueeze(1)  # Add sequence dimension if missing
        
                    
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
fig, ax = plt.subplots()
plt.figure(figsize=(12,6))

ax.plot(iterations, accuracies, label="Validation set accuracy %")
ax.set(xlabel="Iteration", ylabel="Accuracy %",
       title="Accuracy over time")
ax.grid()
plt.legend()

model.eval()
correct = 0
total = 0
with torch.no_grad():  # Disable gradient calculation
    for fields, labels in test_loader:
        if len(fields.shape) == 2:
            fields = fields.unsqueeze(1)  # Add sequence dimension if missing
        
        outputs = model(fields)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy}%')

with open('accuracies.txt', 'a') as accuraciesFile:
    accuraciesFile.write(str(test_accuracy) + '\n')

with open('accuracies.txt', 'r') as accuraciesFile:
    accuraciesList = [float(line.strip()) for line in accuraciesFile]
    print(accuraciesList)

ax = fig.add_subplot(1,2,1)
plt.plot(range(1,len(accuraciesList) + 1), accuraciesList, 'ro-', label="Test accuracy %")
ax.plot(iterations[-1], test_accuracy, 'ro-', label="Test accuracy %")
accuraciesList = []



ax.legend()
plt.tight_layout()
fig.savefig("test.png")
plt.show()