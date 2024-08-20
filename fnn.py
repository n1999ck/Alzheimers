import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.sigmoid = nn.Sigmoid()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)
        # Non-linearity  # NON-LINEAR
        out = self.sigmoid(out)
        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
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
dataset = dataset[1:]
dataset.pop("DoctorInCharge")

test_portion_dataset = dataset[int((len(dataset) * (7/8))):]
test_portion_dataset_labels = dataset_labels[int((len(dataset) * (7/8))):]
train_portion_dataset = dataset[:int((len(dataset) * (7/8)))]
train_portion_dataset_labels = dataset_labels[:int((len(dataset) * (7/8)))]

print(len(test_portion_dataset))
print(len(test_portion_dataset_labels))
print(len(train_portion_dataset))
print(len(train_portion_dataset_labels))


label_mapping = {
    'PatientID': 0,
    'Age': 1,
    'Gender': 2,  # Assuming Gender might be categorical (e.g., 0 for male, 1 for female)
    'Ethnicity': 3,  # Assuming Ethnicity is categorical
    'EducationLevel': 4,  # Assuming this is categorical
    'BMI': 5,
    'Smoking': 6,  # Assuming Smoking is categorical (e.g., 0 for non-smoker, 1 for smoker)
    'AlcoholConsumption': 7,  # Assuming this might be a numeric score
    'PhysicalActivity': 8,  # Assuming this is a numeric score
    'DietQuality': 9,  # Assuming this is a numeric score
    'SleepQuality': 10,  # Assuming this might be a categorical score
    'FamilyHistoryAlzheimers': 11,  # Assuming this is categorical (e.g., 0 for no, 1 for yes)
    'CardiovascularDisease': 12,  # Assuming this is categorical
    'Diabetes': 13,  # Assuming this is categorical
    'Depression': 14,  # Assuming this is categorical
    'HeadInjury': 15,  # Assuming this is categorical
    'Hypertension': 16,  # Assuming this is categorical
    'SystolicBP': 17,
    'DiastolicBP': 18,
    'CholesterolTotal': 19,
    'CholesterolLDL': 20,
    'CholesterolHDL': 21,
    'CholesterolTriglycerides': 22,
    'MMSE': 23,  # Assuming this is a numeric cognitive score
    'FunctionalAssessment': 24,  # Assuming this is a numeric score
    'MemoryComplaints': 25,  # Assuming this is categorical
    'BehavioralProblems': 26,  # Assuming this is categorical
    'ADL': 27,  # Assuming this is a numeric score
    'Confusion': 28,  # Assuming this is categorical
    'Disorientation': 29,  # Assuming this is categorical
    'PersonalityChanges': 30,  # Assuming this is categorical
    'DifficultyCompletingTasks': 31,  # Assuming this is categorical
    'Forgetfulness': 32,  # Assuming this is categorical
    'Diagnosis': 33,  # Assuming this is categorical (e.g., 0 for no Alzheimer's, 1 for Alzheimer's)
    'DoctorInCharge': 34  # Assuming this is categorical or a non-relevant identifier
}
numeric_labels = dataset_labels.map(label_mapping)


dataset_features = np.vstack(train_portion_dataset.values).astype(np.float32)

test_dataset_features = np.vstack(test_portion_dataset.values).astype(np.float32)
print(type(dataset_features))
print(dataset_features.dtype)
print(dataset_labels)
batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(dataset_features) / batch_size)
num_epochs = int(num_epochs)
print(num_epochs)
input_dim = 33
output_dim = 2
hidden_dim = 100

train_dataset = CSVDataset(dataset_features, train_portion_dataset_labels)
test_dataset = CSVDataset(test_dataset_features, test_portion_dataset_labels)
dataset_loader =torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)

test_loader =torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)

print(dataset_loader)
model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
print(model)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(model.parameters())
print(len(list(model.parameters())))

print(list(model.parameters())[0].size())
print(list(model.parameters())[1].size())

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
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / total
        

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))