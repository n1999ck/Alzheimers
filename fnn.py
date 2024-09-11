import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Define a feedforward neural network model
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Define the network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second fully connected layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer
        self.relu1 = nn.ReLU()  # ReLU activation for the first hidden layer
        self.relu2 = nn.ReLU()  # ReLU activation for the second hidden layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for the output layer
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer to prevent overfitting
    
    def forward(self, x):
        # Forward pass through the network
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout(out)  # Apply dropout
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout(out)  # Apply dropout
        out = self.fc3(out)
        out = self.sigmoid(out)  # Apply sigmoid activation
        return out

# Define a custom dataset class for loading data
class CSVDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)  # Convert features to tensor
        self.labels = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)  # Convert labels to tensor and add a dimension
        
    def __len__(self):
        return len(self.features)  # Return the number of samples
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index]  # Return a single sample and its label

# Determine if a GPU is available and set the device, my personal machine always seems to use the CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset and preprocess it
dataset = pd.read_csv('Dataset.csv', encoding="ISO-8859-1")
dataset_labels = dataset.pop("Diagnosis")  # Separate labels from features
dataset.pop("DoctorInCharge")  # Remove unnecessary column

# Split the dataset into training and testing portions
test_portion_dataset = dataset[int((len(dataset) * (7/8))):]
test_portion_dataset_labels = dataset_labels[int((len(dataset) * (7/8))):]
train_portion_dataset = dataset[:int((len(dataset) * (7/8)))]
train_portion_dataset_labels = dataset_labels[:int((len(dataset) * (7/8)))]

# Convert data to numpy arrays
dataset_features = np.vstack(train_portion_dataset.values).astype(np.float32)
test_dataset_features = np.vstack(test_portion_dataset.values).astype(np.float32)

# Scale the features
scaler = StandardScaler()
train_portion_dataset = scaler.fit_transform(train_portion_dataset)  # Fit and transform training data
test_portion_dataset = scaler.transform(test_portion_dataset)  # Transform test data

# Define training parameters
batch_size = 100
n_iters = 1000
num_epochs = n_iters / (len(dataset_features) / batch_size)  # Calculate number of epochs
num_epochs = int(num_epochs)
print(num_epochs)

input_dim = dataset_features.shape[1]  # Number of input features
output_dim = 1  # Number of output classes (binary classification)
hidden_dim = 200  # Number of hidden units

# Create dataset and dataloaders
train_dataset = CSVDataset(train_portion_dataset, train_portion_dataset_labels)
test_dataset = CSVDataset(test_portion_dataset, test_portion_dataset_labels)
dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
model.to(device)  # Move model to GPU if available
print(sum([x.reshape(-1).shape[0] for x in model.parameters()]))  # Print the number of parameters in the model
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Split the training data into training and validation sets
val_split = 0.2
val_size = int(len(train_dataset) * val_split)
train_size = len(train_dataset) - val_size
print(val_size)
print(train_size)

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
print("Length of train_dataset:" + str(len(train_dataset)))
print("Length of val_dataset:" + str(len(val_dataset)))
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Lists to store accuracy and iteration information
accuracies = []
iterations = []
iter = 0

# Training loop
for epoch in range(num_epochs):
    for i, (fields, labels) in enumerate(dataset_loader):
        model.train()  # Set model to training mode
        fields, labels = fields.to(device), labels.to(device)  # Move data to GPU if available
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(fields)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        iter += 1
        if iter % 500 == 0:
            # Calculate accuracy on the validation set
            correct = 0
            total = 0
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                for fields, labels in val_loader:
                    fields, labels = fields.to(device), labels.to(device)  # Move data to GPU if available
                    testOutputs = model(fields).round()  # Get predictions and round to 0 or 1
                    predicted = testOutputs.round()  # Round predictions
                    total += labels.size(0)  # Update total number of samples
                    correct += (predicted == labels).sum().item()  # Count correct predictions
            
            accuracy = 100 * correct / total  # Calculate accuracy
            accuracies.append(accuracy)  # Append accuracy to the list
            iterations.append(iter)  # Append iteration to the list
            # Print current iteration, loss, and accuracy
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

# Print the list of accuracies
print(accuracies)
fig, ax = plt.subplots()
plt.figure(figsize=(12,6))

# Plot accuracy over iterations
ax.plot(iterations, accuracies, label="Validation set accuracy %")
ax.set(xlabel="Iteration", ylabel="Accuracy %",
       title="Accuracy over time")
ax.grid()
plt.legend()

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
predictedArray = []
with torch.no_grad():
    for fields, labels in test_loader:
        fields, labels = fields.to(device), labels.to(device)  # Move data to GPU if available
        outputs = model(fields)  # Forward pass
        predicted = outputs.round()  # Round predictions
        predictedArray.append(outputs)  # Collect outputs
        total += labels.size(0)  # Update total number of samples
        correct += (predicted == labels).sum().item()  # Count correct predictions

# Calculate test accuracy
test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy}%')

# Save test accuracy to a file
with open('accuracies.txt', 'a') as accuraciesFile:
    accuraciesFile.write(str(test_accuracy) + '\n')

# Read and print accuracy from the file
with open('accuracies.txt', 'r') as accuraciesFile:
    accuraciesList = [float(line.strip()) for line in accuraciesFile]
    print(accuraciesList)

# Plot test accuracy over time
ax = fig.add_subplot(1,2,1)
plt.plot(range(1, len(accuraciesList) + 1), accuraciesList, 'ro-', label="Test accuracy %")
ax.plot(iterations[-1], test_accuracy, 'ro-', label="Test accuracy %")
accuraciesList = []

ax.legend()
plt.tight_layout()
fig.savefig("test.png")  # Save the plot
plt.show()  # Display the plot
