# %% [markdown]
# ### Importing the dependencies

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# %% [markdown]
# #### Data Collection and preprocessing

# %%
data = load_breast_cancer()
X = data.data
y = data.target

# %%
print(X)

# %%
print(y)

# %% [markdown]
# ##### Split the dataset into training set and test set

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# %%
print(X.shape)
print(X_train.shape)
print(X_test.shape)

# %% [markdown]
# ##### Standardize the data using standard scaler

# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
type(X_train)

# %%
#converting data into pytorch tensors and move it to GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# %% [markdown]
# ### Neural Network Architecture

# %%
#define the nn architecture
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        print("Shape of out after fc1: " + str(out.shape))
        out = self.relu(out)
        print("Shape of out after relu1: " + str(out.shape))
        out = self.fc2(out)
        print("Shape of out after fc2: " + str(out.shape))
        out = self.sigmoid(out)
        print("Shape of out after sigmoid: " + str(out.shape))
        return out

# %%
# define hyperparameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 100

# %%
#initialize the neural network
model = NeuralNet(input_size,hidden_size,output_size).to(device)

# %%
#define the loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %% [markdown]
# #### Training the neural network

# %%
#training the model 
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train.view(-1,1))
    loss.backward()
    optimizer.step()


    # calculate the accuracy 
    with torch.no_grad():
        predicted = outputs.round()
        correct = (predicted == y_train.view(-1,1)).float().sum()
        accuracy = correct/y_train.size(0)

    if (epoch+1)%10 == 0:
        print(f"Epoch : [{epoch+1}/{num_epochs}], Loss : {loss.item():.4f}, Accuracy : {accuracy.item()*100:.2f}%")

# %% [markdown]
# ### Model evaluation

# %%
#evaluation on training set
model.eval()
with torch.no_grad():
    outputs = model(X_train)
    predicted = outputs.round()
    correct = (predicted == y_train.view(-1,1)).float().sum()
    accuracy = correct/y_train.size(0)
    print(f"Accuracy on training data: {accuracy.item()*100:.2f}%")

# %%
#evaluation on test set
model.eval()
print("In evaluation stage")
with torch.no_grad():
    outputs = model(X_test)
    predicted = outputs.round()
    correct = (predicted == y_test.view(-1,1)).float().sum()
    accuracy = correct/y_test.size(0)
    print(f"Accuracy on training data: {accuracy.item()*100:.2f}%")


