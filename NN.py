import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import Subset
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, feature_choice):
        current_dir = os.getcwd()

        data_paths = {
            "energies": os.path.join("data", "energies.npy"),
            "2-body": os.path.join("data", "features_2b.npy"),
            "3-body": os.path.join("data", "features_3b.npy"),
            "4-body": os.path.join("data", "features_4b.npy"),
            }
        if feature_choice == "2-body":
            features = np.load(data_paths["2-body"])
        elif feature_choice == "3-body":
            features = np.load(data_paths["3-body"])
        elif feature_choice == "4-body":
            features = np.load(data_paths["4-body"])
        elif feature_choice == "2+3-body":
            features_2 = np.load(data_paths["2-body"])
            features_3 = np.load(data_paths["3-body"])
            features = np.hstack((features_2, features_3))
        elif feature_choice == "4+3-body":
            features_4 = np.load(data_paths["4-body"])
            features_3 = np.load(data_paths["3-body"])
            features = np.hstack((features_4, features_3))
        elif feature_choice == "2+3+4-body":
            features_2 = np.load(data_paths["2-body"])
            features_3 = np.load(data_paths["3-body"])
            features_4 = np.load(data_paths["4-body"])
            features = np.hstack((features_2, features_3, features_4))
        else:
            raise ValueError("Invalid feature choice!")

        features = (features - features.mean())/features.std()
        self.features = features

        energies = np.load(data_paths["energies"])
        energies = (energies - energies.mean())/energies.std()
        self.targets = energies
        
        # Make sure both have the same length
        assert len(self.features) == len(self.targets), "Features and targets must have the same length"

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.features)
    
    def __getitem__(self, index):
        # Retrieve the features and target for the given index
        feature_row = self.features[index]  # Convert to numpy array
        target_value = self.targets[index]  # Convert to numpy array

        # Convert the features to a tensor
        features = torch.tensor(feature_row, dtype=torch.float32)
        
        # Convert the target to a tensor
        target = torch.tensor(target_value, dtype=torch.float32)
        
        return features, target

# Paths to your CSV files
features_file = 'data/features_2b.npy'
targets_file = 'data/energies.npy'

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        # Define layers
        self.hidden = nn.Linear(input_size, hidden_size)  # Hidden layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.output = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        # Pass input through the hidden layer, followed by ReLU
        x = self.relu(self.hidden(x))
        
        # Pass the result to the output layer
        x = self.output(x)
        
        return x


def train(model, dataloader, loss_fn, optimizer, num_epochs = 20):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        
        total_loss = 0
        
        for features, targets in dataloader:
            # Forward pass
            outputs = model(features)  # Forward pass through the network
            
            # Compute the loss
            loss = loss_fn(outputs.squeeze(), targets)  # Squeeze to match dimensions if needed
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            
def evaluate(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    
    predictions = []
    targets = []
    
    with torch.no_grad():  # Disable gradient computation
        for features, target in dataloader:
            output = model(features)  # Forward pass to get predictions
            predictions.append(output)
            targets.append(target)
    
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    loss_fn = nn.MSELoss()
    loss = loss_fn(predictions.squeeze(), targets).item()
    error = np.sqrt(loss) / targets.std().item()
    return float(error)
    
def SimpleNN_1_layer_graph(dataset, train_sizes = np.logspace(np.log10(0.01), np.log10(1), 8)):
    all_train_errors = []
    all_test_errors = []
    temp = 1
    for size in train_sizes:

        # Define the subsample size
        
        # Randomly generate unique indices for the subsample
        random_indices = np.random.choice(dataset.features.shape[0], int(size*dataset.features.shape[0]), replace=False)
        
        # Create a subset using the random indices
        subsample = Subset(dataset, random_indices)

        # Split the dataset        
        train_size = int(size* len(subsample) * 0.8)
        test_size = len(subsample) - train_size
        train_dataset, test_dataset = random_split(subsample, [train_size, test_size])
        
        # Create DataLoaders
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Get the number of features in your dataset
        input_size = dataset.features.shape[1]  # Number of features in your dataset
        hidden_size = input_size  # Size of the hidden layer (you can change this)
        output_size = 1  # For regression (single output), set this to 1
        
        # Create the model instance
        model = SimpleNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        
        # Define the loss function and optimizer
        loss_fn = nn.MSELoss()  # Mean Squared Error Loss (for regression)
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

        #train model
        train(model, train_dataloader, loss_fn, optimizer, num_epochs=20)

        #train and test errors
        all_train_errors.append(evaluate(model, train_dataloader))
        all_test_errors.append(evaluate(model, test_dataloader))

        print(f"{temp}/8")
        temp = temp + 1


    all_train_errors = np.array(all_train_errors)
    all_test_errors = np.array(all_test_errors)
    np.save(f"train_errors_NN_1_layer.npy", all_train_errors.T)
    np.save(f"test_errors_NN_1_layer.npy", all_test_errors.T)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes * dataset.features.shape[0], all_train_errors, marker = 'o', label="Training Error", color="r")
    plt.plot(train_sizes * dataset.features.shape[0], all_test_errors, marker = 'o', label="Testing Error", color="g")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Subset Size (Number of Training Samples)")
    plt.ylabel("Root Mean Squared Error / std")
    plt.title("Learning Curve for neural network with 1 hidden layer")
    plt.legend()
    plt.grid()
    plt.savefig("NN_1_layer.png")
    plt.show()

class SimpleNN_multiple_layers(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden_layers, output_size):
        super(SimpleNN_multiple_layers, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Dynamically create hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(n_hidden_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass input through the input layer and activation
        x = self.relu(self.input_layer(x))
        
        # Pass through each hidden layer with activation
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        
        # Pass through the output layer
        x = self.output_layer(x)
        
        return x

def SimpleNN_multiple_layers_graph(dataset, hidden_layers= (np.logspace(0, 1, 4)).astype("int")):
    all_train_errors = []
    all_test_errors = []
    temp = 1
    print(hidden_layers)
    for hidden_layer in hidden_layers:
        # Define the subsample size
        # Randomly generate unique indices for the subsample
        random_indices = np.random.choice(dataset.features.shape[0], int(dataset.features.shape[0]), replace=False)

        # Split the dataset        
        train_size = int(dataset.features.shape[0] * 0.8)
        test_size = dataset.features.shape[0] - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        # Create DataLoaders
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Get the number of features in your dataset
        input_size = dataset.features.shape[1]  # Number of features in your dataset
        hidden_size = 100  # Size of the hidden layer (you can change this)
        output_size = 1  # For regression (single output), set this to 1
        
        # Create the model instance
        model = SimpleNN_multiple_layers(input_size=input_size, hidden_size=hidden_size, n_hidden_layers = hidden_layer, output_size=output_size)
        
        # Define the loss function and optimizer
        loss_fn = nn.MSELoss()  # Mean Squared Error Loss (for regression)
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

        #train model
        train(model, train_dataloader, loss_fn, optimizer, num_epochs=40 + (20*hidden_layer))

        #train and test errors
        all_train_errors.append(evaluate(model, train_dataloader))
        all_test_errors.append(evaluate(model, test_dataloader))

        print(f"{temp}/4")
        temp = temp + 1


    all_train_errors = np.array(all_train_errors)
    all_test_errors = np.array(all_test_errors)
    np.save(f"train_errors_NN_multiple_layers.npy", all_train_errors.T)
    np.save(f"test_errors_NN_multiple_layers.npy", all_test_errors.T)

    plt.figure(figsize=(10, 6))
    plt.plot(hidden_layers, all_train_errors, marker = 'o', label="Training Error", color="r")
    plt.plot(hidden_layers, all_test_errors, marker = 'o', label="Testing Error", color="g")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of hidden layers")
    plt.ylabel("Root Mean Squared Error / std")
    plt.title("Learning Curve for neural network with varying number of hidden layers")
    plt.legend()
    plt.grid()
    plt.savefig("NN_multiple_hidden.png")
    plt.show()

def train_stopping_criteria(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs=20, patience=10):
    best_val_loss = float('inf')  # Initialize with a large number
    epochs_without_improvement = 0  # Counter for early stopping
    best_epoch = 0  # To track which epoch had the best performance
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        
        total_loss = 0
        
        # Training loop
        for features, targets in train_dataloader:
            # Forward pass
            outputs = model(features)  # Forward pass through the network
            
            # Compute the loss
            loss = loss_fn(outputs.squeeze(), targets)  # Squeeze to match dimensions if needed
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()  # Accumulate loss for this batch
        
        # Compute average training loss
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Evaluate on the validation set
        val_loss = evaluate(model, val_dataloader)  # Assume evaluate function calculates validation loss
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.5f}, Val Loss: {val_loss:.5f}")
        
        # Early stopping condition: if validation loss doesn't improve
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            # Save the best model's state_dict (weights)
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1
        
        # If no improvement for 'patience' epochs, stop training
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}, best epoch was {best_epoch+1}")
            break

    # Load the best model (with the lowest validation loss)
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Return the best epoch and the corresponding validation loss
    return best_epoch, best_val_loss

def SimpleNN_multiple_layers_stopping_criteria_graph(dataset, hidden_layers=(np.logspace(0, 1, 4)).astype("int"), patience=10):
    all_train_errors = []
    all_test_errors = []
    all_epochs = []
    temp = 1
    print(hidden_layers)

    for hidden_layer in hidden_layers:
        # Define the subsample size
        random_indices = np.random.choice(dataset.features.shape[0], int(dataset.features.shape[0]), replace=False)

        # Split the dataset        
        train_size = int(dataset.features.shape[0] * 0.8)
        test_size = dataset.features.shape[0] - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        # Split further for validation (e.g., 80% training, 20% validation)
        train_size = int(len(train_dataset) * 0.8)
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Create DataLoaders
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        input_size = dataset.features.shape[1]  # Number of features in your dataset
        hidden_size = 100  # Size of the hidden layer
        output_size = 1  # For regression (single output), set this to 1
        
        # Create the model instance
        model = SimpleNN_multiple_layers(input_size=input_size, hidden_size=hidden_size, n_hidden_layers=hidden_layer, output_size=output_size)
        
        # Define the loss function and optimizer
        loss_fn = nn.MSELoss()  # Mean Squared Error Loss (for regression)
        optimizer = optim.Adam(model.parameters(), lr=0.00001)  # Adam optimizer

        # Train model with early stopping
        best_epoch, _ = train_stopping_criteria(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs=40 + (20 * hidden_layer), patience=patience)
        # Train and test errors
        all_train_errors.append(evaluate(model, train_dataloader))
        all_test_errors.append(evaluate(model, test_dataloader))
        all_epochs.append(best_epoch)

        print(f"{temp}/4")
        temp = temp + 1

    all_train_errors = np.array(all_train_errors)
    all_test_errors = np.array(all_test_errors)
    np.save(f"train_errors_NN_multiple_hidden_stop_crit.npy", all_train_errors.T)
    np.save(f"test_errors_NN_multiple_hidden_stop_crit.npy", all_test_errors.T)

    # Plotting the errors
    plt.figure(figsize=(10, 6))
    plt.plot(hidden_layers, all_train_errors, marker='o', label="Training Error", color="r")
    plt.plot(hidden_layers, all_test_errors, marker='o', label="Testing Error", color="g")

    # Annotate each point with the corresponding number of epochs
    for i, hidden_layer in enumerate(hidden_layers):
        plt.annotate(f'epochs: {all_epochs[i]}', 
                     (hidden_layers[i], all_train_errors[i]), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center', fontsize=9)
        
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of hidden layers")
    plt.ylabel("Root Mean Squared Error / std")
    plt.title("Learning Curve for neural network with varying number of hidden layers")
    plt.legend()
    plt.grid()
    plt.savefig("NN_multiple_hidden_stop_crit.png")
    plt.show()

def SimpleNN_one_layer_multiple_nodes_graph(dataset, hidden_sizes = np.array([10,50,100, 200, 500]), train_sizes=np.logspace(np.log10(0.01), np.log10(1), 8)):
    all_train_errors = []
    all_test_errors = []

    # Set a color palette (choose as many colors as there are hidden_sizes)
    colors = plt.cm.viridis(np.linspace(0, 1, len(hidden_sizes)))
    
    for hidden_size in hidden_sizes:
        train_errors_for_hidden_size = []
        test_errors_for_hidden_size = []

        temp = 1
        for size in train_sizes:
            # Define the subsample size
            random_indices = np.random.choice(dataset.features.shape[0], int(size * dataset.features.shape[0]), replace=False)
            
            # Create a subset using the random indices
            subsample = Subset(dataset, random_indices)
            
            # Split the dataset        
            train_size = int(size * len(subsample) * 0.8)
            test_size = len(subsample) - train_size
            train_dataset, test_dataset = random_split(subsample, [train_size, test_size])
            
            # Create DataLoaders
            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Get the number of features in your dataset
            input_size = dataset.features.shape[1]  # Number of features in your dataset
            output_size = 1  # For regression (single output), set this to 1
            
            # Create the model instance with the current hidden layer size
            model = SimpleNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
            
            # Define the loss function and optimizer
            loss_fn = nn.MSELoss()  # Mean Squared Error Loss (for regression)
            optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
            
            # Train the model
            #train(model, train_dataloader, loss_fn, optimizer, num_epochs=20)

            # Train model with early stopping
            print(40 + (20 * hidden_size))
            patience = 10
            best_epoch, _ = train_stopping_criteria(model, train_dataloader, test_dataloader, loss_fn, optimizer, num_epochs= (40 + (20 * hidden_size)), patience=patience)
            
            # Calculate and store the training and test errors
            train_errors_for_hidden_size.append(evaluate(model, train_dataloader))
            test_errors_for_hidden_size.append(evaluate(model, test_dataloader))
            
            print(f"Hidden Size: {hidden_size}, Training on {temp}/{len(train_sizes)} subsets")
            temp += 1
        
        # Store the errors for each hidden size
        all_train_errors.append(train_errors_for_hidden_size)
        all_test_errors.append(test_errors_for_hidden_size)
    
    # Convert errors to numpy arrays for plotting
    all_train_errors = np.array(all_train_errors)
    all_test_errors = np.array(all_test_errors)
    
    # Save the errors
    np.save(f"train_errors_NN_1_layer_multiple_nodes.npy", all_train_errors.T)
    np.save(f"test_errors_NN_1_layer_multiple_nodes.npy", all_test_errors.T)
    
    # Plot the learning curves
    plt.figure(figsize=(10, 6))
    
    for i, hidden_size in enumerate(hidden_sizes):
        plt.plot(train_sizes * dataset.features.shape[0], all_train_errors[i], marker='o', label=f"Train Error (Hidden Size={hidden_size})", color=colors[i], linestyle='-')
        plt.plot(train_sizes * dataset.features.shape[0], all_test_errors[i], marker='o', label=f"Test Error (Hidden Size={hidden_size})", color=colors[i], linestyle='--')
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Subset Size (Number of Training Samples)")
    plt.ylabel("Root Mean Squared Error / std")
    plt.title("Learning Curve for neural network with varying hidden layer sizes")
    plt.legend()
    plt.grid()
    plt.savefig("NN_1_layer_multiple_nodes.png")
    plt.show()