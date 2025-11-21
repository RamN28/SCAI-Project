import os
import random
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader



# Load the local CSV file (update path if different)
file_path = r'/content/diamonds.csv' # REPLACE WITH OUR DATASET
assert os.path.exists(file_path), f'File not found: {file_path}'
df = pd.read_csv(file_path)
print('Dataset loaded successfully. Showing basic info:')
print(df.shape)
print(df.head())

# Inspect for zeros in 'x', 'y', 'z' and NaNs in general
print("\nNumber of rows with 'x' = 0: ", (df['x'] == 0).sum())
print("Number of rows with 'y' = 0: ", (df['y'] == 0).sum())
print("Number of rows with 'z' = 0: ", (df['z'] == 0).sum())
print("\nNumber of NaN values per column:\n", df.isnull().sum())

# Replace 0 values in x, y, z with NaN to handle them consistently with other NaNs
df[['x', 'y', 'z']] = df[['x', 'y', 'z']].replace(0, np.nan)

# Drop rows with any NaN values that resulted from the 0 replacement or original NaNs
original_rows = len(df)
df.dropna(inplace=True)
dropped_rows = original_rows - len(df)
print(f'\nDropped {dropped_rows} rows containing NaN or zero dimensions (x, y, z).')

# Optional: quick feature engineering — drop 'index' if present or irrelevant columns
to_drop = [c for c in ['id','index'] if c in df.columns]
if len(to_drop):
    df.drop(columns=to_drop, inplace=True)

df.head()



# Work on a copy
df_work = df.copy()

# Categorical columns to one-hot encode
categorical_cols = [c for c in ['cut','color','clarity'] if c in df_work.columns]
if len(categorical_cols):
    df_work = pd.get_dummies(df_work, columns=categorical_cols, drop_first=True)


# Optionally transform the target to reduce skew (log1p) — helps training
y = np.log1p(df_work['price'].values).astype(np.float32)
X = df_work.drop(columns=['price']).values.astype(np.float32)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

class DiamondDataset(Dataset):
    def __init__(self, features, target):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

train_dataset = DiamondDataset(X_train, y_train)
test_dataset = DiamondDataset(X_test, y_test)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Testing dataset size: {len(test_dataset)}")



# What is the number of input features?
# We need to consider all columns except 'price' and 'carat' if we decide to use carat as a feature
# For now, let's assume 'carat' is also part of features, meaning df_encoded.drop('price', axis=1) has X features.
# Let's get the actual input size from the X array created previously.
input_size = X.shape[1]

# What is the number of output classes (what are you predicting)?
output_size = 1 # We are predicting a single continuous value (price)

# How complex is the function you want to learn?
hidden_size = 16 # Increased hidden size for potentially better learning capacity

class DiamondCostPredictor(nn.Module):
    def __init__(self):
        super(DiamondCostPredictor, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.a1 = nn.ReLU() # Using ReLU as a common activation function for hidden layers
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.a2 = nn.ReLU() # Using ReLU again for the second hidden layer
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x1 = self.input_layer(x)
        x_a1 = self.a1(x1)
        x2 = self.fc2(x_a1)
        x_a2 = self.a2(x2)
        classification = self.classifier(x_a2)
        return classification
    


#REPLACE DIAMOND WITH OUR DATA STUFFS 

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Instantiate the model
model = DiamondCostPredictor().to(device)

# Define the loss function (MSE for regression)
loss_fn = nn.MSELoss()

# Use Adam for faster convergence
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print('Model, loss, optimizer, and DataLoaders initialized.')




#Training Loop

num_epochs = 15
print('Starting training...')
for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    for batch_features, batch_targets in train_loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_features.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch [{epoch}/{num_epochs}] — Train Loss: {epoch_loss:.6f}')

print('Training complete.')


def evaluate_model(model, data_loader, loss_fn, device):
    # Set the model to evaluation mode
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    # Safe log-price range for clamping (adjust if needed)
    MIN_LOG_PRICE = 5.0
    MAX_LOG_PRICE = 11.0

    # Disable gradient calculations during evaluation
    with torch.no_grad():
        for batch_features, batch_targets in data_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass
            outputs = model(batch_features)

            # Calculate loss (on raw log-transformed values)
            loss = loss_fn(outputs, batch_targets)
            total_loss += loss.item() * batch_features.size(0)

            # --- CRITICAL FIX: Explicitly Clamp Outputs on the CPU ---
            # Move output to CPU and apply the clamp
            clamped_predictions = torch.clamp(outputs.cpu(), min=MIN_LOG_PRICE, max=MAX_LOG_PRICE)

            # Store clamped predictions and targets
            all_predictions.append(clamped_predictions.numpy().flatten())
            all_targets.append(batch_targets.cpu().numpy().flatten())

    # Calculate average test loss (MSE on log-transformed prices)
    avg_test_loss = total_loss / len(data_loader.dataset)

    # Concatenate all stored arrays
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # Inverse transform clamped predictions and targets
    original_predictions = np.expm1(predictions)
    original_targets = np.expm1(targets)

    # Calculate Root Mean Squared Error (RMSE) on the original price scale
    rmse = np.sqrt(np.mean((original_predictions - original_targets)**2))

    # Calculate Mean Absolute Error (MAE) on the original price scale
    mae = np.mean(np.abs(original_predictions - original_targets))

    return avg_test_loss, rmse, mae

# --- Run Evaluation ---
print('\nStarting testing/evaluation...')
test_loss, test_rmse, test_mae = evaluate_model(model, test_loader, loss_fn, device)

print(f'\n--- Final Test Metrics ---')
print(f'Test Loss (MSE on log-price): {test_loss:.6f}')  #REMEMBER TO CHANGE PRICE TO FRESHNESS/OUTPUT DATA
print(f'Test RMSE (Original Price Scale): ${test_rmse:.2f}')  #REMEMBER TO CHANGE PRICE TO FRESHNESS/OUTPUT DATA
print(f'Test MAE (Original Price Scale): ${test_mae:.2f}')  #REMEMBER TO CHANGE PRICE TO FRESHNESS/OUTPUT DATA
