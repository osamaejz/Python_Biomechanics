import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self, input_size, hidden_size=64, num_layers=1, lr=1e-3,
                 batch_size=32, num_epochs=30, loss_fn="mse"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Select loss function
        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "l1":
            self.criterion = nn.L1Loss()
        elif loss_fn == "smoothl1":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError("loss_fn must be one of: mse, l1, smoothl1")

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def prepare_data(self, X, y, test_split=0.2):
        """
        X: numpy array (n_samples, seq_len, n_features)
        y: numpy array (n_samples,)
        """

        # Normalize inputs (per feature across all timesteps)
        n_samples, seq_len, n_feat = X.shape
        X_reshaped = X.reshape(-1, n_feat)  # (n_samples*seq_len, n_features)
        X_scaled = self.scaler_X.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_feat)

        # Normalize outputs
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Train/test split
        split_idx = int(n_samples * (1 - test_split))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

        # PyTorch datasets
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.float32))
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.float32))

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

    def build_model(self):
        # class LSTMRegressor(nn.Module):
        #     def __init__(self, input_size, hidden_size, num_layers):
        #         super().__init__()
        #         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #         self.fc = nn.Linear(hidden_size, 1)

        #     def forward(self, x):
        #         out, _ = self.lstm(x)
        #         out = out[:, -1, :]   # take last timestep
        #         out = self.fc(out)
        #         return out.squeeze()

        # self.model = LSTMRegressor(self.input_size, self.hidden_size, self.num_layers)

        class LSTMRegressor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=dropout)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1)
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]  # last timestep
                out = self.fc(out)
                return out.squeeze()
            
        self.model = LSTMRegressor(self.input_size, self.hidden_size, self.num_layers)

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for X_batch, y_batch in self.train_loader:
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss/len(self.train_loader):.4f}")

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        preds_list, true_list = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                test_loss += loss.item()
                preds_list.extend(preds.numpy())
                true_list.extend(y_batch.numpy())

        # Inverse-transform
        preds_list = self.scaler_y.inverse_transform(np.array(preds_list).reshape(-1, 1)).flatten()
        true_list = self.scaler_y.inverse_transform(np.array(true_list).reshape(-1, 1)).flatten()

        print(f"Test Loss (scaled): {test_loss/len(self.test_loader):.4f}")
        return np.array(preds_list), np.array(true_list)
