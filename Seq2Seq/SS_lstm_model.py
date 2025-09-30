# lstm_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self, input_size, hidden_size=128, num_layers=2, lr=1e-3,
                 batch_size=32, num_epochs=50, loss_fn="mse", dropout=0.2, device=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout = dropout

        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError("loss_fn must be 'mse' or 'l1'")

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def prepare_data(self, X, y, test_split=0.2): 
        """
        X: numpy array (n_samples, seq_len, n_features)
        y: numpy array (n_samples,) or (n_samples, seq_len)
        Splits -> fits scalers on TRAIN only -> creates DataLoaders
        """
        n_samples, seq_len, n_feat = X.shape
        split_idx = int(n_samples * (1 - test_split))

        X_train_raw = X[:split_idx]
        X_test_raw  = X[split_idx:]
        y_train_raw = y[:split_idx]
        y_test_raw  = y[split_idx:]

        # Fit scaler on TRAIN only (X)
        X_train_flat = X_train_raw.reshape(-1, n_feat)    # (n_train*seq_len, n_feat)
        self.scaler_X.fit(X_train_flat)
        X_train_scaled = self.scaler_X.transform(X_train_flat).reshape(X_train_raw.shape)
        X_test_scaled  = self.scaler_X.transform(X_test_raw.reshape(-1, n_feat)).reshape(X_test_raw.shape)

        # --- Handle y depending on shape ---
        if y_train_raw.ndim == 1:  
            # Case: (n_samples,) -> single value per sequence
            y_train_flat = y_train_raw.reshape(-1, 1)
            y_test_flat  = y_test_raw.reshape(-1, 1)

            self.scaler_y.fit(y_train_flat)
            y_train_scaled = self.scaler_y.transform(y_train_flat).flatten()
            y_test_scaled  = self.scaler_y.transform(y_test_flat).flatten()

        else:
            # Case: (n_samples, seq_len) -> sequence-to-sequence targets
            y_train_flat = y_train_raw.reshape(-1, 1)
            y_test_flat  = y_test_raw.reshape(-1, 1)

            self.scaler_y.fit(y_train_flat)
            y_train_scaled = self.scaler_y.transform(y_train_flat).reshape(y_train_raw.shape)
            y_test_scaled  = self.scaler_y.transform(y_test_flat).reshape(y_test_raw.shape)

        # Create datasets
        train_ds = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                torch.tensor(y_train_scaled, dtype=torch.float32))
        test_ds = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32),
                                torch.tensor(y_test_scaled, dtype=torch.float32))

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

    def build_model(self):
        class LSTMSeq2Seq(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=dropout)
                self.fc = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, 1)
                )

            def forward(self, x):
                # x: (batch, seq_len, input_size)
                out, _ = self.lstm(x)                  # (batch, seq_len, hidden)
                out = self.fc(out)                     # (batch, seq_len, 1)
                out = out.squeeze(-1)                  # (batch, seq_len)
                return out

        self.model = LSTMSeq2Seq(self.input_size, self.hidden_size, self.num_layers, self.dropout)
        self.model.to(self.device)


    def train(self):
        from torch.utils.tensorboard import SummaryWriter

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.model.train()

        # TensorBoard writer (logs saved in "runs/experiment1")
        writer = SummaryWriter(log_dir="runs/experiment1")

        global_step = 0
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.model(X_batch)           # (batch, seq_len)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                # Log training loss for each batch
                writer.add_scalar("Loss/train", loss.item(), global_step)
                global_step += 1

            avg_loss = epoch_loss / n_batches
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

            # Log average loss per epoch
            writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)

        writer.close()

    def evaluate(self):
        self.model.eval()
        preds_list = []
        true_list = []
        test_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds = self.model(X_batch)        # (batch, seq_len)
                loss = self.criterion(preds, y_batch)
                test_loss += loss.item()
                n_batches += 1

                preds_list.append(preds.cpu().numpy())
                true_list.append(y_batch.cpu().numpy())

        print(f"Test Loss (scaled): {test_loss / max(n_batches,1):.4f}")

        preds_arr = np.vstack(preds_list)       # (n_test_samples, seq_len)
        true_arr  = np.vstack(true_list)

        # inverse-transform flattened values
        preds_flat = preds_arr.reshape(-1, 1)
        true_flat  = true_arr.reshape(-1, 1)
        preds_inv = self.scaler_y.inverse_transform(preds_flat).reshape(preds_arr.shape)
        true_inv  = self.scaler_y.inverse_transform(true_flat).reshape(true_arr.shape)

        # return flattened 1D arrays (so sklearn metrics work)
        return preds_inv.flatten(), true_inv.flatten()
