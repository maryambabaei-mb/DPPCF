import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class PrivyNetModel:
    def __init__(
        self,
        model_class,
        input_dim,
        preprocessor,
        privacy_engine=None,
        batch_size=64,
        epochs=15,
        lr=0.01,
        delta=1e-5,
        optimizer=optim.Adam,
        target_delta=1e-5, 
        target_epsilon=1.0,
        sigma=1.0,
        max_per_sample_grad_norm=1.0,
        train_dp=True, 
        use_make_private_epsilon=False
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.delta = delta
        self.preprocessor = preprocessor
        self.privacy_engine = privacy_engine
        self.train_dp = train_dp
        self.use_make_private_epsilon = use_make_private_epsilon
        self.target_delta = target_delta
        self.target_epsilon = target_epsilon
        self.sigma = sigma
        self.max_per_sample_grad_norm = max_per_sample_grad_norm

        self.model = model_class(
            input_dim
        ) 
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)

    def preprocess_data(self, X, y):
        X_transformed = self.preprocessor.transform(X)

        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        y_np = y.to_numpy()

        train_dataset = TensorDataset(
            torch.tensor(X_transformed, dtype=torch.float32),
            torch.tensor(y_np, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        return train_loader

    def fit(self, X_train, y_train):
        self.train_loader = self.preprocess_data(X_train, y_train)

        if self.privacy_engine and self.train_dp and not self.use_make_private_epsilon:

            self.privacy_engine = self.privacy_engine
            (
                self.model,
                self.optimizer,
                self.train_loader,
            ) = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.sigma,
                max_grad_norm=self.max_per_sample_grad_norm,
                poisson_sampling=False,
            )
        elif self.privacy_engine and self.train_dp and self.use_make_private_epsilon:
            self.privacy_engine = self.privacy_engine
            (
                self.model,
                self.optimizer,
                self.train_loader,
            ) = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                max_grad_norm=self.max_per_sample_grad_norm,
                poisson_sampling=False,
                target_delta=self.target_delta,
                target_epsilon=self.target_epsilon,
                epochs=self.epochs,
            )

        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for data, target in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data).squeeze(1)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()

            if self.privacy_engine and self.train_dp:
                epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
                print(f"Train Epoch: {epoch} \t (ε = {epsilon:.2f}, δ = {self.delta})")
            else:
                print(f"Train Epoch: {epoch}")

    def predict(self, X):
        # Ensure X is a DataFrame or compatible format
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Apply the preprocessing steps
        X_transformed = self.preprocessor.transform(X)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X_transformed, dtype=torch.float32)

        # Get predictions from the model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs >= 0.5).float().squeeze()

        return predictions

    def score(self, X, y):
        self.test_loader = self.preprocess_data(X, y)
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data).squeeze(1)
                test_loss += criterion(output, target).item()
                pred = output >= 0.5
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100.0 * correct / len(self.test_loader.dataset)
        return accuracy