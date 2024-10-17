import numpy as np

class InputNormalization:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, X):
        """Calculate the mean and standard deviation from the dataset."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        """Normalize the dataset using the mean and standard deviation."""
        if self.mean is None or self.std is None:
            raise ValueError("Fit the normalization parameters first using the 'fit' method.")
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        """Fit and transform the dataset."""
        self.fit(X)
        return self.transform(X)

# Example Usage:
X_train = np.array([[10, 200], [20, 400], [15, 300]])
normalizer = InputNormalization()
X_train_normalized = normalizer.fit_transform(X_train)
print("Normalized Data:\n", X_train_normalized)
