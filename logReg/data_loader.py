import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_split_data(path: str = "train_test_2025.csv",
                        test_size: float = 0.2,
                        random_state: int = 1142):
    """Load CSV where last column is the target and split into train/test.

    Returns: X_train, X_test, y_train, y_test (numpy arrays)
    """
    scaler = StandardScaler()
    encoder = OneHotEncoder()

    df = pd.read_csv(path, skipinitialspace=True)

    # features: all columns except last
    X = df.iloc[:, :-1]
    # target: last column, map yes/no to 1/0 when present
    y = df.iloc[:, -1]
    if y.dtype == object:
        y = y.map({"yes": 1, "no": 0}).fillna(y)

    num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    # Process numeric columns
    X_num = X[num_cols].values.astype(float)
    X_num = scaler.fit_transform(X_num)

    # Process categorical columns
    X_cat = X[cat_cols].values.astype(str)
    X_cat = encoder.fit_transform(X_cat)

    # Combine processed features
    X = pd.DataFrame(np.hstack([X_num, X_cat]))

    return train_test_split(
            X.values, y.values, test_size=test_size, random_state=random_state, stratify=y.values
        )

class StandardScaler:
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0  # avoid division by zero

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class OneHotEncoder:
    def fit(self, X):
        """
        X: 2D array of categorical values (strings or ints)
        """
        self.categories = [
            np.unique(X[:, i]) for i in range(X.shape[1])
        ]

    def transform(self, X):
        encoded_cols = []

        for i, cats in enumerate(self.categories):
            col = X[:, i]
            one_hot = np.zeros((X.shape[0], len(cats)))

            for j, cat in enumerate(cats):
                one_hot[:, j] = (col == cat).astype(int)

            encoded_cols.append(one_hot)

        return np.concatenate(encoded_cols, axis=1)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)