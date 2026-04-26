import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class DenseLayerNumpy:
    """
    Del 1B: Dense/ANN-lager i NumPy (forward-pass).

    - inputs: 1D-vektor x med shape (n_inputs,)
    - weights: matris W med shape (n_neurons, n_inputs)
    - bias: vektor b med shape (n_neurons,)
    - output: 1D-vektor y med shape (n_neurons,)
    """

    def __init__(self, W: np.ndarray, b: np.ndarray, activation=None):
        self.W = np.asarray(W, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.activation = activation

        if self.W.ndim != 2:
            raise ValueError(f"W must be 2D (n_neurons, n_inputs), got shape={self.W.shape}")
        if self.b.ndim != 1:
            raise ValueError(f"b must be 1D (n_neurons,), got shape={self.b.shape}")
        if self.b.shape[0] != self.W.shape[0]:
            raise ValueError(f"b length must match W rows: b={self.b.shape[0]} W_rows={self.W.shape[0]}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError(f"x must be 1D (n_inputs,), got shape={x.shape}")
        if x.shape[0] != self.W.shape[1]:
            raise ValueError(f"x length must match W cols: x={x.shape[0]} W_cols={self.W.shape[1]}")

        z = self.W @ x + self.b
        if self.activation is None:
            return z
        return self.activation(z)


if __name__ == "__main__":
    x = np.array([2.0, -1.0, 0.5, 3.0])  # (n_inputs=4,)
    W = np.array(
        [
            [0.2, -0.1, 0.4, 1.0],
            [-1.2, 0.3, 0.0, -0.7],
            [0.8, 0.8, 0.8, 0.8],
        ]
    )  # (n_neurons=3, n_inputs=4)
    b = np.array([0.1, -0.2, 0.0])  # (n_neurons=3,)

    layer = DenseLayerNumpy(W, b, activation=sigmoid)
    y = layer.forward(x)

    print("x shape:", x.shape)
    print("W shape:", W.shape)
    print("b shape:", b.shape)
    print("y shape:", y.shape)
    print("y:", y)

