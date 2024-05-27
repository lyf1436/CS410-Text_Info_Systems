if __name__ == '__main__':
    from base import BaseModel
    from utils import split_sequences
else:
    from .base import BaseModel
    from .utils import split_sequences
from numpy import array, hstack
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import numpy as np


class LSTMModel(BaseModel):
    def __init__(self, n_steps_in=3, n_steps_out=2, n_features=3, activation = 'relu', epochs=300, verbose=0):
        self.activation = activation
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.n_features = n_features
        self.epochs = epochs
        self.verbose = verbose
        model = Sequential()
        model.add(LSTM(200, activation=activation, input_shape=(n_steps_in, n_features)))
        model.add(RepeatVector(n_steps_out))
        model.add(LSTM(200, activation=activation, return_sequences=True))
        model.add(TimeDistributed(Dense(n_features)))
        model.compile(optimizer='adam', loss='mse')
        self.model = model
    def fit(self, X, y):
        X = X.reshape((-1, self.n_steps_in, self.n_features))
        y = y.reshape((-1, self.n_steps_out, self.n_features))
        self.model.fit(X, y, epochs=self.epochs, verbose=self.verbose)
        return self
    def predict(self, X):
        X = X.reshape((-1, self.n_steps_in, self.n_features))
        return self.model.predict(X, verbose=self.verbose)
    def score(self, X, y):
        y = y.reshape((-1, self.n_steps_out, self.n_features))
        y_hat = self.predict(X)
        mse = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            mse[i] = np.sqrt(np.mean((y[i] - y_hat[i])**2))
        return mse

if __name__ == '__main__':
    # Define input sequence
    in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

    # Convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))

    # Horizontally stack columns
    dataset = hstack((in_seq1, in_seq2, out_seq))

    # Choose a number of time steps
    n_steps_in, n_steps_out = 3, 2

    # Convert into input/output
    X, y = split_sequences(dataset, n_steps_in, n_steps_out)

    # The dataset knows the number of features, e.g. 2
    n_features = X.shape[2]

    # Define model
    # model = Sequential()
    # model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
    # model.add(RepeatVector(n_steps_out))
    # model.add(LSTM(200, activation='relu', return_sequences=True))
    # model.add(TimeDistributed(Dense(n_features)))
    # model.compile(optimizer='adam', loss='mse')
    model = LSTMModel(n_steps_in=n_steps_in, n_steps_out=n_steps_out, n_features=n_features)

    # Fit model
    model.fit(X, y)

    # Demonstrate prediction
    x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input)
    print(yhat)

