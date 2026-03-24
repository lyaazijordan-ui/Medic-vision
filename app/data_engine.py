import numpy as np
from sklearn.linear_model import LinearRegression

def predict_future(df, column, steps=5):
    y = df[column].values
    X = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(y), len(y) + steps).reshape(-1, 1)
    predictions = model.predict(future_X)

    return predictions


def detect_anomalies(df, column):
    data = df[column]
    mean = data.mean()
    std = data.std()

    anomalies = df[(data > mean + 2*std) | (data < mean - 2*std)]
    return anomalies
