import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.w = None
        self.b = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def fit(self, X_train, y_train):

        n_features = X_train.shape[1]


        self.w = np.zeros(n_features)
        self.losses = []


        for _ in range(self.epochs):



            y_pred = self.predict(X_train)

            loss = np.mean((y_pred - y_train) ** 2)


            dw = (2 / len(X_train)) * np.dot(X_train.T, (y_pred - y_train))
            db = (2 / len(X_train)) * np.sum(y_pred - y_train)


            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            self.losses.append(loss)
        return self
    
