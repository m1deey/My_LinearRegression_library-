import numpy as np

from linear_regression import LinearRegression
from metrics import mse, r2_score

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, r2_score as sklearn_r2


# ------------------------
# Dataset
# ------------------------

X_train = np.array([
    [1, 2],
    [2, 5],
    [3, 1],
    [4, 7],
    [5, 3]
])

y_train = np.array([8, 15, 10, 20, 16])

# ------------------------
# My model
# ------------------------

model = LinearRegression(learning_rate=0.01, epochs=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_train)

print("---- My Model ----")
print("Weights:", model.w)
print("Bias:", model.b)
print("MSE:", mse(y_train, y_pred))
print("R²:", r2_score(y_train, y_pred))


# ------------------------
# sklearn model
# ------------------------

sk_model = SklearnLinearRegression()
sk_model.fit(X_train, y_train)

sk_pred = sk_model.predict(X_train)

print("\n---- sklearn ----")
print("Weights:", sk_model.coef_)
print("Bias:", sk_model.intercept_)
print("MSE:", mean_squared_error(y_train, sk_pred))
print("R²:", sklearn_r2(y_train, sk_pred))

---- My Model ----
Weights: [1.5037037  1.49074074]
Bias: 3.922222222220146
MSE: 0.13962962962963005
R²: 0.9924768518518519

---- sklearn ----
Weights: [1.5037037  1.49074074]
Bias: 3.9222222222222243
MSE: 0.13962962962962963
R²: 0.9924768518518519
