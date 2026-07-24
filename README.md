# My Linear Regression Library

A Linear Regression model built completely from scratch using **NumPy** and **Gradient Descent**.

This project was created to understand how Linear Regression works internally without relying on machine learning libraries.

## Features

- Linear Regression implemented from scratch
- Gradient Descent optimization
- Prediction on new data
- Mean Squared Error (MSE)
- R² Score
- Comparison with scikit-learn
- Training loss visualization

## Project Structure

```
.
├── linear_regression.py
├── metrics.py
├── example.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/m1deey/My_LinearRegression_library.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the example:

```bash
python example.py
```

## Example Output

```
----- My Model -----
Weights: [...]
Bias: ...
MSE: ...
R²: ...

----- scikit-learn -----
Weights: [...]
Bias: ...
MSE: ...
R²: ...
```

## Technologies Used

- Python
- NumPy
- Matplotlib
- scikit-learn (for comparison only)

## Learning Goals

This project demonstrates:

- How Gradient Descent updates model parameters
- How Linear Regression is trained
- How MSE and R² are calculated
- How a custom implementation compares with scikit-learn

## Future Improvements

- Mini-batch Gradient Descent
- L1/L2 Regularization
- Model saving and loading
- Polynomial Regression
- Better documentation and unit tests

## Author

**Yasser**  
GitHub: https://github.com/m1deey
