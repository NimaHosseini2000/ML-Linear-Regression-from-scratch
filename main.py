"""
Linear Regression from Scratch
================================
Implements linear regression using gradient descent — no ML libraries.
Demonstrates the math behind the algorithm: MSE loss, partial derivatives,
and iterative parameter updates.

Dataset: synthetic (advertising spend vs. sales), generated with NumPy.

Author : Nima Hosseini
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# 1. Generate Synthetic Dataset
# -------------------------------

np.random.seed(42)  # ensures reproducibility

# Independent variable (feature)
total_spend = np.random.uniform(20, 100, 100)

# Dependent variable with linear relationship + noise
sales = 0.8 * total_spend + np.random.normal(0, 5, 100)

# Ensure values remain between 20 and 100
sales = np.clip(sales, 20, 100)

# Create DataFrame
df = pd.DataFrame({
    "total_spend": total_spend,
    "sales": sales
})

print(df.head())


# -------------------------------
# 2. Loss Function (Mean Squared Error)
# -------------------------------

def loss_function(m, b, points):
    """
    Compute Mean Squared Error (MSE) for a linear model y = mx + b.

    Parameters
    ----------
    m : float
        Slope of the regression line.
    b : float
        Intercept of the regression line.
    points : pd.DataFrame
        DataFrame with columns 'total_spend' (x) and 'sales' (y).

    Returns
    -------
    float
        Average squared error across all data points.
    """
    total_error = 0

    for i in range(len(points)):
        x = points.iloc[i].total_spend
        y = points.iloc[i].sales

        # Squared error for this point
        total_error += (y - (m * x + b)) ** 2

    return total_error / float(len(points))


# -------------------------------
# 3. Gradient Descent Algorithm
# -------------------------------

def gradient_descent(m_now, b_now, points, L):
    """
    Perform one step of gradient descent to update slope and intercept.

    Computes the partial derivatives of MSE with respect to m and b,
    then moves each parameter in the direction that reduces the loss.

    Parameters
    ----------
    m_now : float
        Current slope estimate.
    b_now : float
        Current intercept estimate.
    points : pd.DataFrame
        Training data with columns 'total_spend' and 'sales'.
    L : float
        Learning rate — controls step size.

    Returns
    -------
    tuple[float, float]
        Updated (m, b) after one gradient descent step.
    """
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].total_spend
        y = points.iloc[i].sales

        prediction = m_now * x + b_now
        error = y - prediction

        # Partial derivatives of MSE w.r.t. m and b
        m_gradient += -(2 / n) * x * error
        b_gradient += -(2 / n) * error

    # Update parameters
    m = m_now - L * m_gradient
    b = b_now - L * b_gradient

    return m, b


# -------------------------------
# 4. Training Loop
# -------------------------------

def train(df, learning_rate=0.0001, epochs=300):
    """
    Train the linear regression model via gradient descent.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    learning_rate : float
        Step size for parameter updates.
    epochs : int
        Number of full passes over the dataset.

    Returns
    -------
    tuple[float, float]
        Final (slope, intercept) after training.
    """
    m = 0  # initial slope
    b = 0  # initial intercept

    for i in range(epochs):

        # Print progress every 50 epochs
        if i % 50 == 0:
            loss = loss_function(m, b, df)
            print(f"Epoch {i:>3} | Loss: {loss:.4f}")

        m, b = gradient_descent(m, b, df, learning_rate)

    return m, b


# -------------------------------
# 5. Visualization
# -------------------------------

def plot_results(df, m, b):
    """
    Scatter-plot the data and overlay the fitted regression line.

    Parameters
    ----------
    df : pd.DataFrame
        Original data with 'total_spend' and 'sales' columns.
    m : float
        Fitted slope.
    b : float
        Fitted intercept.
    """
    plt.scatter(df["total_spend"], df["sales"], label="Data points")

    # Generate smooth regression line across the data range
    x_range = np.linspace(df.total_spend.min(), df.total_spend.max(), 100)
    y_range = m * x_range + b

    plt.plot(x_range, y_range, color="red", label=f"y = {m:.2f}x + {b:.2f}")

    plt.xlabel("Total Spend")
    plt.ylabel("Sales")
    plt.title("Linear Regression from Scratch")
    plt.legend()
    plt.show()


# -------------------------------
# Entry Point
# -------------------------------

if __name__ == "__main__":
    m, b = train(df, learning_rate=0.0001, epochs=300)

    print("\nFinal parameters:")
    print(f"  Slope (m)     : {m:.4f}")
    print(f"  Intercept (b) : {b:.4f}")

    plot_results(df, m, b)
