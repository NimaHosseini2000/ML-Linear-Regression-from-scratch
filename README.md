# Linear Regression from Scratch

A pure-Python implementation of linear regression trained with gradient descent — no scikit-learn or ML frameworks. Built to understand and demonstrate the math behind the algorithm.

## What it covers

- Generating a synthetic dataset (advertising spend → sales)
- Mean Squared Error (MSE) loss function
- Gradient descent with partial derivatives
- Iterative parameter updates (slope & intercept)
- Visualization of the fitted regression line

## How to run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/linear-regression-from-scratch.git
cd linear-regression-from-scratch

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

## Sample output

```
Epoch   0 | Loss: 2489.7321
Epoch  50 | Loss: 42.3107
Epoch 100 | Loss: 27.9854
Epoch 150 | Loss: 27.9512
Epoch 200 | Loss: 27.9511
Epoch 250 | Loss: 27.9511

Final parameters:
  Slope (m)     : 0.7891
  Intercept (b) : 5.3214
```

A scatter plot with the fitted line is displayed at the end.

## Tech stack

| Library    | Purpose                   |
|------------|---------------------------|
| NumPy      | Data generation & math    |
| pandas     | DataFrame handling        |
| matplotlib | Visualization             |

## Key concepts

**MSE Loss**

$$L(m,b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2$$

**Gradient Descent Update**

$$m \leftarrow m - \alpha \cdot \frac{\partial L}{\partial m}, \quad b \leftarrow b - \alpha \cdot \frac{\partial L}{\partial b}$$

## Author

Nima Hosseini
