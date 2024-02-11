import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad

# Define the g function
def g(y, lambda2):
    # Define the infinite product; in practice, we truncate it (N=100 for this example)
    product_result = np.product([(1 + (y / (lambda2 + 2*j))**2)**-1 for j in range(10001)])
    # Calculate the complete g(y|lambda2)
    return 2**(lambda2 - 2) * gamma(lambda2/2)**2 / (np.pi * gamma(lambda2)) * product_result

# Define the f function
def f(lambda1, lambda2, y):
    return np.exp(lambda1 * y + lambda2 * np.log(np.cos(lambda1))) * g(y, lambda2)

# Create a grid of values for lambda1 and lambda2
lambda1_vals = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, 100)
lambda2_vals = np.linspace(0.01, 10, 100)
y_value = 1  # Fixed y value

# Calculate the matrix of function values
f_matrix = np.zeros((100, 100))

for i, lambda1 in enumerate(lambda1_vals):
    for j, lambda2 in enumerate(lambda2_vals):
        f_matrix[i, j] = f(lambda1, lambda2, y_value)

# Generate contour plot
plt.figure(figsize=(8, 6))
cp = plt.contour(lambda1_vals, lambda2_vals, -np.log(f_matrix.T), levels=40)  # Transpose matrix to align axes
plt.clabel(cp, inline=True, fontsize=8)
plt.title('Contour Plot of f')
plt.xlabel('lambda1')
plt.ylabel('lambda2')
plt.show()

print(-np.log(f(1, 10, 1)))