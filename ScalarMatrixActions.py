# Load library
import numpy as np

# Initialize a matrix
A = np.array([[1, 2, 4], [5, 3, 2]])
B = np.array([[1, 3, 4], [1, 1, 1]])

# Initialize constant s
s = 2

# See how element-wise addition works
add_AB = A + B
print(f"Addition:\n{add_AB}")

# See how element-wise subtraction works
sub_AB = A - B
print(f"Subtraction:\n{sub_AB}")

# See how scalar multiplication works
mul_As = A * s
print(f"Scalar multiplication:\n{mul_As}")

# Divide A by s
div_As = A / s
print(f"Division:\n{div_As}")

# What happens if we have a Matrix + scalar?
add_As = A + s
print(f"Matrix & scalar addition:\n{add_As}")
# it adds the scalar to each element in the matrix
