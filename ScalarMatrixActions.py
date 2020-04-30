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
# it adds the scalar to each element in the matrix
add_As = A + s
print(f"Matrix & scalar addition:\n{add_As}")

A = np.array([[1, 1], [1, 2]])
B = np.array([[1, 3], [1, 2]])

# Multiplying matrices:
mul_AB = np.dot(B,A)
print(f"Matrix multiplication:\n{mul_AB}")

# Transposing a matrix: (does not change the caller)
C = np.array([[1, 2, 3], [4, 5, 6]])
C = C.T
print(f"Matrix C transposed is:\n{C}")

# Inverting  a matrix:
D = np.array([[3, 4], [2, 16]])
D = np.linalg.inv(D)
print(f"Matrix D^-1 is:\n{D}")
