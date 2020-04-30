# Load library
import numpy as np

# Initialize a matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# Initialize a vector
v = np.array([[1], [2], [3]])

# Prints the dimensions of a matrix/vector; returns a tuple
print(A.shape)
print(v.shape)

# Prints the number of elements of the matrix/vector
print(A.size)
print(v.size)

# Get the value of the 2nd row and 3rd column of matrix A
A_23 = A[1][2]
print(A_23)
