import numpy as np
import mkl
import time

# Function to perform matrix multiplication and eigen decomposition
def matrix_operations(size, num_trials):
    # Set the number of threads for MKL
    mkl.set_num_threads(4)  # Adjust based on your CPU cores

    # Generate random matrices
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    # Measure time for matrix multiplication
    start_time = time.time()
    for _ in range(num_trials):
        C = np.dot(A, B)  # Matrix multiplication
    elapsed_time = time.time() - start_time
    print(f"Matrix multiplication of size {size}x{size} took {elapsed_time:.4f} seconds.")

    # Measure time for eigen decomposition
    start_time = time.time()
    for _ in range(num_trials):
        eigenvalues, eigenvectors = np.linalg.eig(C)  # Eigen decomposition
    elapsed_time = time.time() - start_time
    print(f"Eigen decomposition of size {size}x{size} took {elapsed_time:.4f} seconds.")

    return eigenvalues, eigenvectors

# Main function to run the program
if __name__ == "__main__":
    size = 500  # Size of the matrices
    num_trials = 10  # Number of trials for averaging

    print("Starting matrix operations...")
    eigenvalues, eigenvectors = matrix_operations(size, num_trials)

    print("Eigenvalues:")
    print(eigenvalues)
