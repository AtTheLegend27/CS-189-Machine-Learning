#Question 6.1: The Einsum Function
import numpy as np


#1: Trace of a 5x5 matrix
A = np.random.rand(5, 5)
trace_np = np.trace(A)
trace_einsum = np.einsum('ii', A)
print("6.1.1: Trace")
print("Trace using np.trace:", trace_np)
print("Trace using np.einsum:", trace_einsum)
print("Norm of difference:", np.linalg.norm(trace_np - trace_einsum))
print()

#2: Matrix product of two 5x5 matrices
B = np.random.rand(5, 5)
product_np = np.dot(A, B)
product_einsum = np.einsum('ij,jk->ik', A, B)
print("6.1.2: Matrix Product")
print("Norm of difference:", np.linalg.norm(product_np - product_einsum))
print()

#3: Batchwise matrix multiplication
batch1 = np.random.rand(3, 4, 5)
batch2 = np.random.rand(3, 5, 6)
batch_product_np = np.matmul(batch1, batch2)
batch_product_einsum = np.einsum('ijk,ikl->ijl', batch1, batch2)
print("6.1.3: Batchwise Matrix Product")
print("Norm of difference:", np.linalg.norm(batch_product_np - batch_product_einsum))
