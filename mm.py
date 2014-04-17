import numpy
import pyopencl as cl
import numpy as np

# create host side array

N = 10  # zise of the array

A_host = 2*np.array((N,N),dtype = numpy.float32)

B_host = 3*np.array((N,N),dtype = numpy.float32)

C_host = 0*np.array((N,N),dtype = numpy.float32)

# openCl C source code for parallel multiplication of two matrix A and B

kernel = """

__kernel__ void matmul(__global float *A, __global float *B, __global float *C)

{

	// each kernel instance has a different global ID

	

