#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#define blockSize 10

// CUDA kernel .

__global__ void matmul(double *A, double *B, double *C, int n)
{

	// get global threadID

	int r = blockIdx.x*blockDim.x+threadIdx.x;
	int c = blockIdx.y*blockDim.y+threadIdx.y;
	
	float temp=0.0;
	for(int k=0;k<n;k++)
	{
	
		temp=temp+A[r*n+k]*B[k*n+c];

	}
	C[r*k+c]=temp;
}

int main(int argc, char* argv[])
{
	// perform matrix multiplication C = A*B
	// A, B and C of size N x N
	// input the host matrix

	int n,m;

	m =10;
	n=m*blockSize;

	printf("Executing matrix multiplication");
	printf("Matrix size : %f",n);

	// allocating memory to the host

	double *h_A, *h_B, *h_C;

	h_A = double[N][N];
	h_B = double[N][N];
	h_C = double[N][N];

	// initialize the matrix on host

	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			h_A[i][j] = 2.0;
			h_B[i][j] = 3.0;
		}
	}

	// allocate memory on device

	int size = n*n*sizeof(double) // size of the memory in bites

	// device input matrix
	double *d_A, *d_B, *d_C;
	
	// allocate memory to each matrix on GPU
	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_C, size);

	dim3 threadBlock(blockSize,blockSize);
	dim3 gridSize(m,m);

	// copy host to device

	cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_B,bytes,cudaMemcpyHostToDevice);

	// execute the kernel

	matmul<<<gridSize, threadBlock>>>(d_A, d_B, d_C, n);

	// matrix multiplication on the CPU

	float temp;
	for (int r=0;r<n;r++)
	{
		for(int c=0;c<n;c++)	
		{	
		temp = 0.0
			for(int k=0;k<n;k++)
			{
			temp=temp+h_A[r*n+k]*h_B[k*n+c];
			}
			h_C[r*n+c]=temp;	
		}
	}

	// memory to store memory on the GPU host

	float *C

	C = float[n][n];

	// copy the GPU result back to CPU

	cudaMemcpy(C,d_C, size,cudaMemcpyDeviceToHost);

	// checking the answer to check if things are woring corect

	for(int r=0,r<n,r++)
	{
		for(int c=0;c<n;c++)
		{
			if(C[r*n+c]!=h_C[r*n+c])
			{
				printf("wrong answer!");
				r=c=n;
			}
		}
	}

	printf("End of the code");


	// release device memory

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// release host memory

	free(h_A);
	free(h_B);
	free(h_C);

return 0;

}

