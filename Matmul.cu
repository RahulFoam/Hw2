#include "stdio.h"
#include "stdlib.h"
#include "math.h"

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
	C[r*+c]=temp;
}

int main(int argc, char* argv[])
{

	//size of matrix
	n=100

	// input the host matrix
	double *h_A;
	double *h_B;

	// output matrix
	double *h_C;

	// device input vectors
	double *d_A;
	double *d_B;

	// output vector
	double *d_C;

	// size in byte of the vector
	size_t bytes = n*sizeof(double);
	
	// alocate memory to each matrix on the host
	h_A = (double*)malloc(bytes);
	h_B = (double*)malloc(bytes);
	h_C = (double*)malloc(bytes);

	// allocate memory to each matrix on GPU
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	int i,j

	// initialize the matrix on host

	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			h_A[i][j] = 2;
			h_B[i][j] = 3;
		}
	}

	// copy host to device

	cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_B,bytes,cudaMemcpyHostToDevice);

	int blockSize, gridSize;

	// number of threads in each block
	blockSize = 64;

	// number of thread blocks in grid

	gridSize = int(cell)((float)n/blockSize);

	// execute the kernel

	matmul<<<<gridSize, blockSize>>>>(d_A, d_B, d_C, n);

	// copy array back to host

	cudaMemcpy(h_C,d_C, bytes,cudaMemcpyDeviceToHost);

	// print the result

	for(i=0;i<n;i++)
	{
    		printf("\n");
			for(j=0;j<=N-1;j++)
			{
				printf("C[%d][%d]=%d\t \n",i,j,C[i][j]);
			}
	}	

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
