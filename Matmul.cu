#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#define blockSize 10

// CUDA kernel .

__global__ void matmul(double *A, double *B, double *C, int n)
{
	int k, temp = 0;
	// get global threadID

	int r = blockIdx.x*blockDim.x+threadIdx.x;
	int c = blockIdx.y*blockDim.y+threadIdx.y;
	
	if(r<n && c < n)
	{
		for(int k=0;k<n;k++)
		{
	
			temp=temp+A[r*n+k]*B[k*n+c];

		}
	C[r*n+c]=temp;
}

void cpu_mm(int *cpu_a, int *cpu_b,int *cpu_c, int n)
{

	int i,j,k,sum;
	for(r=0;r<n,r++)
	{
		for(c=0;c<n,r++)
		{
		sum = 0;
			for(k=0;k<n;k++)
			{
				sum+=cpu_a[r*n+k]*cpu_b[k*n+c];
			}
		cpu_c[r*n+c]=sum;
		}
	}
}


int main(int argc, char* argv[])
{
	// perform matrix multiplication C = A*B
	// A, B and C of size N x N
	// input the host matrix

	int i,j;
	int Grid_dim_x = 1, Grid_dim_y = 1; // grid structure values
	int Block_dim_x = 1,Block_dim_y = 1, // block structure values
	int noThreads_x,no_Threads_y; // number of threads available in device, each dimension
	int noThreads_block; // number of threads in block

	int n = 10;

	int *A, *B, *C, *D;

	int *dev_A,*dev_B,*dev_C;

	int size; // number of bytes in the array

	cudaEvent_t start,stop   // cuda events to measure time
	
	float elapsed_time_ms;  // time for the code

/*****************************************************************************************************/



	
	
