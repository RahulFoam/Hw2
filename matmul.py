import pyopencl as cl
import numpy as np
import time

# context

ctx = cl.create_some_context()
que=cl.CommandQueue(ctx)

# problem size

N = 3500

a = np.array(np.ones((N,N)),dtype=np.float32)

b = np.array(np.ones((N,N)),dtype=np.float32)

c = np.empty(a.shape,dtype=np.float32)

a_buf = cl.Buffer(ctx,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=a)

b_buf = cl.Buffer(ctx,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=b)

c_buf = cl.Buffer(ctx,cl.mem_flags.WRITE_ONLY,a.nbytes)

#
# Matrix-Matrix multiplication: c = a*b kernel
#

kcode = """

__kernel void matmul(__global float* a,__global float* b,__global float* c)
{

int x = get_global_id(0);

int y = get_global_id(1);

const int N = 3500;

int k;

int sum = 0;

for(k=0;k<N;k++)
{

sum+=a[x*N+k]*b[k*N+y];

}	
c[x*N+y]=sum;
}
"""

t0=time.time()

prg =cl.Program(ctx,kcode).build()

run = prg.matmul(que,a.shape,None,a_buf,b_buf,c_buf)

run.wait()

cl.enqueue_copy(que,c,c_buf)

t1=time.time()

print(c)

print "Time required =",(t1-t0)		


