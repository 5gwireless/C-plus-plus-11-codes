https://github.com/puttsk/cuda-tutorial/tree/master/tutorial01


/* Putting things in actions.
The CUDA hello world example does nothing, and even if the program is compiled, nothing will show up on screen. To get things into action, we will looks at vector addition.

 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i ++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Executing kernel 
    vector_add<<<1,1>>>(d_out, d_a, d_b, N);
    
    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}


//$> nvcc vector_add.cu -o vector_add
//$> time ./vector_add


/*
Using time does not give much information about the program performance. NVIDIA provides a commandline profiler tool called nvprof, which give a more insight
information of CUDA program performance. To profile our vector addition, use following command

$> nvprof ./vector_add
*/

// In this tutorial, we demonstrate how to write a simple vector addition in CUDA. We introduced GPU kernels and its execution from host code. Moreover, we introduced the concept of separated memory space between CPU and GPU. We also demonstrate how to manage the device memory.
// However, we still not run program in parallel. The kernel execution configuration <<<1,1>>> indicates that the kernel is launched with only 1 thread. In the next tutorial, we will modify vector addition to run in parallel.
//======================================================================================================================================

// Exercise 1: Parallelizing vector addition using multithread

// In tutorial , we implemented vector addition in CUDA using only one GPU thread. However, the strength of GPU lies in its massive parallelism. In this tutorial, we will explore how to exploit GPU parallelism.

/* Going parallel
CUDA use a kernel execution configuration <<<...>>> to tell CUDA runtime how many threads to launch on GPU. CUDA organizes threads into a group called "thread block". Kernel can launch multiple thread blocks, organized into a "grid" structure.
The syntax of kernel execution configuration is as follows

<<< M , T >>>
Which indicate that a kernel launches with a grid of M thread blocks. Each thread block has T parallel threads.

*/


/*
In this exercise, we will parallelize vector addition from tutorial 01

The new kernel execution configuration is shown below.

vector_add <<< 1 , 256 >>> (d_out, d_a, d_b, N);
CUDA provides built-in variables for accessing thread information. In this exercise, we will use two of them: threadIdx.x and blockIdx.x.

threadIdx.x contains the index of the thread within the block
blockDim.x contains the size of thread block (number of threads in the thread block).
For the vector_add() configuration, ะhe value of threadIdx.x ranges from 0 to 255 and the value of blockDim.x is 256. */



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    for(int i = index; i < n; i += stride){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory 
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Executing kernel 
    vector_add<<<1,256>>>(d_out, d_a, d_b, N);
    
    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}

/*
Following is the profiling result on Tesla M2050

==6430== Profiling application: ./vector_add_thread
==6430== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
 39.18%  22.780ms         1  22.780ms  22.780ms  22.780ms  vector_add(float*, float*, float*, int)
 34.93%  20.310ms         2  10.155ms  10.137ms  10.173ms  [CUDA memcpy HtoD]
 25.89%  15.055ms         1  15.055ms  15.055ms  15.055ms  [CUDA memcpy DtoH]
*/

//$> nvcc vector_add_thread.cu -o vector_add_thread
//$> nvprof ./vector_add_thread

//==================================================================================================

/*Exercise 2: Adding more thread blocks
CUDA GPUs have several parallel processors called Streaming Multiprocessors or SMs. Each SM consists of multiple parallel processors and can run multiple concurrent thread blocks. To take advantage of CUDA GPUs, kernel should be launched with multiple thread blocks. This exercise will expand the vector addition from exercise 1 to uses multiple thread blocks.

Similar to thread information, CUDA provides built-in variables for accessing block information. In this exercise, we will use two of them: blockIdx.x and gridDim.x.

blockIdx.x contains the index of the block with in the grid
gridDim.x contains the size of the grid */


// With 256 threads per thread block, we need at least N/256 thread blocks to have a total of N threads. To assign a thread to a specific element, we need to know a unique index for each thread. Such index can be computed as follow
// int tid = blockIdx.x * blockDim.x + threadIdx.x;

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handling arbitrary vector size
    if (tid < n){
        out[tid] = a[tid] + b[tid];
    }
}

int main(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory 
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);


    // Executing kernel 
    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);
    vector_add<<<grid_size,block_size>>>(d_out, d_a, d_b, N);
    
    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}



//$> nvcc vector_add_grid.cu -o vector_add_grid
//$> nvprof ./vector_add_grid

/*
Following is the profiling result on Tesla M2050

==6564== Profiling application: ./vector_add_grid
==6564== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
 55.65%  20.312ms         2  10.156ms  10.150ms  10.162ms  [CUDA memcpy HtoD]
 41.24%  15.050ms         1  15.050ms  15.050ms  15.050ms  [CUDA memcpy DtoH]
  3.11%  1.1347ms         1  1.1347ms  1.1347ms  1.1347ms  vector_add(float*, float*, float*, int)
*/


/*
Following is the profiling result on Tesla M2050

==6564== Profiling application: ./vector_add_grid
==6564== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
 55.65%  20.312ms         2  10.156ms  10.150ms  10.162ms  [CUDA memcpy HtoD]
 41.24%  15.050ms         1  15.050ms  15.050ms  15.050ms  [CUDA memcpy DtoH]
  3.11%  1.1347ms         1  1.1347ms  1.1347ms  1.1347ms  vector_add(float*, float*, float*, int)


*/

//===================================================================================================

Comparing Performance
	Version	        Execution Time(ms)	   Speedup
	1 thread	       1425.29	             1.00x
    1 block	            22.78	             62.56x
     Multiple blocks	1.13	            1261.32x