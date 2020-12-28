
//In the simplest of terms, developers can group threads into blocks and a collection of blocks form a grid. Maximum number of threads per block is 1024. 
//The number of blocks per grid can go up to 2³¹-1, i.e. 2,147,483,647. i.e. the maximum value of a signed int



CUDA compiler and profiler is installed with the toolkit, the debugger and visual profiler may require separate installation.

Compiler — nvcc automatically adds the CUDA source headers and links the CUDA runtime libraries.

Profiler — nvprof command line profiler that profiles kernel execution times and runtime API calls

Debugger — cuda-gdb compiler with nvcc -g to enable verbose debugging output

Visual Profiler — nvvp the swiss army knife of CUDA application optimization. Includes detailed analysis of core utilization, register usage, global memory access and timing profiles.