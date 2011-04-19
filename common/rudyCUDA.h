#ifndef __RUDY_CUDA_H__
#define __RUDY_CUDA_H__
#include <stdio.h>

static void HandleError( cudaError_t err,
						const char *file,
						int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
			   file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define HANDLE_NULL( a ) {if (a == NULL) { printf( "Host memory failed in %s at line %d\n", __FILE__, __LINE__ ); exit( EXIT_FAILURE );}}

//The macro CUPRINTF is defined for architectures
//with different compute capabilities.
#if __CUDA_ARCH__ < 200   //Compute capability 1.x architectures
#define CUPRINTF cuPrintf
#else           //Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
                blockIdx.y*gridDim.x+blockIdx.x,\
                threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
                __VA_ARGS__)
#endif


#endif  
