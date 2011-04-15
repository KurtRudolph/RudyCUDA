/** 
	* @file gatherDeviceInfo.h
	* @author Kurt Robert Rudolph
	* @description This is the header file for a
  * set of functions which gather and store
	* information about a the availble CUDA 
	* enabled devices.
	*/


#ifndef __RUDY_DEVICE_INFO_H__
#define __RUDY_DEVICE_INFO_H__

#include <rudyCUDA.h>

struct rudy_cudaDeviceProp_t {
	cudaDeviceProp prop;
	char* name[256];
	size_t* totalGloblalMemory;
	int* regsPerBlock;
	int* warpSize;
	size_t* memPitch;
	int* maxThreadsPerBlock;
	int* maxThreadsDim[3];
	int* maxGridSize[3];
	size_t* totalConstMem;
	int* major;
	int* minor;
	int* clockrate;
	size_t* textureAlignment;
	int* deviceOverlap;
	int* multiProcessorCount;
	int* kernelExecTimeoutEnabled;
	int* integrated;
	int* canMapHostMemory;
	int* computeMode;
	int* maxTexture1D;
	int* maxTexture2D[2];
	int* maxTexture3D[3];
	int* maxTexture2DArray[3];
	int* concurrentKernels;
}rudy_cudaDeviceProp;

/**
  * @brief gather information for the available devices
  * @return devicePropertiesArray an of points to #rudy_cudaDeviceProp
	* storeing the properties of the availble devices
  */ 
cudaDeviceProp ** rudy_deviceInfo_gather(void);

#endif
