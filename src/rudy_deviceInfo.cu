/**
  * @file rudy_deviceInfo.cu
	* @author Kurt Robert Rudolph
	* @description This file defines functions which 
	* gather info about a device
	*/
#include "rudy_deviceInfo.h"

deviceInfo * rudy_deviceInfo_gather( void) {
	int i;
	deviceInfo * devices = (deviceInfo*) malloc( sizeof( deviceInfo));
	HANDLE_ERROR( cudaGetDeviceCount( &( devices->deviceCount)));
	devices->devicePropertiesArray = (cudaDeviceProp**) malloc( sizeof( cudaDeviceProp*) * devices->deviceCount);
	for( i= 0; i< devices->deviceCount; i++) {
		devices->devicePropertiesArray[i] = (cudaDeviceProp*) malloc( sizeof( cudaDeviceProp));	
		HANDLE_ERROR( cudaGetDeviceProperties( devices->devicePropertiesArray[i], i));
	}
	return devices;
}

void rudy_deviceInfo_free( deviceInfo * devices) {
	int i;	
	for( i= 0; i< devices->deviceCount; i++) {
		free( devices->devicePropertiesArray[i]);
		devices->devicePropertiesArray[i] = NULL;
	}
	free( devices->devicePropertiesArray);
	devices->devicePropertiesArray = NULL;
	free( devices);
	devices = NULL;
}

void rudy_deviceInfo_print( deviceInfo * devices) {
	int i;
	for ( i=0; i< devices->deviceCount; i++) {
		printf( "   --- General Information for device %d ---\n", i);
		printf( "Name:  %s\n", devices->devicePropertiesArray[i]->name);
		printf( "Compute capability:  %d.%d\n", devices->devicePropertiesArray[i]->major, devices->devicePropertiesArray[i]->minor);
		printf( "Clock rate:  %d\n", devices->devicePropertiesArray[i]->clockRate);
		printf( "Device copy overlap:  ");
		if( devices->devicePropertiesArray[i]->deviceOverlap)
			printf( "Enabled\n");
		else
			printf( "Disabled\n");
		printf( "Kernel execution timeout :  ");
		if( devices->devicePropertiesArray[i]->kernelExecTimeoutEnabled)
			printf( "Enabled\n");
		else
			printf( "Disabled\n");
		printf( "   --- Memory Information for device %d ---\n", i);
		printf( "Total global mem:  %ld\n", devices->devicePropertiesArray[i]->totalGlobalMem);
		printf( "Total constant Mem:  %ld\n", devices->devicePropertiesArray[i]->totalConstMem);
		printf( "Max mem pitch:  %ld\n", devices->devicePropertiesArray[i]->memPitch);
		printf( "Texture Alignment:  %ld\n", devices->devicePropertiesArray[i]->textureAlignment);
		printf( "   --- MP Information for device %d ---\n", i);
		printf( "Multiprocessor count:  %d\n", devices->devicePropertiesArray[i]->multiProcessorCount);
		printf( "Shared mem per mp:  %ld\n", devices->devicePropertiesArray[i]->sharedMemPerBlock);
		printf( "Registers per mp:  %d\n", devices->devicePropertiesArray[i]->regsPerBlock);
		printf( "Threads in warp:  %d\n", devices->devicePropertiesArray[i]->warpSize);
		printf( "Max threads per block:  %d\n",	devices->devicePropertiesArray[i]->maxThreadsPerBlock);
		printf( "Max thread dimensions:  (%d, %d, %d)\n",	devices->devicePropertiesArray[i]->maxThreadsDim[0], devices->devicePropertiesArray[i]->maxThreadsDim[1], devices->devicePropertiesArray[i]->maxThreadsDim[2]);
		printf( "Max grid dimensions:  (%d, %d, %d)\n", devices->devicePropertiesArray[i]->maxGridSize[0], devices->devicePropertiesArray[i]->maxGridSize[1], devices->devicePropertiesArray[i]->maxGridSize[2]);
		printf( "\n");
	}
}
