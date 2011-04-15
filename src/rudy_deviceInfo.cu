/**
  * @file rudy_deviceInfo.cu
	* @author Kurt Robert Rudolph
	* @description This file defines functions which 
	* gather info about a device
	*/
#include "rudy_deviceInfo.h"

cudaDeviceProp ** rudy_deviceInfo_gather(void){
	int i;
	int deviceCount;
	HANDLE_ERROR( cudaGetDeviceCount( &deviceCount));
	cudaDeviceProp ** devicePropertiesArray = (cudaDeviceProp**) malloc (sizeof (cudaDeviceProp*) * deviceCount);
	for (i = 0; i< deviceCount; i++) {
		devicePropertiesArray[i] = (cudaDeviceProp*) malloc (sizeof (cudaDeviceProp));	
		HANDLE_ERROR( cudaGetDeviceProperties (devicePropertiesArray[i], i));
	}
	printf("\n rud_deviceInfo_gather()\n");

	return devicePropertiesArray;
}
