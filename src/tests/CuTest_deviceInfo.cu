/**
	* @file test_gatherDeviceInfo
	* @author Kurt Robert Rudolph
	* @description This file tests the functionality 
	* of gatherDeviceInfo.c
	*/
#include <assert.h>
#include <setjmp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <rudy_deviceInfo.h>	

#include "CuTest.h"

void Test_deviceInfo_gather(CuTest* tc)
{
	int i;
	int deviceCount;
	cudaDeviceProp prop;
	deviceInfo * devices = rudy_deviceInfo_gather();

	HANDLE_ERROR( cudaGetDeviceCount (&deviceCount));
	CuAssertTrue( tc, deviceCount == devices->deviceCount);	

	for( i = 0; i< deviceCount; i++){
		HANDLE_ERROR( cudaGetDeviceProperties( &prop, i));

		CuAssertStrEquals( tc, prop.name, devices->devicePropertiesArray[i]->name);
		CuAssertTrue( tc, prop.major == devices->devicePropertiesArray[i]->major);
		CuAssertTrue( tc, prop.minor == devices->devicePropertiesArray[i]->minor);
		CuAssertTrue( tc, prop.clockRate == devices->devicePropertiesArray[i]->clockRate);
		CuAssertTrue( tc, prop.deviceOverlap == devices->devicePropertiesArray[i]->deviceOverlap);
		CuAssertTrue( tc, prop.kernelExecTimeoutEnabled == devices->devicePropertiesArray[i]->kernelExecTimeoutEnabled);
		CuAssertTrue( tc, prop.totalGlobalMem == devices->devicePropertiesArray[i]->totalGlobalMem);
		CuAssertTrue( tc, prop.totalConstMem == devices->devicePropertiesArray[i]->totalConstMem);
		CuAssertTrue( tc, prop.memPitch ==  devices->devicePropertiesArray[i]->memPitch);
		CuAssertTrue( tc, prop.textureAlignment == devices->devicePropertiesArray[i]->textureAlignment);
		CuAssertTrue( tc, prop.multiProcessorCount == devices->devicePropertiesArray[i]->multiProcessorCount);
		CuAssertTrue( tc, prop.sharedMemPerBlock == devices->devicePropertiesArray[i]->sharedMemPerBlock);
		CuAssertTrue( tc, prop.regsPerBlock == devices->devicePropertiesArray[i]->regsPerBlock);
		CuAssertTrue( tc, prop.warpSize == devices->devicePropertiesArray[i]->warpSize);
		CuAssertTrue( tc, prop.maxThreadsPerBlock = devices->devicePropertiesArray[i]->maxThreadsPerBlock);
		CuAssertTrue( tc, prop.maxThreadsDim[0] == devices->devicePropertiesArray[i]->maxThreadsDim[0]);
		CuAssertTrue( tc, prop.maxThreadsDim[1] == devices->devicePropertiesArray[i]->maxThreadsDim[1]);
		CuAssertTrue( tc, prop.maxThreadsDim[2] == devices->devicePropertiesArray[i]->maxThreadsDim[2]);
		CuAssertTrue( tc, prop.maxGridSize[0] == devices->devicePropertiesArray[i]->maxGridSize[0]);
		CuAssertTrue( tc, prop.maxGridSize[1] == devices->devicePropertiesArray[i]->maxGridSize[1]);
		CuAssertTrue( tc, prop.maxGridSize[2] == devices->devicePropertiesArray[i]->maxGridSize[2]);
  }
	rudy_deviceInfo_free( devices);
}
	
void Test_deviceInfo_free(CuTest* tc)
{
	int i;
	int deviceCount;
	deviceInfo * devices = rudy_deviceInfo_gather();	
	

	HANDLE_ERROR( cudaGetDeviceCount (&deviceCount));	
	CuAssertTrue( tc, deviceCount == devices->deviceCount);	

	CuAssertTrue( tc, devices->devicePropertiesArray != NULL);
	cudaDeviceProp ** testPropPointerArray = devices->devicePropertiesArray;
	
	for( i = 0; i< deviceCount; i++){
		CuAssertTrue( tc, devices->devicePropertiesArray[i] != NULL);
		CuAssertTrue( tc, testPropPointerArray[i] == devices->devicePropertiesArray[i]);
	}

	rudy_deviceInfo_free(devices);

	for( i = 0; i< deviceCount; i++){
		CuAssertTrue( tc, devices == NULL);
		CuAssertTrue( tc, testPropPointerArray[i] == NULL);
	}	
}


