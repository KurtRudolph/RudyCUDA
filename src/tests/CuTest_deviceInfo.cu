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
	HANDLE_ERROR( cudaGetDeviceCount (&deviceCount));
	cudaDeviceProp ** devicePropertiesArray = rudy_deviceInfo_gather();
	
	for (i = 0; i< deviceCount; i++){	
		CuAssertTrue (tc, devicePropertiesArray[i] != NULL); 
	}	
}

