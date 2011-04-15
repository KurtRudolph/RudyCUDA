/**
	* @file test_gatherDeviceInfo
	* @author Kurt Robert Rudolph
	* @description This file tests the functionality 
	* of gatherDeviceInfo.c
	*/
#include "test_.h"
	
void TestCuStringNew(CuTest* tc)
{
	int i;
	int deviceCount;
	HANDLE_ERROR( cudaGetDeviceCount (&deviceCount));
	cudaDeviceProp ** devicePropertiesArray = gatherDeviceInfo();
	
	for (i = 0; i< deviceCount; i++){	
		CuAssertTrue (tc, devicePropertiesArray[i] != NULL); 
	}	
}


CuSuite* CuGetSuite(void)
{
	CuSuite* suite = CuSuiteNew();
	return suite;
}
