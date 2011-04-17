#include "rudy_deviceInfo.h"

int main (int argc, char *argv[]) {
	cudaDeviceProp ** devicePropertiesArray = rudy_deviceInfo_gather();	
	printf( "\ncomputeMode: %d\n", devicePropertiesArray[0]->computeMode);
	rudy_deviceInfo_free( devicePropertiesArray);	
	return 0;
}
