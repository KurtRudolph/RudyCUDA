#include "rudy_deviceInfo.h"

int main (int argc, char *argv[]) {
	deviceInfo* devices = rudy_deviceInfo_gather();	
	rudy_deviceInfo_print( devices);
	
	rudy_deviceInfo_free( devices);	
	return 0;
}
