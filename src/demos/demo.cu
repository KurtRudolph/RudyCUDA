#include "rudyCUDA_deviceInfo.h"

int main (int argc, char *argv[]) {
	
	rudyCUDA_deviceInfo *devices = NULL;
	rudyCUDA_deviceInfo_gather( &devices);	
	
	rudyCUDA_deviceInfo_print( &devices);
	
	rudyCUDA_deviceInfo_free( &devices);	
	return 0;
}
