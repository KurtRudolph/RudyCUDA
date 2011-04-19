/** 
	* @file rudyCUDA_deviceInfo.h
	* @author Kurt Robert Rudolph
	* @description This is the header file for a
  * set of functions which gather and store
	* information about a the availble CUDA 
	* enabled devices.
	*/


#ifndef __RUDY_DEVICE_INFO_H__
#define __RUDY_DEVICE_INFO_H__

#include <rudyCUDA.h>

/**
  * @struct deviceInfo
  * @brief Stores available CUDA device info
  * @param deviceCount 
  * The number of availbel CUDA devices (important for accessing the array)
  * @param devicePropertiesArray A pointer the a list of pointers of of available 
  * devices
  */
typedef struct rudyCUDA_deviceInfo_t {
	int deviceCount;
	cudaDeviceProp ** devicePropertiesArray;
} rudyCUDA_deviceInfo;

/**
  * @brief Gather information for the available devices
  * @return devicePropertiesArray an of points to #rudyCUDA_cudaDeviceProp
	* storeing the properties of the availble devices
  */ 
rudyCUDA_deviceInfo * rudyCUDA_deviceInfo_gather( void);

/**
	* @brief Free the memory of a cudaDeviceProp**
	* @param devicePropertiesArray the cudaDeviceProp** to 
  * be freed from memory and set to NULL
  */
void rudyCUDA_deviceInfo_free( rudyCUDA_deviceInfo * devices);

/**
  * @brief Print all device info
  */
void rudyCUDA_deviceInfo_print(rudyCUDA_deviceInfo * devices);

#endif
