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

/**
  * @brief gather information for the available devices
  * @return devicePropertiesArray an of points to #rudy_cudaDeviceProp
	* storeing the properties of the availble devices
  */ 
cudaDeviceProp ** rudy_deviceInfo_gather(void);

#endif
