/**
  * @file rudy_typedef.h
	* @author Kurt Robert Rudolph
  * @description This file defines the description 
  * various type which are fairly common but depending
  * the system may not be defined.
  */

#ifndef __RUDY_TYPEDEF_H__
	#define __RUDY_TYPEDEF_H__

/*	#ifndef bool
		typedef int bool;
		#define FALSE (0)
		#define TRUE !FALSE
	#endif //bool */

	#ifndef uint8_t
		typedef unsigned char uint8_t;
	#endif //uint8_t

	#ifndef uint16_t
		typedef unsigned short uint16_t;
	#endif //uint16_t

	#ifndef uint32_t
		typedef unsigned int uint32_t;
	#endif //uint32_t

#endif //__RUDY_TYPEDEF_H__

