/**
  * @file rudyCUDA_IO.h
  * @author Kurt Robert Rudolph
  * @description this file will is set of functions for 
  * performing input output.  
  */


#ifndef __RUDY_CUDA__IO_H__
#define __RUDY_CUDA__IO_H__

#include <rudyCUDA.h>

/**
  * @brief Opens a file and store the data within GPU memory
  * @param file The function directs this pointer to the locaiton 
  * of the FILE object opened.  
  * @param fileName The name of the file to be opened.
  */
rudyCUDA_state rudyCUDA_IO_openFile( FILE ** file, char * fileName);

#endif //__RUDY_CUDA__IO_H__
