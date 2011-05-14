/**
  * @file rudyCUDA_bitmap.h
  * @author Kurt Robert Rudolph
  * @descrioption Supports image storage, modification and update
	*/

#ifndef __RUDY_CUDA__BITMAP_H
#define __RUDY_CUDA__BITMAP_H

#include <rudyCUDA.h>

/** 
  * @struct bitmap_compression
  * @brief type of compression being used
  */
typedef enum rudyCUDA_bitmap_compression_t{
  rudyCUDA_bitmap_compressione_RGB = 0,
  rudyCUDA_bitmap_compression_RLE8,
  rudyCUDA_bitmap_compression_RLE4,
  rudyCUDA_bitmap_compression_BITFIELDS,
  rudyCUDA_bitmap_compression_JPEG,
  rudyCUDA_bitmap_compression_PNG,
} rudy_CUDA_bitmap_compression;

/**
  * @struct rudyCUDA_bitmap_pixelRGB_t
  * @brief Stores the 256bit RGB valies.
  * @param blue The blue intensity of the pixel.
  * @param green The green intensity of the pixel.
  * @param red The alpha intenity of the pixel.
  * @param alpha The ocupency of the pixel. 
  */
typedef struct rudyCUDA_bitmap_pixelRGB_t {
  uint8_t blue;
  uint8_t green;
  uint8_t red;
  uint8_t alpha;
} rudyCUDA_bitmap_pixelRGB;


/**
  * @struct rudyCUDA_bitmap_header_t
  * @param filesize The size of the bmp file in bytes.
  * @param creator0 
  * @param creator1 
  * @param bimapDataOffset The number of bites fromt he begining of 
  * the file to the starting address of the bitmap data. 
  * @param bitmapType A pair of hex codes representing the type of bitmap.
  */
typedef struct rudyCUDA_bitmap_header_t{
  uint32_t fileSize;
  uint16_t creator1;
  uint16_t creator2;
  uint32_t bitmapDataOffset;
  uint8_t bitmapType[2];
} rudyCUDA_bitmap_header;

/**
  * @struct rudyCUDA_bitmap_headerInfo_t
  * @brief stores the header inforomation included in the header of the bitmap file
  * @param header_size The size of the header file in bytes.  
  * @param pixelWidth The number of pixels spanning the width of the bitmap.  
  * @param pixelHeight The number of pixels spanning the height of the bitmap.
  * @param numberOfColorPlanes The number of color panes being used.
  * @param bitsPerPixel The number of bits per pixel.
	* @param bitmap_compression_t The type of compression employed by the bitmap.
  * @param bitmap_dataSize The size of strictly the raw bitmap data.
  * \emph{NOT} the size of the file containing the data.
  * @param bitmap_horizontalResolution The horizontal resolution of the image.
  * @param bitmap_virticalResolution The virtical resolutino of the image.
  * @param numberOfImportantColors number of important colors used by the bitmap
  */
typedef struct rudyCUDA_bitmap_headerInfo_t {
	uint32_t header_size;
	uint32_t pixelWidth;
	uint32_t pixelHeight;
	uint16_t numberOfColorPlanes;
	uint16_t bitsPerPixel;
	rudyCUDA_bitmap_compression_t compression;
	uint32_t bitmap_dataSize;
	uint32_t bitmap_horizontalResolution;
	uint32_t bitmap_virticalResolution;
	uint32_t numberOfPaletteColors;
	uint32_t numberOfImportantColors;
} rudyCUDA_bitmap_headerInfo;

typedef struct bitmap_t bitmap;

/**
  * @brief Creates a bitmap
  * @param map The #bitmap where the map is to be created.
  * @param width The width of the #bitmap to be created.
  * @param height The height of the #bitmap to be created.
  * @param depth The depth of the #bitmap to be created.
  * @return The state of the function.
  */
rudyCUDA_state rudyCUDA_bitmap_create( bitmap_t ** map, uint32_t *width, uint32_t *height, uint32_t **depth);

/**
  * @brief Loads the data from file into a #bitmap
  * @param fileName
  * @return The state of the function. 
  */
rudyCUDA_state rudyCUDA_bitmap_createFromFile( const char *fileName); 

/**
  * @brief Free's the memory of a #bitmap.
  * @param map The #bitmap to be freeed from memory
  */
rudyCUDA_state rudyCUDA_bitmap_free( bitmap **map);


#endif //__RUDY_CUDA__BITMAP_H
