/* CUJ2K - JPEG2000 Encoder on CUDA
http://cuj2k.sourceforge.net/

Copyright (c) 2009 Norbert Fuerst, Martin Heide, Armin Weiss, Simon Papandreou, Ana Balevic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE. */


/*
Data structures for storing tiled and untiled pictures.

struct Picture
Stores a tiled picture. 'device_mem' is a pointer to the allocated
device memory.

struct Tile
One tile in a picture. imgData_d[channel] is a pointer to start
address of top-left corner in the picture. Only a pointer, but no
seperately allocated memory.

struct Bitmap
un-tiled picture. imgData[channel] is page-locked host memory.
Allocation is done in imgData[0] for all 3 channels. Samples
are 32 bit, so imgData is in fact an int** pointer.
*/
#include <stdint.h>
#include <stdlib.h>

#ifndef BITMAP_H
#define BITMAP_H

/*type of compression*/
#define LOSSLESS 1
#define LOSSY 2
#define REVERSIBLE 1
#define IRREVERSIBLE 2

//TIFF magic
#define BYTE_ORDER_INTEL 0x4949
#define BYTE_ORDER_MOTEL 0x4d4d

#define BYTE_PER_TAG 12

#define DATA_TYPE_POS 2
#define DATA_COUNT_POS 4
#define DATA_VALUE_POS 8 //could be offsets or real value, depends on the datatypes and counts


#define DATA_TYPE_BYTE 1
#define DATA_TYPE_ASCII 2
#define DATA_TYPE_SHORT 3
#define DATA_TYPE_LONG 4
#define DATA_TYPE_RATIONAL 5 //2 longs

#define IFD_OFFSETS_POS 4

#define TIFF_ID 42

#define TAG_WIDTH 256
#define TAG_HEIGHT 257
#define TAG_BITS_PER_SAMPLE 258
#define TAG_SAMPLE_FORMAT 339
#define TAG_STRIP_OFFSETS 273  //Beginning of DATA
#define TAG_ROWS_PER_STRIP 278
#define TAG_STRIP_BYTE_COUNTS 279 //Size of each strip
#define TAG_COMPRESSION 259
#define TAG_COLOR_SPACE 262


//Could be ignored
#define TAG_DESCRIPTION 270
#define TAG_MODEL 272

#define TAG_XRES 282
#define TAG_MIN_SAMPLE_VALUE 280  //statistics only

#define TAG_SOFTWARE 305
#define TAG_DATETIME 306
#define TAG_ARTIST 312

struct simpleTIFF {
	int xDim, yDim, channels;

  short bps; //bits per samples
  short clrSpace;


	int area_alloc; //dimension x*y of allocated imgData

	/* image data; pixel at position (row,column) in color_channel can be
	   accessed by imgData[color_channel][column + row*xDim] after casting
	   to (char**), (short**) etc. */
	void **imgData;
};

int readTiffData(simpleTIFF *img, int dataOffSets, int channels, unsigned char *src);
//end of tiff



struct Tile;
struct Bitmap;

/* container to store original image and tiles*/
struct Picture {
	/* width, height of original bitmap, number of color channels (e.g. 3 for RGB)*/
	int xSize, ySize, channels;

	int area_alloc; //dimension x*y of allocated device_mem

	/*number of tiles*/
	int tile_number, tile_number_alloc;

	/*array of tiles*/
	struct Tile *tiles;
	void *device_mem;

	int cb_xdim_exp, cb_ydim_exp;
	int tilesize;
};

/* struct for storing a bitmap with 8bpp per color channel */
struct Bitmap {
	/* width, height of original bitmap, number of color channels (e.g. 3 for RGB)*/
	int xDim, yDim, channels;

	int area_alloc; //dimension x*y of allocated imgData

	/* image data; pixel at position (row,column) in color_channel can be
	   accessed by imgData[color_channel][column + row*xDim] after casting
	   to (char**), (short**) etc. */
	void **imgData;
};

/* struct for storing tiles along with their position in the original image*/
struct Tile {
	/* width, height of tile, number of color channels (e.g. 3 for RGB)*/
	int xDim, yDim, channels;

	/* position in original image (top left corner of tile) */
	int xPos, yPos;
	char  QS; //quantization style for each channel (ready for QCD/QCC marker)

	//for testing purposes. TODO: Needs to be removed
	void **imgData;

	/* image data on GPU; pixel at position (row,column) in color_channel of tile can be
	   accessed by imgData[color_channel][column + row*xDim].
	   To work on data in Kernel, only pointer adress can be passed, array itself is in host memory */
	void *imgData_d[3];
};



/* reset struct member variables */
void bmpReset(struct Bitmap *img);
void picReset(struct Picture *pic);

/* free allocated memory if imgData!=NULL */
extern "C"
void bmpFree(struct Bitmap *img);

/* free allocated memory if imgData!=NULL */
extern "C"
void tiffFree(struct simpleTIFF *img);
//free function for original image
extern "C"
void tileFree(struct Tile *img);
//free function for tile that contains dwt
void tileFree_dwt(struct Tile *img);

void free_picture(struct Picture *pic);

int tiffRead(const unsigned char* src0, size_t insize, struct simpleTIFF *img, const char *filename);
/* load bitmap from file; returns 0 on success, !=0 else */
int bmpRead(struct Bitmap *img, const char *filename);
/* writes bitmap to file; returns 0 on success, !=0 else */
int bmpWrite(struct Bitmap *img, const char *filename);

void resize_bmp(struct Bitmap *out, struct Bitmap *in, int w, int h);

/*read/write any format by calling ImageMagick's convert
  these functions are not thought for productive use,
  but only for more convenient testing*/
//http://www.imagemagick.org/download/binaries/ImageMagick-6.5.4-3-Q8-windows-dll.exe
//^^Windows version of ImageMagick
int any_img_read(const unsigned char* src, size_t insize, struct Bitmap *bm, const char *filename);
int any_img_write(struct Bitmap *bm, const char *filename);

#endif
