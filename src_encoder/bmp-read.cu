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
Windows .BMP-File reader
only for format version 3.0 and 8bpp, 24bpp

int bmpRead(struct Bitmap *img, const char *filename)
Reads .BMP file, converts sample size from 8 bit to 32 bit.
Data is stored in page-locked host memory.

int any_img_read(struct Bitmap *bm, const char *filename)
Reads .BMP files directly. For other extensions, calls
ImageMagick's commandline tool 'convert' to convert the
picture to a temporary BMP file and loads this.

void {bmp,pic}Reset(.....)
resets a Bitmap/Picture struct so that it is clear that
no memory has been alocated.

void free_picture(...), void bmpFree(....)
Frees the device/host memory of a Bitmap/Picture.
*/


#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil.h>
#include "bitmap.h"
#include "waveletTransform.h"
#include "tier1.h"

/* size of header data in .BMP file */
#define BMP_HEADER_SIZE 54
/* magic number identifying a .BMP-file */
#define BMP_MAGIC 19778



void bmpReset(struct Bitmap *img) {
	img->xDim = img->yDim = img->channels = img->area_alloc = 0;
	img->imgData = NULL;
}

void picReset(struct Picture *pic) {
	pic->xSize = pic->ySize = pic->channels = 0;
	pic->tile_number = pic->tile_number_alloc = 0;
	pic->cb_xdim_exp = pic->cb_ydim_exp = pic->tilesize = 0;
	pic->area_alloc = 0;
	pic->tiles = NULL;
	pic->device_mem = NULL;
}


/*Frees the memory of the imgData-pointer*/
void bmpFree(struct Bitmap *img) {
	int ch;

	if(img->imgData != NULL) {
		cudaFreeHost(img->imgData[0]); //ein free reicht, weil nur ein array für alle channels allokiert wurde
		free(img->imgData);
	}
	bmpReset(img);
}

//frees device mem and tile array; this function is currently used!
void free_picture(struct Picture *pic) {
	cutilSafeCall(cudaFree(pic->device_mem));
	pic->device_mem = NULL;

	free(pic->tiles);          
	pic->tiles=NULL;
}

void subb_free(subband *subb) {
	int i;
	if(subb->Typ == LLsubdiv) {
		for(i=0; i<4; i++) {
			subb_free(subb->subbands[i]);
			free(subb->subbands[i]);
			subb->subbands[i] = NULL;
		}
	}
	else {
		if(subb->Typ == LLend)
			cutilSafeCall(cudaFree(subb->daten_d));
		free(subb->K_msbs);
		/*for(i=0; i < subb->nCodeblocks; i++)
			cb_free(&(subb->codeblocks[i]));*/
		subb->daten_d = NULL; subb->K_msbs=NULL; //subb->codeblocks=NULL;
	}
}

void tileFree_dwt(struct Tile *img) {
	int ch;
	subband **subb_p = (subband**) img->imgData; 
	
	if(img->imgData != NULL) {
		for(ch = 0; ch < img->channels; ch++) {
			subb_free(subb_p[ch]);
			free(subb_p[ch]);
			subb_p[ch] = NULL;
		}
		free(img->imgData);
		img->imgData = NULL;
	}	
}

void tileFree(struct Tile *img) {
	int ch;
	if(img->imgData != NULL) {
		for(ch = 0; ch < img->channels; ch++)
			free(img->imgData[ch]);
		free(img->imgData);
		img->imgData = NULL;	
	}
}

//resizes 'in' to size (w,h), writes it to 'out'; also allocates needed memory
void resize_bmp(struct Bitmap *out, struct Bitmap *in, int w, int h) {
	int **out_data = (int**) malloc(3*sizeof(int*));
	cutilSafeCall (cudaMallocHost((void**)out_data,3*w*h*sizeof(int)));
	out_data[1] = &out_data[0][w*h];
	out_data[2] = &out_data[0][2*w*h];
	int **in_data = (int**)in->imgData;

	if(w > in->xDim  ||  h > in->yDim) {
		printf("resize error: size is larger than input pic\n");
		exit(1);
	}

	for(int y = 0; y < h; y++) {
		for(int x = 0; x < w; x++) {
			int out_pos = x + y*w;
			int in_pos = x + y*(in->xDim);
			out_data[0][out_pos] = in_data[0][in_pos];
			out_data[1][out_pos] = in_data[1][in_pos];
			out_data[2][out_pos] = in_data[2][in_pos];
		}
	}
	out->xDim=w;
	out->yDim=h;
	out->imgData = (void**)out_data;
	out->channels=3;
}


int bmpRead(struct Bitmap *img, const char *filename) {
	char *header, *data;
	/* dataOffset = position of image data in file */
	int bytes, error, dataOffset, dataSize;
	int w, h;
	
	/* =1 if image is stored bottom-row first, top-row last */
	int bottomUp = 1;
	
	/* 16-bit value for bpp */
	/* magic number at first 2 bytes of file */
	short bpp, magic;

	//type of compression; only uncompressed bitmaps allowed
	int biCompression, biClrUsed; 

	unsigned char colortable[256][4]; //order: B,G,R,empty
	
	/* if row size in bytes isn't a multiple of 4, then each row is filled
	   with 1 to 3 zero bytes */
	int fillBytes;

	FILE *f;
	
	/* open in read binary mode */
	f = fopen(filename, "rb");
	if(f == NULL) {
		printf("bmpRead: file '%s' not found.\n", filename);
		return(-1);
	}
	
	header = (char*) malloc(BMP_HEADER_SIZE);
	
	/* size_t fread ( void * ptr, size_t size, size_t count, FILE * stream ); */
	
	bytes = (int)fread((void*)header, 1, BMP_HEADER_SIZE, f);
	if(bytes != BMP_HEADER_SIZE) {
		fclose(f); free(header);
		printf("bmpRead: read error in '%s'.\n", filename);
		return(-1);
	}
		
	magic = *((short*) header);
	bpp = *((short*) (header+28));
	biCompression = *((int*)(header+30));
	biClrUsed = *((int*)(header+46));
	dataOffset = *((int*) (header+10));
	w = *((int*) (header+18));
	h = *((int*) (header+22));
	if(h < 0) {
		/* this means that the top-row is stored first */
		h = -h;
		bottomUp = 0;		
	}
	
	free(header); header=NULL;
	
	//printf("magic: %d\n", magic);
	//printf("header: 0:%d 1:%d\n", header[0], header[1]);
	//printf("Windows bitmap, w:%d h:%d bpp:%d data-offset:%d\n", w, h, bpp, dataOffset);
	
	if(magic != BMP_MAGIC) {
		fclose(f);
		printf("bmpRead: '%s' is not a bitmap-file.\n", filename);
		return(-1);
	}
	if(bpp != 24 && bpp != 8) {
		fclose(f);
		printf("bmpRead: only 24 or 8 bpp supported (file '%s').\n", filename);
		return(-1);
	}
	if(biCompression != 0) {
		fclose(f);
		printf("bmpRead: only uncompressed BMPs allowed (file '%s').\n", filename);
		return(-1);
	}

	if(bpp == 8) {
		//read color table
		if(biClrUsed==0)
			biClrUsed = 256;
		if(biClrUsed < 0  || biClrUsed > 256) {
			printf("bmpRead: Color table error in '%s'.\n", filename);
			fclose(f);
			return(-1);
		}

		bytes = (int)fread((void*)colortable, 1, biClrUsed*4, f);
		if(bytes != biClrUsed*4) {
			printf("bmpRead: read error in file '%s'.\n", filename);
			fclose(f);
			return(-1);
		}
	}

	if(bpp == 24) {
		/* 3*w = number of bytes per row */
		fillBytes = 4 - ((3*w) % 4);
		/* 4 fillBytes are of course 0 fillbytes => modulo 4 */
		fillBytes %= 4; 
		/* calculate size of image date in .BMP-file, inclusive fill-bytes */
		dataSize = (3*w + fillBytes) * h;
	}
	else { // 8 bpp
		fillBytes = 4 - (w % 4);	
		/* 4 fillBytes are of course 0 fillbytes => modulo 4 */
		fillBytes %= 4; 
		/* calculate size of image date in .BMP-file, inclusive fill-bytes */
		dataSize = (w + fillBytes) * h;
	}
	data =(char*) malloc(dataSize);


	//printf("Starting to read image data from file\n");
	
	/* int fseek ( FILE * stream, long int offset, int origin ); 
	   SEEK_SET = from beginning of file */
	error = fseek(f, dataOffset, SEEK_SET);
	bytes = (int)fread((void*)data, 1, dataSize, f);
	fclose(f);
	if((bytes != dataSize) || (error != 0)) {
		free(data);
		printf("bmpRead: read error in file '%s'.\n", filename);
		return(-1);
	}
	
	//printf("Starting to store image data in array\n");
	
	{
		int ch, row, col;
		int **intData;
		unsigned char *data_i = (unsigned char*) data;
	
		//only one array for the 3 channels (to make memcpy easier)
		if(img->imgData == NULL) {
			//printf("bmpRead: completely new mem\n");
			intData = (int**) malloc(3*sizeof(int*));
			cutilSafeCall(cudaMallocHost((void**)intData,3*w*h*sizeof(int)));
			img->area_alloc = w*h;
		}	
		else {
			//printf("bmpRead: reuse\n");
			intData = (int**) (img->imgData);
			if(w*h > img->area_alloc) {
				//printf("bmpRead: reuse must free\n");
				cutilSafeCall(cudaFreeHost(intData[0]));
				//printf("bmpRead: reuse must malloc\n");
				cutilSafeCall(cudaMallocHost((void**)intData,3*w*h*sizeof(int)));
				img->area_alloc = w*h;
			}
		}
		intData[1] = &intData[0][w*h];
		intData[2] = &intData[0][2*w*h];
		/*intData = (int**) malloc(3*sizeof(int*));
		cudaMallocHost((void**)intData,3*w*h*sizeof(int));*/
		if(bottomUp)
			row = h-1;
		else
			row = 0;
		
		if(bpp == 24) {
			while(row>=0 && row<h) {
				for(col = 0; col < w; col++) {
					/* image is stored in BGR order */
					for(ch = 2; ch >= 0; ch--)
						intData[ch][col + row*w] = (int)(*(data_i++));
				}
				/* at the end of the row: skip fillbytes */
				data_i += fillBytes;
				
				if(bottomUp) row--;
				else         row++;
			}
		}
		else { //8 bpp
			while(row>=0 && row<h) {
				for(col = 0; col < w; col++) {
					/*// grey image: use same value for each B,G,R
					for(ch = 2; ch >= 0; ch--)
						intData[ch][col + row*w] = (int)(*data_i);*/
					//lookup in color table
					for(ch = 2; ch >= 0; ch--)
						intData[ch][col + row*w] = (int)(colortable[(*data_i)][ch]);
					data_i++;
				}
				/* at the end of the row: skip fillbytes */
				data_i += fillBytes;
				
				if(bottomUp) row--;
				else         row++;
			}
		}
		
		img->imgData = (void**)intData;
	}
	
	//printf("stored in array. Freeing memory...\n");	
	
	free(data); data=NULL;
	img->xDim = w;
	img->yDim = h;
	img->channels = 3;
	//printf("Bitmap reading complete.\n");
	
	return 0;
}


// :( BMP uses different byte order compared to JPEG2000
// => use other functions
static int write_int2(FILE *fp, unsigned int value) {
	fwrite((void*)&value, sizeof(value), 1, fp);
	return 0;
}
static int write_short2(FILE *fp, unsigned short value) {
	fwrite((void*)&value, sizeof(value), 1, fp);
	return 0;
}

int bmpWrite(struct Bitmap *img, const char *filename) {
	int **data = (int**)img->imgData;
	int x,y,ch, w=img->xDim, h=img->yDim, count;
	int row_size = (3*img->xDim + 3) / 4 * 4; //fill to 4 byte-bounds
	FILE *fp = fopen(filename, "wb");
	if(fp == NULL) {
		printf("bmpWrite: Could not open output file '%s'\n", filename);
		return(-1);
	}

	//file header
	write_short2(fp, BMP_MAGIC);
	write_int2(fp, 54 + row_size*img->yDim); //file size
	write_int2(fp, 0);
	write_int2(fp, 54); //data offset

	//image info
	write_int2(fp, 40);
	write_int2(fp, img->xDim);
	write_int2(fp, img->yDim);
	write_short2(fp, 1);
	write_short2(fp, 24); //bpp
	write_int2(fp, 0); //uncompressed

	write_int2(fp, 0); //image data size, but may write 0
	write_int2(fp, 0);
	write_int2(fp, 0);
	write_int2(fp, 0); //color table
	write_int2(fp, 0);
	//header complete

	//write image data from bottom to top row
	for(y = h-1; y >= 0; y--) {
		for(x = 0; x < w; x++) {
			for(ch = 2; ch >= 0; ch--)
				fputc(data[ch][y*w + x], fp);
		}
		//fill up to 4-byte alignment
		count = 3*w;
		while((count%4) != 0) {
			count++;
			fputc(0, fp);
		}
	}
	fclose(fp);
	return 0;
}


/*read/write any format by calling convert
  these functions are not thought for productive use,
  but only for more convenient testing.
  .BMP output works as before without any additional program*/

//Path where to find ImageMagick's convert(.exe)
#ifdef UNIX
static const char CONVERT_PATH[] = "convert";
#else
//adjust to your needs
static const char CONVERT_PATH[] = "..\\ImageMagick\\convert";
#endif


// returns true if 'filename' ends with ".bmp" (ignores case)
static int is_bmp(const char *filename) {
	const char BMP_EXT[] = ".bmp";
	int i, j;
	if(strlen(filename) < 4)
		return 0;
	i = (int)strlen(filename) - 4;
	j = 0;
	while(j < 4) {
		if(tolower(filename[i]) != BMP_EXT[j])
			return 0;
		i++; j++;
	}
	return 1;
}

int any_img_read(struct Bitmap *bm, const char *filename) {
	if(is_bmp(filename))
		return bmpRead(bm, filename);
	else {
		int ret;
		char cmdline[2000 + L_tmpnam];
#ifdef UNIX
		char temp_file[L_tmpnam];
		tmpnam(temp_file); //get temp filename
#else
		//VC++'s tmpnam() does not use proper TEMP folder, so use _tempnam instead
		char *temp_file;
		temp_file = _tempnam(NULL, "convert"); //returns filename in TEMP folder
#endif		
		//printf("temp file is: '%s'\n", temp_file);

		//Create Windows bitmap 3.0 with 24bpp
		sprintf(cmdline, "%s \"%s\" -type truecolor \"BMP3:%s\"", CONVERT_PATH, filename, temp_file);
		//printf("cmdline = '%s'\n", cmdline);
		ret = system(cmdline);
		//system("pause");
		if(ret != 0) {
			printf("Could not convert '%s' to temp BMP '%s' !\n", filename, temp_file);
			printf("convert returned error %d\n", ret);
			printf("cmdline was: '%s'\n", cmdline);
			free(temp_file);
			return ret;
		}
		ret = bmpRead(bm, temp_file);
		remove(temp_file); //clean up temp file
#ifndef UNIX		
		free(temp_file); // _tempnam allocates memory for filename
#endif
		return ret;
	}
}


int any_img_write(struct Bitmap *bm, const char *filename) {
	if(is_bmp(filename))
		return bmpWrite(bm, filename);
	else {
		char cmdline[2000 + L_tmpnam], temp_file[L_tmpnam];
		int ret;
		tmpnam(temp_file); //get temp filename
		//printf("Saving to temp BMP '%s'...\n", temp_file);
		ret = bmpWrite(bm, temp_file);
		if(ret != 0) {
			//printf("Could not write temp .bmp file!\n");
			return ret;
		}

		sprintf(cmdline, "%s \"BMP:%s\" \"%s\"", CONVERT_PATH, temp_file, filename);
		//printf("Converting format, output is '%s'...\n", filename);
		//printf("cmdline = '%s'\n", cmdline);
		ret = system(cmdline);
		//system("pause");
		remove(temp_file); //clean up temp file
		if(ret != 0) {
			printf("Could not convert temp BMP '%s' to output '%s' !\n", temp_file, filename);
			printf("convert returned error %d\n", ret);
			printf("cmdline was: '%s'\n", cmdline);
			return ret;
		}
		return ret;
	}
}
