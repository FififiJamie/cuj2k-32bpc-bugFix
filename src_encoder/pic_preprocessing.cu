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

/* Picture preprocessing operations for JPEG2000 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pic_preprocessing.h"
//#include <cutil.h>
//#include <cutil_inline.h>
#include <helper_cuda.h>
#include <helper_timer.h>

//sets pointers, but doesn't copy any data
//TODO: Free pic->original_d on device at the end of processing
void tiling (struct Picture *pic, struct Bitmap *img, int cb_dim) {
	//struct Picture *pic = NULL;
	struct Tile *tile = NULL;
	int tilesize,size, chan;
	int tile_numberx, tile_numbery,tile_number_total, tileIdx;
	int xpos, ypos, xstep, ystep;
	int xDim =img->xDim, yDim =img->yDim;

	if(cb_dim == 0) {
		//set size automatically
		//set codeblock size and according tilesize
		// => gain more speed with small pictures
		if(xDim <= 256  &&  yDim <= 256) {  //700^2, tiles 256
			pic->cb_xdim_exp = 4; //16x16
		}
		else if(xDim <= 512  &&  yDim <= 512) {  //1500^2, tiles 512
			pic->cb_xdim_exp = 5; //32x32
		}
		else {
			pic->cb_xdim_exp = 6; //64x64
		}
		tilesize = 1024;
	}
	else {
		//user-defined codeblock size
		switch(cb_dim) {
			case 16:
				pic->cb_xdim_exp=4;
				tilesize = 256;
				break;
			case 32:
				pic->cb_xdim_exp=5;
				tilesize = 512;
				break;
			default: //64
				pic->cb_xdim_exp=6;
				tilesize = 1024;
		}
	}
	pic->cb_ydim_exp = pic->cb_xdim_exp;
	pic->tilesize = tilesize;

	xstep = tilesize;
	ystep = tilesize;

	/*Calculation of tile number to allocate memory*/
	tile_numberx = (xDim + (tilesize - 1))/tilesize;
	tile_numbery = (yDim + (tilesize - 1))/tilesize;
	tile_number_total = tile_numberx * tile_numbery;

	if(tile_number_total > pic->tile_number_alloc) {
		//printf("alloc tiles\n");
		free(pic->tiles);
		tile = (struct Tile*) malloc (tile_number_total * sizeof (struct Tile));
		pic->tiles = tile;
		pic->tile_number_alloc = tile_number_total;
	}
	else
		tile = pic->tiles;

	int *imgData_d[3];

	size = 3* xDim*yDim*sizeof(int);
	if(xDim*yDim > pic->area_alloc) {
		//printf("tiling: free old dev.mem\n");
		checkCudaErrors(cudaFree(pic->device_mem));
		//printf("tiling: malloc dev.mem\n");
		checkCudaErrors(cudaMalloc((void **) &imgData_d[0], size));
		pic->device_mem = (void*)imgData_d[0];
		pic->area_alloc = xDim*yDim;
	}
	else
		imgData_d[0] = (int*)(pic->device_mem);

	imgData_d[1]= &(imgData_d[0][xDim*yDim]);
	imgData_d[2]= &(imgData_d[1][xDim*yDim]);

	pic->xSize = img->xDim;
	pic->ySize = img->yDim;
	pic->channels = img->channels;

	//used to malloc memory on device for original data
	//Transfering data from original image to container
	//pic = (struct Picture*) malloc (sizeof (struct Picture));

	//checkCudaErrors(cudaMalloc((void **) &imgData_d[0], size));

	//work variables
	xpos=0;
	ypos=0;
	tileIdx = 0;

	/*move (xpos/ypos) from tile to tile  (always points at top left corner)*/
	for (ypos=0; ypos<yDim; ypos +=ystep){
		/*step can not go over image boundaries*/
		if ((ypos+tilesize) > yDim)
			ystep = yDim - ypos;
		else ystep = tilesize;

		for (xpos=0; xpos<xDim; xpos +=xstep){
			/*step can not go over image boundaries*/
			if ((xpos+tilesize) > xDim)
				xstep = xDim - xpos;
			else xstep = tilesize;

			//transfering all the tile data to current tile
			tile[tileIdx].xDim = xstep;
			tile[tileIdx].yDim = ystep;
			tile[tileIdx].xPos = xpos;
			tile[tileIdx].yPos = ypos;
			tile[tileIdx].channels = img->channels;
            tile[tileIdx].QS = 0x22; //standard set to irreversible
			//setting pointer to original data array
			for (chan=0;chan < pic->channels;chan++){
				tile[tileIdx].imgData_d[chan] = &(imgData_d[chan][xpos+ypos*xDim]);
			}

			pic->tile_number = tileIdx + 1; //min 1 tile
			tileIdx++;
		}
	}
	//return pic;
}

void tiling2 (struct Picture *pic, struct simpleTIFF *img, int cb_dim) {
	//struct Picture *pic = NULL;
	struct Tile *tile = NULL;
	int tilesize,size, chan;
	int tile_numberx, tile_numbery,tile_number_total, tileIdx;
	int xpos, ypos, xstep, ystep;
	int xDim =img->xDim, yDim =img->yDim;

	if(cb_dim == 0) {
		//set size automatically
		//set codeblock size and according tilesize
		// => gain more speed with small pictures
		if(xDim <= 256  &&  yDim <= 256) {  //700^2, tiles 256
			pic->cb_xdim_exp = 4; //16x16
		}
		else if(xDim <= 512  &&  yDim <= 512) {  //1500^2, tiles 512
			pic->cb_xdim_exp = 5; //32x32
		}
		else {
			pic->cb_xdim_exp = 6; //64x64
		}
		tilesize = 1024;
	}
	else {
		//user-defined codeblock size
		switch(cb_dim) {
			case 16:
				pic->cb_xdim_exp=4;
				tilesize = 256;
				break;
			case 32:
				pic->cb_xdim_exp=5;
				tilesize = 512;
				break;
			default: //64
				pic->cb_xdim_exp=6;
				tilesize = 1024;
		}
	}
	pic->cb_ydim_exp = pic->cb_xdim_exp;
	pic->tilesize = tilesize;

	xstep = tilesize;
	ystep = tilesize;

	/*Calculation of tile number to allocate memory*/
	tile_numberx = (xDim + (tilesize - 1))/tilesize;
	tile_numbery = (yDim + (tilesize - 1))/tilesize;
	tile_number_total = tile_numberx * tile_numbery;

	if(tile_number_total > pic->tile_number_alloc) {
		//printf("alloc tiles\n");
		free(pic->tiles);
		tile = (struct Tile*) malloc (tile_number_total * sizeof (struct Tile));
		pic->tiles = tile;
		pic->tile_number_alloc = tile_number_total;
	}
	else
		tile = pic->tiles;

	int *imgData_d[3];

	size = 3* xDim*yDim*sizeof(int);
	if(xDim*yDim > pic->area_alloc) {
		//printf("tiling: free old dev.mem\n");
		checkCudaErrors(cudaFree(pic->device_mem));
		//printf("tiling: malloc dev.mem\n");
		checkCudaErrors(cudaMalloc((void **) &imgData_d[0], size));
		pic->device_mem = (void*)imgData_d[0];
		pic->area_alloc = xDim*yDim;
	}
	else
		imgData_d[0] = (int*)(pic->device_mem);

	imgData_d[1]= &(imgData_d[0][xDim*yDim]);
	imgData_d[2]= &(imgData_d[1][xDim*yDim]);

	pic->xSize = img->xDim;
	pic->ySize = img->yDim;
	pic->channels = img->channels;

	//used to malloc memory on device for original data
	//Transfering data from original image to container
	//pic = (struct Picture*) malloc (sizeof (struct Picture));

	//checkCudaErrors(cudaMalloc((void **) &imgData_d[0], size));

	//work variables
	xpos=0;
	ypos=0;
	tileIdx = 0;

	/*move (xpos/ypos) from tile to tile  (always points at top left corner)*/
	for (ypos=0; ypos<yDim; ypos +=ystep){
		/*step can not go over image boundaries*/
		if ((ypos+tilesize) > yDim)
			ystep = yDim - ypos;
		else ystep = tilesize;

		for (xpos=0; xpos<xDim; xpos +=xstep){
			/*step can not go over image boundaries*/
			if ((xpos+tilesize) > xDim)
				xstep = xDim - xpos;
			else xstep = tilesize;

			//transfering all the tile data to current tile
			tile[tileIdx].xDim = xstep;
			tile[tileIdx].yDim = ystep;
			tile[tileIdx].xPos = xpos;
			tile[tileIdx].yPos = ypos;
			tile[tileIdx].channels = img->channels;
            tile[tileIdx].QS = 0x22; //standard set to irreversible
			//setting pointer to original data array
			for (chan=0;chan < pic->channels;chan++){
				tile[tileIdx].imgData_d[chan] = &(imgData_d[chan][xpos+ypos*xDim]);
			}

			pic->tile_number = tileIdx + 1; //min 1 tile
			tileIdx++;
		}
	}
	//return pic;
}



#define MAX_BLOCKS 65000
//Kernel for dcshift + reversible color transform
__global__ void rct_kernel(int *imgData_0,int *imgData_1,int *imgData_2, int range, int shift, int tile)
{
	int n = threadIdx.x + blockIdx.x*256 + tile*MAX_BLOCKS*256; //proceeding linewise
	if (n<range){ //more threads than pixels, therefore check if in range
		int Y,U,V;

		//DC-Shift
		imgData_0[n]= (int)((int)imgData_0[n] - shift);
		imgData_1[n]= (int)((int)imgData_1[n] - shift);
		imgData_2[n]= (int)((int)imgData_2[n] - shift);

		/*RCT:
		   R=imgData_i_0[n]
		   G=imgData_i_1[n]
		   B=imgData_i_2[n] */
		Y = (imgData_0[n] + 2*imgData_1[n] + imgData_2[n])>>2; //little tweak, instead of floor(../4)
		U = imgData_2[n] - imgData_1[n];
		V = imgData_0[n] - imgData_1[n];

		imgData_0[n] = Y;
		imgData_1[n] = U;
		imgData_2[n] = V;
	}
}


//Kernel for dcshift + irreversible color transform
__global__ void ict_kernel(int *imgData_0,int *imgData_1,int *imgData_2, int range, int tile)
{
	int n = threadIdx.x + blockIdx.x*256 + tile*MAX_BLOCKS*256; //proceeding linewise
	if (n<range){ //more threads than pixels, therefore check if in range
		float Y,C_r,C_b;
		float *imgData_f_0;
		float *imgData_f_1;
		float *imgData_f_2;

		//DC-Shift
		imgData_0[n]= (int) ((int)imgData_0[n] - 128);
		imgData_1[n]= (int) ((int)imgData_1[n] - 128);
		imgData_2[n]= (int) ((int)imgData_2[n] - 128);

		imgData_f_0 = (float*)imgData_0;
		imgData_f_1 = (float*)imgData_1;
		imgData_f_2 = (float*)imgData_2;

		/*ICT:
		   R=imgData_f_0[n]
		   G=imgData_f_1[n]
		   B=imgData_f_2[n] */
		Y = (0.299f*(float)imgData_0[n] + 0.587f*(float)imgData_1[n] + 0.114f*(float)imgData_2[n]);
		C_r = (-0.16875f*(float)imgData_0[n] - 0.33126f*(float)imgData_1[n] + 0.5f*(float)imgData_2[n]);
		C_b = (0.5f*(float)imgData_0[n] + (-0.41869f*(float)imgData_1[n]) - 0.08131f*(float)imgData_2[n]);

		imgData_f_0[n] = Y;
		imgData_f_1[n] = C_r;
		imgData_f_2[n] = C_b;
	}
}


void dcshift_mct (struct Picture *pic, int mode, int bps, cudaStream_t stream){
	int pixels = pic->ySize * pic->xSize;
	int rangecheck = pic->ySize * pic->xSize;
	int blockmultiple;
	int gridDim;
	int processed_per_kernel;
	//pointers for original data
	int *imgData_0 = (int*) pic->tiles[0].imgData_d[0];
	int *imgData_1 = (int*) pic->tiles[0].imgData_d[1];
	int *imgData_2 = (int*) pic->tiles[0].imgData_d[2];

	int shift = (int)(pow(2.0, (float)bps));
	shift = shift /2;

	for(int tileIdx=0;pixels > 0;tileIdx++){
		blockmultiple = (int) ceil((double)pixels/(double)(256*MAX_BLOCKS));
		if (blockmultiple > 1)
			gridDim = MAX_BLOCKS;
		else
			gridDim=(int) ceil((double)pixels/256);

		processed_per_kernel = gridDim*256;
		//processed sequentially
		//kernel dimensions
        dim3 dimGrid(gridDim);
		dim3 dimBlock(256);  //256 Threads for best gpu occupancy, compare cuda occupancy calculator

        //kernel calls
		if (mode == LOSSLESS)
			rct_kernel<<< dimGrid, dimBlock,0, stream >>>(imgData_0,imgData_1,imgData_2,rangecheck,shift,tileIdx);
		else
			ict_kernel<<< dimGrid, dimBlock,0, stream >>>(imgData_0,imgData_1,imgData_2,rangecheck,tileIdx);

		pixels -= processed_per_kernel;
	}
}
