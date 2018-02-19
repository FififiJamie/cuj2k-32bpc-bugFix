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
Tier 1 main file; includes mq-coder.cu, bpc_kernel.cu, bpc_luts.cu and pcrd.cu

void tier1_pre_and_kernel(......)
organize input codeblocks on host; copy them to device;
call kernel (can be several calls). Also copies buffers back
to host after each kernel if local memory is used. If no local
memory is used, this function is asynchronous.

void tier1_post(......)
performs PCRD size optimization if it is enabled (by calling
pcrd_opt in pcrd.cu). Synchronous.

void tier1_memcpy_async(struct Tier1_Pic *t1pic, cudaStream_t stream, int use_local_mem)
copies codeblocks back to host; if no local memory was used, also
copies output buffers back to host. Asynchronous.

struct Raw_CB
input codeblock, pointer v points to top-left position in subband; 
stores codeblock width and height; also stores information about 
the color channel and the subband (for PCRD)

void tier1_pic_reset(struct Tier1_Pic *t1pic)
Resets pointers and size quantities to signal that no memory has
been allocated yet

void tier1_capacity(struct Tier1_Pic *t1pic, int n_cbs, int bufsize_d, int bufsize_h, int use_local_mem)
Allocate memory if sizes are bigger than last picture; otherwise, re-use old memory

void tier1_free(struct Tier1_Pic *t1pic)
Free all memory allocated for Tier 1
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <cutil.h>
#include <cutil_inline.h>
#include "tier1.h"
#include "rate-control.h"
#include "waveletTransform.h"
#include "misc.h"


//#define PRINT_CX_D
//#define PRINT_COEFF

//for BPC
//#define THREADS_PER_BLOCK      128

//size of temporary arrays which moved from local to global memory
#define PCRD_TEMP_ARRAY_SIZE (MAX_PASSES+1)

typedef unsigned sign_magn_t;


struct Raw_CB {
	// coefficients: bit 31: sign, =1 if negative; bits 30..0: magnitude
	sign_magn_t *v;
	int w, h;

	int color_channel, dwt_level;
	SubbandTyp subbType;
	float subb_quantstep;
};

extern __shared__ char s_data[];

#include "pcrd.cu"
#include "mq-coder.cu"
#include "bpc_luts.cu"
#include "bpc_kernel.cu"

//set everything to NULL so that we can safely call cudaFree()
void tier1_pic_reset(struct Tier1_Pic *t1pic) {
	t1pic->n_cbs = t1pic->n_cbs_alloc = -1;
	t1pic->global_buf_alloc_d = t1pic->global_buf_alloc_h = -1;
	//t1pic->state_mem_alloc = -1;
	t1pic->global_states_d = NULL;

	//t1pic->global_trunc_len_d = NULL;
	t1pic->global_dist_d = NULL;
	t1pic->global_saved_states_d = NULL;

	t1pic->cbs_d = t1pic->cbs_h = NULL;
	t1pic->rawcbs_d = t1pic->rawcbs_h = NULL;
	t1pic->slope_max_d = t1pic->slope_max_h = NULL;
	t1pic->global_buf_d = t1pic->global_buf_h = NULL;

	t1pic->pcrd_sizes_alloc = -1;
	t1pic->pcrd_sizes_h = t1pic->pcrd_sizes_d = NULL;
}

void tier1_capacity(struct Tier1_Pic *t1pic, int n_cbs, int bufsize_d, int bufsize_h, int use_local_mem) {
	if(n_cbs > t1pic->n_cbs_alloc) {

		//printf("malloc %d codeblocks\n", n_cbs);

		t1pic->n_cbs = t1pic->n_cbs_alloc = n_cbs;

		cutilSafeCall(cudaFreeHost(t1pic->rawcbs_h));
		cutilSafeCall(cudaFreeHost(t1pic->cbs_h));
		cutilSafeCall(cudaFree(t1pic->rawcbs_d));
		cutilSafeCall(cudaFree(t1pic->cbs_d));

		if(!use_local_mem) {
			cutilSafeCall(cudaFree(t1pic->global_states_d));
			cutilSafeCall(cudaFree(t1pic->global_dist_d));
			cutilSafeCall(cudaFree(t1pic->global_saved_states_d));

			cutilSafeCall(cudaMalloc((void**)&t1pic->global_states_d,    MAX_STATE_ARRAY_SIZE * n_cbs * sizeof(unsigned)));
			cutilSafeCall(cudaMalloc((void**)&t1pic->global_dist_d,         sizeof(float)*PCRD_TEMP_ARRAY_SIZE*n_cbs));
			cutilSafeCall(cudaMalloc((void**)&t1pic->global_saved_states_d, sizeof(uint4)*PCRD_TEMP_ARRAY_SIZE*n_cbs));
		}

		cutilSafeCall(cudaMallocHost((void**)&t1pic->rawcbs_h, sizeof(struct Raw_CB) * n_cbs));
		cutilSafeCall(cudaMallocHost((void**)&t1pic->cbs_h, sizeof(struct Codeblock) * n_cbs));
		cutilSafeCall(cudaMalloc((void**)&t1pic->rawcbs_d, sizeof(struct Raw_CB)     * n_cbs));
		cutilSafeCall(cudaMalloc((void**)&t1pic->cbs_d,    sizeof(struct Codeblock)  * n_cbs));
	}
	else
		t1pic->n_cbs = n_cbs;

	if(bufsize_d > t1pic->global_buf_alloc_d) {
		//printf("malloc device buffers\n");
		cutilSafeCall(cudaFree(t1pic->global_buf_d));
		cutilSafeCall(cudaMalloc((void**)&t1pic->global_buf_d, bufsize_d));
		t1pic->global_buf_alloc_d = bufsize_d;
	}
	if(bufsize_h > t1pic->global_buf_alloc_h) {
		//printf("malloc host buffers\n");
		cutilSafeCall(cudaFreeHost(t1pic->global_buf_h));
		cutilSafeCall(cudaMallocHost((void**)&t1pic->global_buf_h, bufsize_h));
		t1pic->global_buf_alloc_h = bufsize_h;
	}


	if(t1pic->slope_max_d == NULL) {
		//printf("malloc slope_max\n");

		cutilSafeCall(cudaMalloc((void**)&t1pic->slope_max_d, sizeof(float)));
		cutilSafeCall(cudaMallocHost((void**)&t1pic->slope_max_h, sizeof(float)));
	}
}

void tier1_free(struct Tier1_Pic *t1pic) {
	cutilSafeCall(cudaFreeHost(t1pic->rawcbs_h)); t1pic->rawcbs_h=NULL;
	cutilSafeCall(cudaFreeHost(t1pic->cbs_h));    t1pic->cbs_h = NULL;
	cutilSafeCall(cudaFree(t1pic->rawcbs_d));     t1pic->rawcbs_d=NULL;
	cutilSafeCall(cudaFree(t1pic->cbs_d));        t1pic->cbs_d=NULL;

	//cutilSafeCall(cudaFree(t1pic->global_trunc_len_d));	t1pic->global_trunc_len_d=NULL;
	cutilSafeCall(cudaFree(t1pic->global_dist_d));			t1pic->global_dist_d=NULL;
	cutilSafeCall(cudaFree(t1pic->global_saved_states_d)); t1pic->global_saved_states_d=NULL;

	cutilSafeCall(cudaFree(t1pic->global_states_d));  t1pic->global_states_d=NULL;

	cutilSafeCall(cudaFreeHost(t1pic->global_buf_h)); t1pic->global_buf_h=NULL;
	cutilSafeCall(cudaFree(t1pic->global_buf_d));     t1pic->global_buf_d=NULL;

	cutilSafeCall(cudaFree(t1pic->slope_max_d));      t1pic->slope_max_d=NULL; 
	cutilSafeCall(cudaFreeHost(t1pic->slope_max_h));  t1pic->slope_max_h=NULL;

	cutilSafeCall(cudaFreeHost(t1pic->pcrd_sizes_h)); t1pic->pcrd_sizes_h=NULL;
	cutilSafeCall(cudaFree(t1pic->pcrd_sizes_d));     t1pic->pcrd_sizes_d=NULL;
}


//recursively counts all codeblocks in a subband
int count_cb_subb(subband *subb, int cb_xdim, int cb_ydim) {
	if(subb->Typ == LLsubdiv) { /* no data, just subdivision */
		int i, sum=0;
		for(i = 0; i < 4; i++)
			sum += count_cb_subb(subb->subbands[i], cb_xdim, cb_ydim); /* recursion */
		return sum;
	}
	else {
		int n_x = (subb->Xdim + cb_xdim - 1) / cb_xdim;
		int n_y = (subb->Ydim + cb_ydim - 1) / cb_ydim;
		subb->nCBx = n_x;
		subb->nCBy = n_y;
		subb->nCodeblocks = n_x*n_y;
		return(n_x*n_y);
	}
}

int count_cb_pic(struct Picture *pic, int cb_xdim, int cb_ydim) {
	int tile_i, ch_i, sum=0;
	for(tile_i = 0; tile_i < pic->tile_number; tile_i++) {
		struct Tile *t = &(pic->tiles[tile_i]);
		for(ch_i = 0; ch_i < pic->channels; ch_i++)
			sum += count_cb_subb(((subband**) t->imgData) [ch_i], cb_xdim, cb_ydim);
	}
	return sum;
}


// ****************** copying Raw_CBs from Picture to plain array
void copy_subb_to_rawcbs(subband *subb, 
						 int dwt_level, 
						 struct Raw_CB *rawcbs, 
						 struct Codeblock *cbs, 
						 int *cb_i,
						 int bmp_width,
						 int cb_xdim, int cb_ydim, int channel)
{
	if(subb->Typ == LLsubdiv) { /* no data, just subdivision */
		int i;
		for(i = 0; i < 4; i++)
			copy_subb_to_rawcbs(subb->subbands[i], dwt_level+1, rawcbs, cbs, cb_i, bmp_width,
				cb_xdim, cb_ydim, channel); /* recursion */
	}
	else {
		//printf("dwt lvl %d              - type %d\n", dwt_level, (int)(subb->Typ));
		/* dimensions of each code block (last ones can be smaller) */
		int w, h, leftX, topY;

		assert(dwt_level >= 0);

		//use big codeblock array, start at correct index; needn't allocate memory
		subb->codeblocks = &(cbs[*cb_i]);

		//only allocate here; copy data after GPU computation
		//subb->codeblocks = (struct Codeblock*) malloc(sizeof(struct Codeblock) * subb->nCodeblocks);

		dbgTier1b("Splitting subband (type %d) of Xdim=%d Ydim=%d into %d codeblocks\n",
			(int)subb->Typ, subb->Xdim, subb->Ydim, subb->nCodeblocks);

		//DEBUG: copy subband data from host to device ******************
		//cutilSafeCall(cudaMalloc((void**)&(subb->daten_d), sizeof(int) * subb->Xdim * subb->Ydim));
		//cutilSafeCall(cudaMemcpy(subb->daten_d, subb->daten, sizeof(int) * subb->Xdim * subb->Ydim, cudaMemcpyHostToDevice));
		//TODO: remove^^ *************************************************

		/* partitioning into codeblocks */
		h = cb_ydim;
		for(topY = 0; topY < subb->Ydim; topY += cb_ydim) {
			/* bottommost codeblocks can be smaller */
			if(topY + cb_ydim > subb->Ydim)
				h = subb->Ydim - topY;

			w = cb_xdim;
			for(leftX = 0; leftX < subb->Xdim; leftX += cb_xdim) {
				/* rightmost codeblocks can be smaller */
				if(leftX + cb_xdim > subb->Xdim)
					w = subb->Xdim - leftX;

				rawcbs[*cb_i].w = w;
				rawcbs[*cb_i].h = h;
				rawcbs[*cb_i].dwt_level = dwt_level;
				rawcbs[*cb_i].subbType = subb->Typ;
				rawcbs[*cb_i].subb_quantstep = subb->fl_quantstep;
				rawcbs[*cb_i].color_channel = channel;
				//rawcbs[*cb_i].scanw = subb->Xdim;
				//printf("daten_d=%X\n", subb->daten_d);
				rawcbs[*cb_i].v = (sign_magn_t*) &(subb->daten_d[leftX + topY*bmp_width]);
				dbgTier1b("\nCodeblock %d (size %dx%d)\n", *cb_i, w, h);

				//conv_sign_magn(&(rawcbs[*cb_i]), &(subb->daten[leftX + topY*(subb->Xdim)]), subb);
				(*cb_i)++;
			}
		}
	}
}

void copy_pic_to_rawcbs(struct Picture *pic, 
						struct Raw_CB *rawcbs, 
						struct Codeblock *cbs,
						int cb_xdim, int cb_ydim) 
{
	int tile_i, ch_i, cb_i=0;
	for(tile_i = 0; tile_i < pic->tile_number; tile_i++) {
		struct Tile *t = &(pic->tiles[tile_i]);
		for(ch_i = 0; ch_i < pic->channels; ch_i++) {
			/*if(N_DWT_LEVELS == 0)
				copy_subb_to_rawcbs(((subband**) t->imgData) [ch_i], 0, rawcbs, &cb_i);
			else*/
				copy_subb_to_rawcbs(((subband**) t->imgData) [ch_i], -1, rawcbs, cbs, &cb_i, pic->xSize,
					cb_xdim, cb_ydim, ch_i);
		}
	}
	//printf("copied %d rawcbs to plain array\n", cb_i);
}

//********************** copying Codeblocks from plain array to Picture

void print_cb(struct Codeblock *cb) {
	int i;
	for(i=0; i < cb->L; i++) {
		printf("%02X,", cb->B_h[i]);
	}
}

//int max_mq_len = 0;

//TODO: move to Tier 2, so don't need this here**************************
void copy_cbs_to_subb(subband *subb, struct Codeblock *cbs, int *cb_i, int *max_mq_len) {
	if(subb->Typ == LLsubdiv) { /* no data, just subdivision */
		int i;
		for(i = 0; i < 4; i++)
			copy_cbs_to_subb(subb->subbands[i], cbs, cb_i, max_mq_len); /* recursion */
	}
	else {
		int cb_i_subb;
		struct Codeblock *cb;
		subb->K_msbs = (int*) malloc(sizeof(int) * subb->nCodeblocks);
		//cutilSafeCall(cudaFree(subb->daten_d));

		for(cb_i_subb=0; cb_i_subb < subb->nCodeblocks; cb_i_subb++) {
			//needn't copy codeblocks
			//subb->codeblocks[cb_i_subb] = cbs[*cb_i];
			cb = &(subb->codeblocks[cb_i_subb]);

			subb->K_msbs[cb_i_subb] = subb->K_max - cb->nBitplanes;
			assert(subb->K_msbs[cb_i_subb] >= 0);

			*max_mq_len = max(*max_mq_len, cb->L);

			//printf("cb#%3d: L=%3d, Kmsbs=%2d\n", *cb_i, cb->L, subb->K_msbs[cb_i_subb]);
			//print_cb(&(subb->codeblocks[cb_i_subb]));
			//printf("\n");*/
			(*cb_i)++;
		}
	}
}


void copy_cbs_to_pic(struct Picture *pic, struct Codeblock *cbs, int mq_buf_size) {
	int tile_i, ch_i, cb_i=0;
	int max_mq_len=0;

	for(tile_i = 0; tile_i < pic->tile_number; tile_i++) {
		struct Tile *t = &(pic->tiles[tile_i]);
		for(ch_i = 0; ch_i < pic->channels; ch_i++)
			copy_cbs_to_subb(((subband**) t->imgData) [ch_i], cbs, &cb_i, &max_mq_len);
	}

	//printf("max_mq_len %d\n", max_mq_len);
	if(max_mq_len > mq_buf_size) {
		//should never occur, size should be sufficient
		printf("Tier1 error: needed %d bytes per MQ buffer, but only %d allocated.\n", 
			max_mq_len, mq_buf_size);
	}
	//printf("copied %d cbs back to picture\n", cb_i);
}
//****************************************************************





//********************************** Preparation + Kernel invocation

//target_size is used if PCRD is enabled, else it is not used
void tier1_pre_and_kernel(struct Picture *pic, 
						  struct Tier1_Pic *t1pic,
						  int threads_per_block, 
						  int mode /*LOSSY or LOSSLESS*/, 
						  int enable_pcrd /*0 or 1*/,
						  cudaStream_t stream,
						  int max_cb_per_kernel,
						  int use_local_mem,
						  unsigned int timer,
						  double *time_to_gpu, double *time_from_gpu)
{
	int cb_xdim = (1 << (pic->cb_xdim_exp));
	int cb_ydim = (1 << (pic->cb_ydim_exp));

	//printf("codeblocks are %dx%d\n", cb_xdim, cb_ydim);
	//printf("tilesize is %d\n", pic->tilesize);

	int n_cbs = count_cb_pic(pic, cb_xdim, cb_ydim);

	//STREAMING: always only one kernel call
	if(max_cb_per_kernel <= 0)
		max_cb_per_kernel=999999999;
	max_cb_per_kernel = min(max_cb_per_kernel, n_cbs);


	//size in elemtents, not in bytes
	//int state_array_size = (cb_ydim/2 + 2) * (cb_xdim+2);
	//printf("state_array_size %d\n", state_array_size);
	//calculate how much state memory is needed for one kernel launch
	//int total_state_mem = state_array_size * min(max_cb_per_kernel, n_cbs) * sizeof(unsigned);

	//TODO: different values for reversible/irreversible
	int mq_buf_size;
	
	if(mode==LOSSLESS) {
		if(cb_xdim==64)       mq_buf_size=5000;
		else if (cb_xdim==32) mq_buf_size=1500;
		else /*16*/           mq_buf_size= 352;
	}
	else {
		if(cb_xdim==64)       mq_buf_size=4000;
		else if (cb_xdim==32) mq_buf_size=1300;
		else /*16*/           mq_buf_size= 320;
	}
	t1pic->mq_buf_size = mq_buf_size;


	//printf("%d codeblocks\n", n_cbs);

	//*time_tier1 = (double)clock()/(double)CLOCKS_PER_SEC;

	//printf("%d codeblocks in picture\n", n_cbs);
	//printf("bpc + mq coding...\n");

	if(use_local_mem) {
		tier1_capacity(t1pic, n_cbs, max_cb_per_kernel * mq_buf_size, //device buffer neednt store all codeblocks
			n_cbs * mq_buf_size /*host-buf*/, use_local_mem);
	}
	else {
		tier1_capacity(t1pic, n_cbs, n_cbs * mq_buf_size, 
			n_cbs * mq_buf_size /*host-buf*/, use_local_mem);
	}

	copy_pic_to_rawcbs(pic, t1pic->rawcbs_h, t1pic->cbs_h, cb_xdim, cb_ydim);

	if(time_to_gpu != NULL) {
		cutilSafeCall(cudaThreadSynchronize());
		cutResetTimer(timer);
		cutStartTimer(timer);
	}

	cutilSafeCall(cudaMemcpyAsync(t1pic->rawcbs_d, t1pic->rawcbs_h, 
		sizeof(struct Raw_CB) * n_cbs, cudaMemcpyHostToDevice, stream));

	if(time_to_gpu != NULL) {
		cutilSafeCall(cudaThreadSynchronize());
		cutStopTimer(timer);
		*time_to_gpu = cutGetTimerValue(timer);
	}

	//cutilSafeCall(cudaStreamSynchronize(stream));

	//write 0.0f == 0x00000000
	if(enable_pcrd)
		cutilSafeCall(cudaMemset(t1pic->slope_max_d, 0, sizeof(float)));

	dim3 dimBlock(threads_per_block);
	//dim3 dimGrid((n_cbs + threads_per_block - 1) / threads_per_block);
	int s_mem_size = threads_per_block * ALIGNED_MQENC_SIZE;

	//printf("kernel %dx%d threads, %d codeblocks\n", dimGrid.x, dimBlock.x, n_cbs);
	//double kernel_time = (double)clock()/(double)CLOCKS_PER_SEC;

	if(time_from_gpu != NULL)
		*time_from_gpu = 0.0;

	int start_idx=0;
	//cudaThreadSynchronize();
	while(n_cbs > 0) {
		int n_cb_encode = min(max_cb_per_kernel, n_cbs);
		dim3 dimGrid((n_cb_encode + threads_per_block - 1) / threads_per_block);

		if(use_local_mem) {
			encode_cb_kernel_local_mem<<< dimGrid, dimBlock, s_mem_size, stream >>>
				(t1pic->rawcbs_d + start_idx, t1pic->cbs_d + start_idx, n_cb_encode, 
				 t1pic->slope_max_d, pic->xSize, mode, enable_pcrd,
				 t1pic->global_buf_d,
				 t1pic->global_buf_h + start_idx*mq_buf_size, 			
				 mq_buf_size);
			cutilCheckMsg("Tier 1 kernel failed!");

			if(time_from_gpu != NULL) {
				cutilSafeCall(cudaThreadSynchronize());
				cutResetTimer(timer);
				cutStartTimer(timer);
			}

			cutilSafeCall(cudaMemcpyAsync(t1pic->global_buf_h + start_idx*mq_buf_size, 
				t1pic->global_buf_d, n_cb_encode * t1pic->mq_buf_size, 
				cudaMemcpyDeviceToHost, stream));

			if(time_from_gpu != NULL) {
				cutilSafeCall(cudaThreadSynchronize());
				cutStopTimer(timer);
				*time_from_gpu += cutGetTimerValue(timer);
			}
		} 
		else {
			encode_cb_kernel_global_mem<<< dimGrid, dimBlock, s_mem_size, stream >>>
				(t1pic->rawcbs_d + start_idx, t1pic->cbs_d + start_idx, n_cb_encode, 
				 t1pic->slope_max_d, pic->xSize, mode, enable_pcrd,
				 t1pic->global_buf_d + start_idx*mq_buf_size,
				 t1pic->global_buf_h + start_idx*mq_buf_size, 			
				 mq_buf_size, 
				 t1pic->global_states_d       + MAX_STATE_ARRAY_SIZE*start_idx,
				 t1pic->global_dist_d         + PCRD_TEMP_ARRAY_SIZE*start_idx, 
				 t1pic->global_saved_states_d + PCRD_TEMP_ARRAY_SIZE*start_idx);
		}

		n_cbs -= max_cb_per_kernel;
		start_idx += max_cb_per_kernel;
	}
	//kernel_time = (double)clock()/(double)CLOCKS_PER_SEC - kernel_time;
	//printf("returned from kernel, %lf secs\n", kernel_time);
}



// ****************** PCRD + other post-kernel operations

void tier1_post(struct Picture *pic,
				struct Tier1_Pic *t1pic,
				int target_size,
				int /*boolean*/ enable_pcrd,
				cudaStream_t stream)
{
	//** Tier1-kernel has finished at this point

	if(enable_pcrd) {
		//printf("pcrd!\n");
		cutilSafeCall(cudaMemcpyAsync(t1pic->slope_max_h, t1pic->slope_max_d, sizeof(float), 
			cudaMemcpyDeviceToHost, stream));
		cudaStreamSynchronize(stream); //must have slope value copied!
		pcrd_opt(t1pic, t1pic->cbs_d, t1pic->n_cbs, pic->channels * pic->tile_number, 
			*(t1pic->slope_max_h), target_size, stream);
	}

	/*printf("\n\nUsed %0.3lf MB out of %0.3lf MB buffers\n", ((double)t1pic->global_buf_ofs)/1000000.0,
		((double)(t1pic->n_buffers_alloc*MQ_BUFFER_LEN))/1000000.0);
	printf("= %3.1lf%%\n", ((double)t1pic->global_buf_ofs) / ((double)(t1pic->n_buffers_alloc*MQ_BUFFER_LEN)) * 100.0);
	printf("Codeblocks fixed memory size: %0.3lf MB\n\n", ((double)(t1pic->n_cbs*sizeof(Codeblock)))/1000000.0);*/


	//printf("tier1 done\n");
}


void tier1_memcpy_async(struct Tier1_Pic *t1pic, cudaStream_t stream, int use_local_mem) {
	//copy codeblock data back to Picture
	cutilSafeCall(cudaMemcpyAsync(t1pic->cbs_h, t1pic->cbs_d, sizeof(struct Codeblock) * t1pic->n_cbs, 
		cudaMemcpyDeviceToHost, stream));

	if(!use_local_mem) {
		cutilSafeCall(cudaMemcpyAsync(t1pic->global_buf_h, 
				t1pic->global_buf_d, t1pic->n_cbs * t1pic->mq_buf_size, 
				cudaMemcpyDeviceToHost, stream));
	}

	/*cutilSafeCall(cudaMemcpyAsync(t1pic->global_buf_h, t1pic->global_buf_d, t1pic->global_buf_used, 
		cudaMemcpyDeviceToHost, stream));*/
}
