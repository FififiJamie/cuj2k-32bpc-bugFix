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
Tier 1 include file

struct Codeblock
Stores information about each codeblock: Start address of
buffer on device and host, number of coding passes, slope values,
truncation points

struct Tier1_Pic
Stores buffers for Tier 1 and their allocated sizes, so that
they can be re-used for consecutive encodings.
*/

#ifndef TIER1_H
#define TIER1_H

#include "bitmap.h"
#include "waveletTransform.h"
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <cuda_runtime.h>



//#define TIER1_ENABLE_DEBUG

/* enable debug output? */
#define dbgTier1(...)
//#define dbgTier1(...) (printf(__VA_ARGS__))

/*less frequent messages*/
#define dbgTier1b(...)
//#define dbgTier1b(...) (printf(__VA_ARGS__))



/* laziness + type names might change.... */
#define WhichSubband SubbandTyp
#define QuantCoeffType ausgangsTyp


/* default codeblock size; codeblocks on the right and bottom can be smaller */
//legal ..._EXP values: 2 <= x <= 10 (but consider max. total cb size of 4096 samples)
//...,4=>16, 5=>32, 6=>64,...

#define CB_MAX_XDIM 64
#define CB_MAX_YDIM 64

#define MAX_STATE_ARRAY_SIZE ((CB_MAX_XDIM+2) * (CB_MAX_YDIM/2 + 2))


#define N_CONTEXTS 19

//12:reversible, 9:irreversible, so use 12
#define MAX_BITPLANES 28
//max. number of conding passes
#define MAX_PASSES (3*MAX_BITPLANES - 2)  //do not change here!!!

//maximal length of an MQ codeword, so needn't use mallocs
//pure noise: 4895 bytes (lossless)
//normal picture: 3552 bytes (lossless)

//#define MAX_LEN_MQ 4096

//size of one MQ encoder in shared memory; aligned to 32bit to avoid conflicts
//4x 32bit variables   and   2x N_CONTEXTSx 8bit variables
#define ALIGNED_MQENC_SIZE (((4*sizeof(unsigned) + 2*N_CONTEXTS*sizeof(char) + 3) / 4) * 4)*2
//56 bytes => max 292 threads per block


/* BPC- + MQ-encoded codeblock
   stores the arithmetic encoded bit stream */
struct Codeblock {
	int Xdim, Ydim, nBitplanes;
	//SubbandTyp subbType; /* << needed here? */

	/* number of coding passes in this codeblock */
	int nCodingPasses;

	/* MQ-encoded data (pointer to global device mem) + length */
	unsigned char *B_d, *B_h;
	//int B_offset; //offset from starting address of global buffer
	int L;

	// feasible truncation point information:
	// trunc_len:R, slopes:R-D-slopes
	// ...[n]: values *after* coding pass n
	int n_feasible_trunc;
	int feasible_passes[MAX_PASSES+1], trunc_len[MAX_PASSES+1];
	//needed because we can't encode 0 coding passes
	int trunc_len_1;
	float slopes[MAX_PASSES+1];
	//chosen truncation point (index for above arrays)
	//int trunc_point;
}__attribute__((aligned(512)));


//Data needed for Tier1
struct Tier1_Pic {
	//number of codeblocks, number of allocated codeblocks
	int n_cbs, n_cbs_alloc;
	//all codeblocks in picture; host and device
	struct Codeblock *cbs_d, *cbs_h;
	struct Raw_CB *rawcbs_d, *rawcbs_h;

	unsigned char *global_buf_d, *global_buf_h;
	int global_buf_alloc_d, global_buf_alloc_h, /*global_buf_used,*/ mq_buf_size; //allocated memory in global buffer

	unsigned *global_states_d;

	//these were local before; used for truncation + pcrd
	//int *global_trunc_len_d;
	float *global_dist_d;
	uint4 *global_saved_states_d;

	/*maximum slope value in the whole picture (over all codeblocks,
	  over all subbands and tiles), for finding the optimal lambda value*/
	float *slope_max_d, *slope_max_h;

	int pcrd_sizes_alloc;
	int *pcrd_sizes_h, *pcrd_sizes_d;
};

extern "C"
void tier1_pic_reset(struct Tier1_Pic *t1pic);
extern "C"
void tier1_free(struct Tier1_Pic *t1pic);


/* tier1 on whole picture */
extern "C"
void copy_cbs_to_pic(struct Picture *pic, struct Codeblock *cbs, int mq_buf_size);

extern "C"
void tier1_pre_and_kernel(struct Picture *pic,
						  struct Tier1_Pic *t1pic,
						  int threads_per_block,
						  int mode /*LOSSY or LOSSLESS*/,
						  int enable_pcrd /*0 or 1*/,
						  cudaStream_t stream,
						  int max_cb_per_kernel,
						  int use_local_mem,
						  StopWatchInterface *timer,
						  double *time_to_gpu, double *time_from_gpu /*these may be NULL: no timing*/, int bps );


void tier1_post(struct Picture *pic,
				struct Tier1_Pic *t1pic,
				int target_size,
				int /*boolean*/ enable_pcrd,
				cudaStream_t stream);


extern "C"
void tier1_memcpy_async(struct Tier1_Pic *t1pic, cudaStream_t stream, int use_local_mem);

#endif
