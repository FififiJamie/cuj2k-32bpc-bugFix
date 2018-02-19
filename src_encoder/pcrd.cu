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
PCRD functions; included by tier1.cu

__device__ void pcrd_calc_slopes(.....)
Calculate slope values (distortion decrease divided by rate
increase) for all feasible truncation points. Slopes are multiplied
with a weight based on DWT type, color channel, subband level
and subband type. Called from Tier 1 kernel.

__global__ void trunc_size_cb(struct Codeblock *cbs, int n_cbs, float lambda, int *sizes)
Kernel: each thread calculates the sum of the sizes of
some codeblocks (based on the slope value 'lambda') and stores
this sum in the array 'sizes'.

int trunc_size_pic(...., float lambda, .....)
Calculates the size of the whole picture for given lambda.
This is done by calling the kernel trunc_size_cb and adding
up the values after copying them to the host.

void pcrd_opt(..., int target_size, ....)
Chooses truncation points so that target file size is matched
while minimizing the distortion. If target size is too small/big,
the smallest/biggest reachable value is chosen.
Implementation: Bisection for slope values, calling trunc_size_pic
for each slope value
*/


//1st part of PCRD: calc. feasible truncation points
//and their R-D-slopes

// "almost infinity", but leave bounds for multiplication with delta_l
#define ALMOST_INF (FLT_MAX / 1000.0f)

#define WEIGHT_MAX_CHANNEL 1 //treat red and blue chrominance equally
#define WEIGHT_MAX_LEVEL 3

//channels: 0=luminance   1,2=chrominance
//distortion weights for subbands. SUBB_WEIGHTS[ch][level][subb_type],
//where level=0 are the biggest subbands

__constant__ float SUBB_WEIGHTS_REVERSIBLE[WEIGHT_MAX_CHANNEL+1][WEIGHT_MAX_LEVEL+1][5] = {
// ...Y
{
//  subdiv     LH,      HL,      HH,     LLend
	{ 0.0f, 0.1000f, 0.1000f, 0.0500f, 1.0000f},  //level 0 = biggest subbands (unimportant)
	{ 0.0f, 0.2000f, 0.2000f, 0.1000f, 1.0000f},  //      1
	{ 0.0f, 0.4000f, 0.4000f, 0.2000f, 1.0000f},  //      2
	{ 0.0f, 0.8000f, 0.8000f, 0.4000f, 1.0000f}   //      3 = smallest, contains LL
}, {
	{ 0.0f, 0.0010f, 0.0010f, 0.0005f, 1.0000f},  //HH was 0.0013f before
	{ 0.0f, 0.1000f, 0.1000f, 0.0250f, 1.0000f},
	{ 0.0f, 0.3000f, 0.3000f, 0.0800f, 1.0000f},
	{ 0.0f, 0.8000f, 0.8000f, 0.4000f, 1.0000f}
} };

// ...Y2
/*{
//  subdiv     LH,      HL,      HH,     LLend
	{ 0.0f, 0.1000f, 0.1000f, 0.0500f, 1.0000f},  //level 0 = biggest subbands (unimportant)
	{ 0.0f, 0.2000f, 0.2000f, 0.1000f, 1.0000f},  //      1
	{ 0.0f, 0.4000f, 0.4000f, 0.2000f, 1.0000f},  //      2
	{ 0.0f, 0.8000f, 0.8000f, 0.4000f, 1.0000f}   //      3 = smallest, contains LL
}, {
	{ 0.0f, 0.0010f, 0.0010f, 0.0005f, 1.0000f},  //HH was 0.0013f before
	{ 0.0f, 0.1000f, 0.1000f, 0.0500f, 1.0000f},
	{ 0.0f, 0.3000f, 0.3000f, 0.1500f, 1.0000f},
	{ 0.0f, 0.8000f, 0.8000f, 0.4000f, 1.0000f}
} };*/


__constant__ float SUBB_WEIGHTS_IRREVERSIBLE[WEIGHT_MAX_CHANNEL+1][WEIGHT_MAX_LEVEL+1][5] = {
// ...X
/*{
//  subdiv     LH,      HL,      HH,     LLend
	{ 0.0f, 0.1000f, 0.1000f, 0.0250f, 1.0000f},  //level 0 = biggest subbands (unimportant)
	{ 0.0f, 0.2000f, 0.2000f, 0.0500f, 1.0000f},  //      1
	{ 0.0f, 0.4000f, 0.4000f, 0.1000f, 1.0000f},  //      2
	{ 0.0f, 0.8000f, 0.8000f, 0.2000f, 1.0000f}   //      3 = smallest, contains LL
}, {
	{ 0.0f, 0.0100f, 0.0100f, 0.0025f, 1.0000f},
	{ 0.0f, 0.1000f, 0.1000f, 0.0250f, 1.0000f},
	{ 0.0f, 0.3000f, 0.3000f, 0.0800f, 1.0000f},
	{ 0.0f, 0.8000f, 0.8000f, 0.2000f, 1.0000f}
} };*/


//...X1
/*{
//  subdiv     LH,      HL,      HH,     LLend
	{ 0.0f, 0.1000f, 0.1000f, 0.0500f, 1.0000f},  //level 0 = biggest subbands (unimportant)
	{ 0.0f, 0.2000f, 0.2000f, 0.1000f, 1.0000f},  //      1
	{ 0.0f, 0.4000f, 0.4000f, 0.2000f, 1.0000f},  //      2
	{ 0.0f, 0.8000f, 0.8000f, 0.4000f, 1.0000f}   //      3 = smallest, contains LL
}, {
	{ 0.0f, 0.0010f, 0.0010f, 0.0013f, 1.0000f},
	{ 0.0f, 0.1000f, 0.1000f, 0.0250f, 1.0000f},
	{ 0.0f, 0.3000f, 0.3000f, 0.0800f, 1.0000f},
	{ 0.0f, 0.8000f, 0.8000f, 0.4000f, 1.0000f}
} };*/

//...X2
{
//  subdiv     LH,      HL,      HH,     LLend
	{ 0.0f, 0.0100f, 0.0100f, 0.0050f, 1.0000f},  //level 0 = biggest subbands (unimportant)
	{ 0.0f, 0.2000f, 0.2000f, 0.1000f, 1.0000f},  //      1
	{ 0.0f, 0.4000f, 0.4000f, 0.2000f, 1.0000f},  //      2
	{ 0.0f, 0.8000f, 0.8000f, 0.4000f, 1.0000f}   //      3 = smallest, contains LL
}, {
	{ 0.0f, 0.0010f, 0.0010f, 0.0005f, 1.0000f},  //HH was 0.0013f before
	{ 0.0f, 0.1000f, 0.1000f, 0.0250f, 1.0000f},
	{ 0.0f, 0.3000f, 0.3000f, 0.0800f, 1.0000f},
	{ 0.0f, 0.8000f, 0.8000f, 0.4000f, 1.0000f}
} };




__device__
void pcrd_calc_slopes(struct Codeblock *cb,
			 		  //int *trunc_len,
					  float *dist,
					  SubbandTyp subb_type,
					  int dwt_level,
					  int color_channel,
					  float quantstep,
					  float *slope_max,
					  int mode)
{
	int n_feasible;
	//int feasible[MAX_PASSES+1];
	int last; //last (currently) feasible truncation point
	//float slopes[MAX_PASSES+1];
	int pass, i;
	float delta_d, delta_l;

	//dwt_level = WEIGHT_MAX_LEVEL - min(dwt_level, WEIGHT_MAX_LEVEL); //stay <= 4
	color_channel = min(color_channel, WEIGHT_MAX_CHANNEL);
	dwt_level = min(dwt_level, WEIGHT_MAX_LEVEL); //stay <= 4

	// also consider subband weights and quantization steps
	if(mode==LOSSLESS) {
		for(pass = 1; pass <= cb->nCodingPasses; pass++)
			dist[pass] *= SUBB_WEIGHTS_REVERSIBLE[color_channel][dwt_level][(int)subb_type]
						  * quantstep * quantstep / (float)(cb->Xdim * cb->Ydim);
	}
	else {
		for(pass = 1; pass <= cb->nCodingPasses; pass++)
			dist[pass] *= SUBB_WEIGHTS_IRREVERSIBLE[color_channel][dwt_level][(int)subb_type]
						  * quantstep * quantstep / (float)(cb->Xdim * cb->Ydim);
	}

	cb->slopes[0] = ALMOST_INF; // +INF
	n_feasible = 1;
	cb->feasible_passes[0] = 0;
	last = 0;
	for(pass = 1; pass <= cb->nCodingPasses; pass++) {
		delta_d = dist[last] - dist[pass];
		delta_l = (float)(cb->trunc_len[pass] - cb->trunc_len[last]);
		if(delta_d > 0.0f) {
			while(delta_d >= cb->slopes[last]*delta_l) {
				n_feasible--; //exclude last element
				last = cb->feasible_passes[n_feasible-1]; //get new last element
				delta_d = dist[last] - dist[pass];
				delta_l = (float)(cb->trunc_len[pass] - cb->trunc_len[last]);
			}
			last = pass;/*+1; why +1 (marcellin)??? doesnt work like that*/
			cb->feasible_passes[n_feasible++] = last;
			cb->slopes[last] = delta_d/delta_l;
		}
	}

	cb->n_feasible_trunc = n_feasible;
	for(i = 0; i < n_feasible; i++) {
		pass = cb->feasible_passes[i];
		//cb->feasible_passes[i] = pass;
		cb->trunc_len[i] = cb->trunc_len[pass];
		cb->slopes[i] = cb->slopes[pass];
		if((cb->slopes[i] > *slope_max) && (i != 0)) //don't store +INF
			*slope_max = cb->slopes[i];
		//printf("passes=%d   len=%d   slope=%f\n",
		//	cb->feasible_passes[i], cb->trunc_len[i], cb->slopes[i]);
	}
	//printf("\n");
}



// *************************************************************************
// 2nd part of PCRD: find optimal lambda for given filesize


#define FILE_OVERHEAD 181
#define TILE_OVERHEAD 14
#define CB_OVERHEAD 3

#define CBS_PER_THREAD_PCRD 16
#define THREADS_PER_BLOCK_PCRD 4

//computes the sum of the sizes of CBS_PER_THREAD_PCRD codeblocks
__global__ void trunc_size_cb(struct Codeblock *cbs, int n_cbs, float lambda, int *sizes) {
	//index for result
	int i_res = threadIdx.x + blockIdx.x*THREADS_PER_BLOCK_PCRD;
	int cb_start = i_res * CBS_PER_THREAD_PCRD;
	int cb_stop = min((i_res+1)*CBS_PER_THREAD_PCRD, n_cbs);
	int cb_i, size=0;
	for(cb_i = cb_start; cb_i < cb_stop; cb_i++) {
		struct Codeblock *cb = &(cbs[cb_i]);
		int j = 0;
		while((j+1 < cb->n_feasible_trunc) && (cb->slopes[j+1] > lambda))
			j++;
		if(j > 0) {
			//cb->trunc_point = j;
			cb->nCodingPasses = cb->feasible_passes[j];
			cb->L = cb->trunc_len[j];
			size += cb->trunc_len[j];
		}
		else {
			//wants to truncate all coding passes
			//=> encode at least one coding pass
			//cb->trunc_point = -1;
			cb->nCodingPasses = 1;
			cb->L = cb->trunc_len_1;
			size += cb->trunc_len_1;
		}
	}
	if(cb_start < n_cbs)
		sizes[i_res] = size;
}



int trunc_size_pic(struct Codeblock *cbs_d, int n_cbs, float lambda,
				   int n_threads, int n_blocks, int *sizes_d, int *sizes_h,
				   cudaStream_t stream)
{
	int size, i;

	dim3 dimGrid(n_blocks);
	dim3 dimBlock(THREADS_PER_BLOCK_PCRD);
	trunc_size_cb<<< dimGrid, dimBlock, 0, stream >>>(cbs_d, n_cbs, lambda, sizes_d);
	//waits automatically for kernel to finish
	checkCudaErrors(cudaMemcpyAsync(sizes_h, sizes_d, sizeof(int)*n_threads,
		cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaStreamSynchronize(stream)); //wait, we need the values to sum up

	for(size=i=0; i < n_threads; i++)
		size += sizes_h[i];

	return size;
}


#define MAX_ITERATIONS 50
//allowed filesize difference is calculated proportional
//to target_size by factor DIFF_RATIO, but has the minimal value MIN_DIFF
#define DIFF_RATIO 0.02f
#define MIN_DIFF 400
//2nd part of PCRD: finds the optimal lambda value for the given filesize
void pcrd_opt(struct Tier1_Pic *t1pic, struct Codeblock *cbs_d, int n_cbs, int n_ch_tiles,
			  float slope_max, int target_size, cudaStream_t stream)
{
	float lambda_min = -1.0f, lambda_max = slope_max * 2.0f, lambda_mid;
	int size, allowed_diff, iterations=0;
	int min_size, max_size, overhead;
	int count_refine = 0;

	int n_threads = (n_cbs+CBS_PER_THREAD_PCRD-1) / CBS_PER_THREAD_PCRD;
	int n_blocks = (n_threads+THREADS_PER_BLOCK_PCRD-1) / THREADS_PER_BLOCK_PCRD;
	//int *sizes_d, *sizes_h;
	//sizes_h = (int*)malloc(sizeof(int)*n_threads);

	if(n_threads > t1pic->pcrd_sizes_alloc) {
		t1pic->pcrd_sizes_alloc = n_threads;

		checkCudaErrors(cudaFreeHost(t1pic->pcrd_sizes_h));
		checkCudaErrors(cudaFree(t1pic->pcrd_sizes_d));

		checkCudaErrors(cudaMallocHost((void**)&t1pic->pcrd_sizes_h, sizeof(int)*n_threads));
		checkCudaErrors(cudaMalloc((void**)&t1pic->pcrd_sizes_d, sizeof(int)*n_threads));
	}

	overhead = FILE_OVERHEAD + n_ch_tiles*TILE_OVERHEAD + n_cbs*CB_OVERHEAD;
	//printf("lambda_min: %f   lambda_max: %f\n", lambda_min, lambda_max);

	//when target size is out of possible range, algorithm needn't be run
	min_size = trunc_size_pic(cbs_d, n_cbs, lambda_max, n_threads, n_blocks,
		                      t1pic->pcrd_sizes_d, t1pic->pcrd_sizes_h, stream) + overhead;
	if(target_size <= min_size) {
		printf("pcrd: target size %0.1lfk is too small => using %0.1lfk.\n",
			(double)target_size/1000.0, (double)min_size/1000.0);
		return; //data is already truncated to smallest size
	}
	max_size = trunc_size_pic(cbs_d, n_cbs, lambda_min, n_threads, n_blocks,
		                      t1pic->pcrd_sizes_d, t1pic->pcrd_sizes_h, stream) + overhead;
	if(target_size >= max_size) {
		printf("pcrd: target size %0.1lfk is too big => using %0.1lfk.\n",
			(double)target_size/1000.0, (double)max_size/1000.0);
		return; //data is already truncated to biggest size
	}

	allowed_diff = max(MIN_DIFF, (int)(DIFF_RATIO * (float)target_size));
	//printf("target size:%d   allowed diff:%d\n", target_size, allowed_diff);

	//find optimal lambda by bisection
	do {
		lambda_mid = 0.5f * (lambda_min+lambda_max);

		size = trunc_size_pic(cbs_d, n_cbs, lambda_mid, n_threads, n_blocks,
		                      t1pic->pcrd_sizes_d, t1pic->pcrd_sizes_h, stream) + overhead;

		if(size < target_size)
			lambda_max = lambda_mid; //smaller lambda => makes file bigger
		else
			lambda_min = lambda_mid; //bigger lambda => makes file smaller
		iterations++;
		if(count_refine == 0) {
			if(abs(target_size - size) < allowed_diff)
				count_refine = 1;
		}
		else
			count_refine++;
	} while(count_refine<20  &&  iterations<MAX_ITERATIONS);
	//printf("wanted:%d  got:%d,   %d iterations\n", target_size, size, iterations);
	//assert(iterations<MAX_ITERATIONS); //shouldn't happen => stop
	/*if(iterations==MAX_ITERATIONS)
		printf("PCRD: max. iterations reached\n");*/
	//checkCudaErrors(cudaFreeHost(sizes_h));
	//checkCudaErrors(cudaFree(sizes_d));
}
