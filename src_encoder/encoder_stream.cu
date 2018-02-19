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

//  -----MAIN---- function for JPEG2000 Encoder, streaming enabled

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cutil.h>
#include <cutil_inline.h>
#include "file_access.h"
#include "bitmap.h"
#include "waveletTransform.h"
#include "tier2.h"
#include "tier1.h"
#include "pic_preprocessing.h"
#include "BufferedWriter.h"
#include "rate-control.h"
#include "misc.h"
#include "device.h"
#include "encoder_main.h"

#ifdef ENABLE_WILDCARDS
#include <windows.h>
#endif


#define ANZ_STREAMS 3

int main_stream(int argc, char **argv, int *arg_i,
			int opt_device, int opt_bench, char *opt_bench_prefix, 
			int opt_use_local_mem, int opt_streaming, int opt_max_cb_per_kernel,
			char *opt_mj2)
{
	int opt_mode=LOSSLESS;
	int opt_pcrd=NO_PCRD;
	float opt_size_factor=0.3f;
	int opt_abs_size=0;
	char *in_format=NULL, out_format[FILENAME_LEN];
	int in_filename_type, out_filename_type; //if it's a single file (0) or sth like picture%d.bmp (1)
	int file_counter=0;
	int first_to_find; //boolean
	int opt_quant_enable=1;
	int opt_cb_dim=0; //codeblock size, 16, 32 or 64; 0 means automatic (calculated depending on pic size)
	int opt_fileformat=FORMAT_JP2;
	int bench_counter=1;

	double locstart, locend, timetotal;
	FILE *fp_total;

	int size;
	int no_more_files, total_files_compressed=0;

	//int kernel_timeout;

#ifdef ENABLE_WILDCARDS
	HANDLE h_find;
	WIN32_FIND_DATAA find_data;
#endif

	
	/*for(int k=0; k < argc; k++)
		printf("arg%d: '%s'\n", k, argv[k]);*/

	locstart=(double)clock()/(double)CLOCKS_PER_SEC;
	
	const int nstreams = ANZ_STREAMS; //3-level pipeline
    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
    for(int i = 0; i < nstreams; i++)
        cutilSafeCall( cudaStreamCreate(&(streams[i])) );

	//compression options for each stream
	char filein[FILENAME_LEN], fileout[nstreams][FILENAME_LEN];
	int j2k_filesize[nstreams];
	int pcrd[nstreams], mode[nstreams], quant_enable[nstreams], cb_dim[nstreams], 
		fileformat[nstreams];

	struct Bitmap img[nstreams];
	struct Picture pic[nstreams];
	struct Tier1_Pic t1pic[nstreams];
	struct Buffer buffer;
	int progress[nstreams]={NOT_STARTED_YET,NOT_STARTED_YET,NOT_STARTED_YET};

	FILE *fp;
	
	int prev=2,akt=0,next=1; //DEBUG: does this make sense?

	int* wavelet_temp_d;
	cutilSafeCall(cudaMalloc((void**)&wavelet_temp_d,1024*1024*sizeof(int)));

	for(int i=0; i < nstreams; i++) {
		bmpReset(&(img[i])); //important!
		picReset(&(pic[i]));
		tier1_pic_reset(&(t1pic[i]));
	}

	out_filename_type = NOT_SPECIFIED;
	mj2_set_options(opt_mj2, &opt_fileformat, out_format, &out_filename_type);

	if(opt_bench) {
		char filename[200];
		char comment[] = "# Filesize[MB]   execution time average per file[s]\n";
		sprintf(filename, "%s_stream.txt", opt_bench_prefix);
		fp_total=fopen(filename, "w");
		fprintf(fp_total, "%s", comment);

		unsigned int timer=0;
		cutCreateTimer(&timer);

		no_more_files = 0;
		while(1) { //file loop
			progress[0] = progress[1] = progress[2] = NOT_STARTED_YET;
			prev=2; akt=0; next=1;
			bench_counter=BENCH_RUNS;

			first_to_find=1;
			no_more_files = fetch_next_filename(argc, argv, arg_i, &file_counter,
				&in_format, out_format, &in_filename_type, &out_filename_type,
				&opt_fileformat,
				&opt_mode, &opt_quant_enable, 
				&opt_pcrd, &opt_abs_size, &opt_size_factor,
				&opt_cb_dim);
			if(no_more_files)
				break;

			strcpy(filein, in_format);

			FILE *f = fopen(filein, "rb");
			fseek(f, 0, SEEK_END);
			double mb = (double)ftell(f) / 1000000.0;
			fclose(f);
			fprintf(fp_total, "%5.3lf ", mb);

			cutResetTimer(timer); cutStartTimer(timer);

			//wait until all streams have finished
			while (progress[0]!=DONE || progress[1]!=DONE || progress[2]!=DONE) {

				//dcshift,dwt (aktuell)	
				if(progress[akt] == RUNNING) {
					// *********** MCT akt *************
					dcshift_mct(&pic[akt],mode[akt], streams[akt]);

					// ********** DWT akt ***************
					for (int i=0; i<pic[akt].tile_number;i++)
						DWT (&pic[akt].tiles[i], 4, pic[akt].xSize, mode[akt], quant_enable[akt], streams[akt],wavelet_temp_d);

					//cudaStreamSynchronize(streams[akt]);


					// *************** Tier1 kernel akt ****************
					tier1_pre_and_kernel(&pic[akt], &(t1pic[akt]), 64 /*threads_per_block*/,
										 mode[akt], pcrd[akt], streams[akt], opt_max_cb_per_kernel,
										 opt_use_local_mem, /*no timing*/0,NULL,NULL);
				} //if(progress[akt]==RUNNING)
			
				if(progress[prev] == RUNNING) {
					//cudaStreamSynchronize(streams[prev]);
					// ************** Tier1 memcpy async prev *************
					tier1_memcpy_async (&(t1pic[prev]), streams[prev], opt_use_local_mem);
				}

				if(bench_counter > 0) {
					progress[next] = RUNNING;
					int status = any_img_read(&(img[next]), filein);
					if(status != 0)
						exit(1);
					strcpy(fileout[next], "output.j2k");
					file_counter++;

					printf("%s (%dx%d) -> %s\n", filein, img[next].xDim, img[next].yDim, fileout[next]);

					mode[next] = opt_mode;
					pcrd[next] = opt_pcrd;
					quant_enable[next] = opt_quant_enable;
					cb_dim[next] = opt_cb_dim;
					fileformat[next] = opt_fileformat;
					if(opt_pcrd==SIZE_ABS)
						j2k_filesize[next] = opt_abs_size;
					else if(opt_pcrd==SIZE_REL)
						j2k_filesize[next] = (int)(opt_size_factor * (float)(img[next].channels * 
							img[next].xDim * img[next].yDim));
					bench_counter--;
				}
				else
					progress[next] = DONE;

				if(progress[next] == RUNNING) {
					// ***************** tiling(nächstes) (without data transfer)
					//pic[next] = .......
					tiling (&pic[next], &img[next], cb_dim[next]);

					// *************** transfering data to device (originally in tiling)(memcpyasync host_to_device)
					size = img[next].xDim * img[next].yDim*sizeof(int);

					cutilSafeCall (cudaMemcpyAsync((void*)pic[next].tiles->imgData_d[0],(void*)img[next].imgData[0],3*size,cudaMemcpyHostToDevice,streams[next]));
				}

				if(progress[prev] == RUNNING) {
					// must wait, copy_cbs_to_pic needs host data ready
					cutilSafeCall(cudaStreamSynchronize(streams[prev])); 

					copy_cbs_to_pic(&pic[prev], t1pic[prev].cbs_h, t1pic[prev].mq_buf_size);

					InitializeBuffer(&buffer);
					encodeCodeStream(&buffer, &pic[prev], &(t1pic[prev]), 4, mode[prev], quant_enable[prev]);
					fp = fopen(fileout[prev], "wb");
					write_output_file(fp, &pic[prev], &buffer, fileformat[prev]);
					/*write_fileformat(fp, &pic[prev]);
					write_codestream_box(fp, &buffer);*/
					fclose(fp);
					//printf("FILESIZE: %d\n", buffer.ByteCounter+85);

					free(buffer.Data);
					total_files_compressed++;
				}
			
				if(progress[akt] == RUNNING) {
					//nachtier1 (akt)
					tier1_post(&pic[akt], &(t1pic[akt]), j2k_filesize[akt], pcrd[akt], streams[akt]);
					//cutilSafeCall(cudaFree(pic[akt].tiles[0].imgData_d[0]));
				}

				akt=(akt+1)%3;
				next=(akt+1)%3;
				prev=(akt+2)%3;
			}//pipelineschleifen-ende
			///////////////////////////////////////////////////////////////

			cutStopTimer(timer);
			timetotal = cutGetTimerValue(timer) / 1000.0;

			fprintf(fp_total, "%5.6lf\n", timetotal / (double)BENCH_RUNS);
		} //file loop

		cutDeleteTimer(timer);

		fclose(fp_total);
	}
	else { //NO BENCHMARK

		first_to_find=1;
		//out_filename_type = NOT_SPECIFIED;
		no_more_files = fetch_next_filename(argc, argv, arg_i, &file_counter,
			&in_format, out_format, &in_filename_type, &out_filename_type,
			&opt_fileformat,
			&opt_mode, &opt_quant_enable, 
			&opt_pcrd, &opt_abs_size, &opt_size_factor,
			&opt_cb_dim);

		//////////////////////////////////////////////////////////////////////
		//pipeline-Schleife

		//wait until all streams have finished
		while (progress[0]!=DONE || progress[1]!=DONE || progress[2]!=DONE) {

			//dcshift,dwt (aktuell)	
			if(progress[akt] == RUNNING) {
				// *********** MCT akt *************
				dcshift_mct(&pic[akt],mode[akt], streams[akt]);

				// ********** DWT akt ***************
				for (int i=0; i<pic[akt].tile_number;i++)
					DWT (&pic[akt].tiles[i], 4, pic[akt].xSize, mode[akt], quant_enable[akt], streams[akt],wavelet_temp_d);

				//cudaStreamSynchronize(streams[akt]);


				// *************** Tier1 kernel akt ****************
				tier1_pre_and_kernel(&pic[akt], &(t1pic[akt]), 64 /*threads_per_block*/,
									 mode[akt], pcrd[akt], streams[akt], opt_max_cb_per_kernel,
									 opt_use_local_mem, /*no timing*/0,NULL,NULL);
			} //if(progress[akt]==RUNNING)
		
			if(progress[prev] == RUNNING) {
				//cudaStreamSynchronize(streams[prev]);
				// ************** Tier1 memcpy async prev *************
				tier1_memcpy_async (&(t1pic[prev]), streams[prev], opt_use_local_mem);
			}

			progress[next] = DONE;
			while(!no_more_files) {
				int status=0;
				if(in_filename_type == FORMAT_STR)
					sprintf(filein, in_format, file_counter);
#ifdef ENABLE_WILDCARDS
				else if(in_filename_type == WILDCARDS) {
					if(first_to_find) {
						first_to_find=0;
						h_find = FindFirstFileA(in_format, &find_data);
						if(h_find == INVALID_HANDLE_VALUE) {
							status=1; //indicate error
							printf("No files found matching '%s'.\n", in_format);
						}
						else {
							cut_filename(in_format); //we need the folder name
							strcpy(filein, in_format); //copy folder name
							strcat(filein, find_data.cFileName);
						}
					}
					else {
						if(FindNextFileA(h_find, &find_data) == 0) {
							status=1; //indicate error
							FindClose(h_find); //free handle
						}
						else {
							strcpy(filein, in_format); //copy folder name
							strcat(filein, find_data.cFileName);
						}
					}
				}
#endif
				else {
					strcpy(filein, in_format);
				}

				if(status==0)
					status = any_img_read(&(img[next]), filein);

				if(status==0) {
					progress[next] = RUNNING;

					if(out_filename_type == SINGLE_FILE)
						strcpy(fileout[next], out_format);
					else if(out_filename_type == FORMAT_STR)
						sprintf(fileout[next], out_format, file_counter);
					else //output not specified
						replace_ext(fileout[next], filein, JPEG2000_EXT[opt_fileformat]);
					file_counter++;

					printf("%s (%dx%d) -> %s\n", filein, img[next].xDim, img[next].yDim, fileout[next]);

					mode[next] = opt_mode;
					pcrd[next] = opt_pcrd;
					quant_enable[next] = opt_quant_enable;
					cb_dim[next] = opt_cb_dim;
					fileformat[next] = opt_fileformat;
					if(opt_pcrd==SIZE_ABS)
						j2k_filesize[next] = opt_abs_size;
					else if(opt_pcrd==SIZE_REL)
						j2k_filesize[next] = (int)(opt_size_factor * (float)(img[next].channels * 
							img[next].xDim * img[next].yDim));

					if(in_filename_type == SINGLE_FILE) {
						first_to_find=1;
						no_more_files = fetch_next_filename(argc, argv, arg_i, &file_counter,
							&in_format, out_format, &in_filename_type, &out_filename_type,
							&opt_fileformat,
							&opt_mode, &opt_quant_enable, 
							&opt_pcrd, &opt_abs_size, &opt_size_factor,
							&opt_cb_dim);
					}

					break;
				}
				else {
					first_to_find=1;
					no_more_files = fetch_next_filename(argc, argv, arg_i, &file_counter,
						&in_format, out_format, &in_filename_type, &out_filename_type,
						&opt_fileformat,
						&opt_mode, &opt_quant_enable, 
						&opt_pcrd, &opt_abs_size, &opt_size_factor,
						&opt_cb_dim);
				}
			}


			if(progress[next] == RUNNING) {
				// ***************** tiling(nächstes) (without data transfer)
				//pic[next] = .......
				tiling (&pic[next], &img[next], cb_dim[next]);

				// *************** transfering data to device (originally in tiling)(memcpyasync host_to_device)
				size = img[next].xDim * img[next].yDim*sizeof(int);

				cutilSafeCall (cudaMemcpyAsync((void*)pic[next].tiles->imgData_d[0],(void*)img[next].imgData[0],3*size,cudaMemcpyHostToDevice,streams[next]));
			}

			if(progress[prev] == RUNNING) {
				// must wait, copy_cbs_to_pic needs host data ready
				cutilSafeCall(cudaStreamSynchronize(streams[prev])); 

				copy_cbs_to_pic(&pic[prev], t1pic[prev].cbs_h, t1pic[prev].mq_buf_size);

				InitializeBuffer(&buffer);
				encodeCodeStream(&buffer, &pic[prev], &(t1pic[prev]), 4, mode[prev], quant_enable[prev]);
				fp = fopen(fileout[prev], "wb");
				write_output_file(fp, &pic[prev], &buffer, fileformat[prev]);
				/*write_fileformat(fp, &pic[prev]);
				write_codestream_box(fp, &buffer);*/
				fclose(fp);
				//printf("FILESIZE: %d\n", buffer.ByteCounter+85);

				free(buffer.Data);
				total_files_compressed++;
			}
		
			if(progress[akt] == RUNNING) {
				//nachtier1 (akt)
				tier1_post(&pic[akt], &(t1pic[akt]), j2k_filesize[akt], pcrd[akt], streams[akt]);
				//cutilSafeCall(cudaFree(pic[akt].tiles[0].imgData_d[0]));
			}

			akt=(akt+1)%3;
			next=(akt+1)%3;
			prev=(akt+2)%3;
		}//pipelineschleifen-ende
		///////////////////////////////////////////////////////////////
	}
	//tempspeicher für Wavlet freen
	cutilSafeCall(cudaFree(wavelet_temp_d));
    // destroy streams
    for(int i = 0; i < nstreams; i++)
        cutilSafeCall( cudaStreamDestroy(streams[i]) );

	for(int i=0; i < nstreams; i++) {
		bmpFree(&(img[i]));
		free_picture (&pic[i]);
		tier1_free(&(t1pic[i]));
	}
	
	locend=(double)clock()/(double)CLOCKS_PER_SEC;
	printf("\n\n%d file(s) compressed, total Time:   %lf\n", total_files_compressed, locend-locstart);

	mj2_wrapper(opt_mj2);

	//system("pause");
	return 0;
}
