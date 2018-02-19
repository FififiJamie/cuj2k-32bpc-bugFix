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

//  -----MAIN---- function for JPEG2000 Encoder, no streaming

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <time.h>
//#include <cutil.h>
//#include <cutil_inline.h>
#include <helper_cuda.h>
#include <helper_timer.h>
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

//using 0 for stream-number means no streaming == synchronous memcopies
#define NO_STREAM 0



int main_nostream(const unsigned char* src, size_t insize,
                    unsigned char** outputJ2k, size_t* outSize, int argc, char **argv, int *arg_i,
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

	FILE *fp_total, *fp_pre, *fp_dwt, *fp_t1, *fp_pcrd, *fp_t2, *fp_to_gpu, *fp_from_gpu;

	double locstart, locend, timetotal, timepre, timedwt, timetier1, timepcrd, timetier2, timetogpu, timefromgpu;
	double filetotal, filepre, filedwt, filetier1, filepcrd, filetier2, filetogpu, filefromgpu,
		file_t1_togpu, file_t1_fromgpu;

	int size;
	int no_more_files, total_files_compressed=0;

#ifdef ENABLE_WILDCARDS
	HANDLE h_find;
	WIN32_FIND_DATAA find_data;
#endif

	/*for(int k=0; k < argc; k++)
		printf("arg%d: '%s'\n", k, argv[k]);*/

	//Check for streaming compatible gpu
	locstart=(double)clock()/(double)CLOCKS_PER_SEC;


	//compression options for each stream
	char filein[FILENAME_LEN], fileout[FILENAME_LEN];
	int j2k_filesize;
	int pcrd, mode, quant_enable, cb_dim, fileformat;

	int progress=NOT_STARTED_YET;

	//struct Bitmap img;
  struct simpleTIFF img;
	struct Picture pic;
	struct Tier1_Pic t1pic;
	struct Buffer buffer;

	//JPEG2000 filesize = BMP filesize * size_factor (when PCRD enabled)
	FILE *fp;

	int* wavelet_temp_d;
	checkCudaErrors(cudaMalloc((void**)&wavelet_temp_d,1024*1024*sizeof(int)));

	//bmpReset(&img);
	picReset(&pic);
	tier1_pic_reset(&t1pic);

	out_filename_type = NOT_SPECIFIED;
	mj2_set_options(opt_mj2, &opt_fileformat, out_format, &out_filename_type);



	if(opt_bench) {
		//unsigned int timer_part=0, timer_total=0, timer_memcpy=0;
		StopWatchInterface *timer_part=NULL, *timer_total=NULL, *timer_memcpy=NULL;
		char filename[200];
		char comment[] = "# Filesize[MB]   execution time average per file[s]\n";

		sprintf(filename, "%s_total.txt", opt_bench_prefix);
		fp_total=fopen(filename, "w");
		fprintf(fp_total, "%s", comment);

		sprintf(filename, "%s_pre.txt", opt_bench_prefix);
		fp_pre=fopen(filename, "w");
		fprintf(fp_pre, "%s", comment);

		sprintf(filename, "%s_dwt.txt", opt_bench_prefix);
		fp_dwt=fopen(filename, "w");
		fprintf(fp_dwt, "%s", comment);

		sprintf(filename, "%s_tier1.txt", opt_bench_prefix);
		fp_t1=fopen(filename, "w");
		fprintf(fp_t1, "%s", comment);

		sprintf(filename, "%s_pcrd.txt", opt_bench_prefix);
		fp_pcrd=fopen(filename, "w");
		fprintf(fp_pcrd, "%s", comment);

		sprintf(filename, "%s_tier2.txt", opt_bench_prefix);
		fp_t2=fopen(filename, "w");
		fprintf(fp_t2, "%s", comment);

		sprintf(filename, "%s_mem_to_gpu.txt", opt_bench_prefix);
		fp_to_gpu=fopen(filename, "w");
		fprintf(fp_to_gpu, "%s", comment);

		sprintf(filename, "%s_mem_from_gpu.txt", opt_bench_prefix);
		fp_from_gpu=fopen(filename, "w");
		fprintf(fp_from_gpu, "%s", comment);


		sdkCreateTimer(&timer_part);
		sdkCreateTimer(&timer_total);
		sdkCreateTimer(&timer_memcpy);

		while(1) {
			first_to_find=1;
			no_more_files = fetch_next_filename(argc, argv, arg_i, &file_counter,
				&in_format, out_format, &in_filename_type, &out_filename_type,
				&opt_fileformat,
				&opt_mode, &opt_quant_enable,
				&opt_pcrd, &opt_abs_size, &opt_size_factor,
				&opt_cb_dim);
			if(no_more_files)
				break;
			bench_counter=BENCH_RUNS;
			strcpy(filein, in_format);

			FILE *f = fopen(filein, "rb");
			fseek(f, 0, SEEK_END);
			double mb = (double)ftell(f) / 1000000.0;
			fclose(f);

			fprintf(fp_total, "%5.3lf ", mb);
			fprintf(fp_pre, "%5.3lf ", mb);
			fprintf(fp_dwt, "%5.3lf ", mb);
			fprintf(fp_t1, "%5.3lf ", mb);
			fprintf(fp_pcrd, "%5.3lf ", mb);
			fprintf(fp_t2, "%5.3lf ", mb);
			fprintf(fp_to_gpu, "%5.3lf ", mb);
			fprintf(fp_from_gpu, "%5.3lf ", mb);

			timetotal= timepre= timedwt= timetier1= timepcrd= timetier2 = timetogpu = timefromgpu = 0.0;

			while(bench_counter > 0) {

				sdkResetTimer(&timer_total); sdkStartTimer(&timer_total);

				//int status = any_img_read(src, insize, &(img), filein);
				//if(status != 0)
				//	return 1; //exit(1);

				strcpy(fileout, "output.j2k");
				file_counter++;

				//printf("%s (%dx%d) -> %s\n", filein, img.xDim, img.yDim, fileout);

				mode = opt_mode;
				pcrd = opt_pcrd;
				quant_enable = opt_quant_enable;
				cb_dim = opt_cb_dim;
				fileformat = opt_fileformat;
				if(opt_pcrd==SIZE_ABS)
					j2k_filesize = opt_abs_size;
				else if(opt_pcrd==SIZE_REL)
					j2k_filesize = (int)(opt_size_factor * (float)(img.channels *
						img.xDim * img.yDim));

				sdkResetTimer(&timer_part); sdkStartTimer(&timer_part);

				// ***************** tiling(next) (without data transfer)
				//tiling (&pic, &img, cb_dim);

				sdkStopTimer(&timer_part);
				filepre = sdkGetTimerValue(&timer_part);

				// ******** transfering data to device (originally in tiling)(memcpyasync host_to_device)
				size = img.xDim * img.yDim*sizeof(int);

				sdkResetTimer(&timer_part); sdkStartTimer(&timer_part);

				checkCudaErrors (cudaMemcpyAsync((void*)pic.tiles->imgData_d[0],(void*)img.imgData[0],3*size,cudaMemcpyHostToDevice,NO_STREAM));

				sdkStopTimer(&timer_part);
				filetogpu = sdkGetTimerValue(&timer_part);

				sdkResetTimer(&timer_part); sdkStartTimer(&timer_part);

				//dcshift,dwt (aktuell)
				// *********** MCT akt *************
				//dcshift_mct(&pic,mode, NO_STREAM);
				checkCudaErrors(cudaThreadSynchronize());

				sdkStopTimer(&timer_part);
				filepre += sdkGetTimerValue(&timer_part);


				sdkResetTimer(&timer_part);
				sdkStartTimer(&timer_part);

				printf("pic.tile_number = %d\n",pic.tile_number );
				// ********** DWT akt ***************
				for (int i=0; i<pic.tile_number;i++)
					//DWT (&pic.tiles[i], 4, pic.xSize, mode, quant_enable, NO_STREAM,wavelet_temp_d);
				checkCudaErrors(cudaThreadSynchronize());

				sdkStopTimer(&timer_part);
				filedwt = sdkGetTimerValue(&timer_part);

				sdkResetTimer(&timer_part); sdkStartTimer(&timer_part);

				// *************** Tier1 kernel akt ****************
				//tier1_pre_and_kernel(&pic, &(t1pic), 64 /*threads_per_block*/,
				//					 mode, pcrd, NO_STREAM, opt_max_cb_per_kernel,
				//					 opt_use_local_mem,timer_memcpy, &file_t1_togpu, &file_t1_fromgpu);
				checkCudaErrors(cudaThreadSynchronize());

				sdkStopTimer(&timer_part);
				//measure time, but subtract memcpy times
				filetier1 = sdkGetTimerValue(&timer_part) - file_t1_togpu - file_t1_fromgpu;

				sdkResetTimer(&timer_part); sdkStartTimer(&timer_part);

				//filepcrd=(double)clock()/(double)CLOCKS_PER_SEC;
				//nachtier1 (akt)
				tier1_post(&pic, &(t1pic), j2k_filesize, pcrd, NO_STREAM);
				checkCudaErrors(cudaThreadSynchronize());

				sdkStopTimer(&timer_part);
				filepcrd = sdkGetTimerValue(&timer_part);
				//checkCudaErrors(cudaFree(pic.tiles[0].imgData_d[0]));
				//filepcrd=(double)clock()/(double)CLOCKS_PER_SEC - filepcrd;

				sdkResetTimer(&timer_part); sdkStartTimer(&timer_part);

				// ************** Tier1 memcpy async prev *************
				tier1_memcpy_async (&(t1pic), NO_STREAM, opt_use_local_mem);

				// must wait, copy_cbs_to_pic needs host data ready
				//checkCudaErrors(cudaStreamSynchronize(NO_STREAM));
				checkCudaErrors(cudaThreadSynchronize());

				sdkStopTimer(&timer_part);
				filefromgpu = sdkGetTimerValue(&timer_part);

				sdkResetTimer(&timer_part); sdkStartTimer(&timer_part);

				copy_cbs_to_pic(&pic, t1pic.cbs_h, t1pic.mq_buf_size);

				InitializeBuffer(&buffer);
				//encodeCodeStream(&buffer, &pic, &(t1pic), 4, mode, quant_enable);
				//fp = fopen(fileout, "wb");
                fp = fmemopen((void *)*outputJ2k, buffer.ByteCounter + 85, "wb");
    			*outSize = buffer.ByteCounter + 85;
				/*write_fileformat(fp, &pic);
				write_codestream_box(fp, &buffer);*/
				write_output_file(fp, &pic, &buffer, fileformat, 8);
				fclose(fp);
				//printf("FILESIZE: %d\n", buffer.ByteCounter+85);

				free(buffer.Data);
				//filetier2=(double)clock()/(double)CLOCKS_PER_SEC - filetier2;
				//filetotal=(double)clock()/(double)CLOCKS_PER_SEC - filetotal;
				sdkStopTimer(&timer_part);
				filetier2 = sdkGetTimerValue(&timer_part);
				sdkStopTimer(&timer_total);
				filetotal = sdkGetTimerValue(&timer_total);

				filetogpu += file_t1_togpu;
				filefromgpu += file_t1_fromgpu;

				timetotal += filetotal;
				timepre += filepre;
				timedwt += filedwt;
				timetier1 += filetier1;
				timepcrd += filepcrd;
				timetier2 += filetier2;
				timetogpu += filetogpu;
				timefromgpu += filefromgpu;

				total_files_compressed++;
				bench_counter--;
			} //loop bench_counter

			fprintf(fp_total, "%5.6lf\n", timetotal  / 1000.0 / (double)BENCH_RUNS);
			fprintf(fp_pre, "%5.6lf\n", timepre  / 1000.0 / (double)BENCH_RUNS);
			fprintf(fp_dwt, "%5.6lf\n", timedwt  / 1000.0 / (double)BENCH_RUNS);
			fprintf(fp_t1, "%5.6lf\n", timetier1  / 1000.0 / (double)BENCH_RUNS) ;
			fprintf(fp_pcrd, "%5.6lf\n", timepcrd  / 1000.0 / (double)BENCH_RUNS);
			fprintf(fp_t2, "%5.6lf\n", timetier2  / 1000.0 / (double)BENCH_RUNS);
			fprintf(fp_to_gpu, "%5.6lf\n", timetogpu  / 1000.0 / (double)BENCH_RUNS);
			fprintf(fp_from_gpu, "%5.6lf\n", timefromgpu  / 1000.0 / (double)BENCH_RUNS);

		} //file loop

		fclose(fp_total);
		fclose(fp_pre);
		fclose(fp_dwt);
		fclose(fp_t1);
		fclose(fp_pcrd);
		fclose(fp_t2);
		fclose(fp_to_gpu);
		fclose(fp_from_gpu);

		sdkDeleteTimer(&timer_part);
		sdkDeleteTimer(&timer_total);
		sdkDeleteTimer(&timer_memcpy);
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
			//printf("TETS no more file = %d\n", no_more_files);
		while (1) {
			progress = DONE;
			while(!no_more_files) {
				//printf("TETS no more file in= %d\n", no_more_files);

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

				if(status==0) //chaneg here for tiff
          status = tiffRead(src, insize, &(img), filein);
					//status = any_img_read(src, insize, &(img), filein);   ////  Nyriad: file input!!!!!!!!!!!!

				if(status==0) {
					//printf("in status TETS no more file in= %d\n", no_more_files);
					progress = RUNNING;

					if(out_filename_type == SINGLE_FILE)
						strcpy(fileout, out_format);
					else if(out_filename_type == FORMAT_STR)
						sprintf(fileout, out_format, file_counter);
					else { //output not specified
						//printf("filein=%s JPEG2000_EXT[opt_fileformat]=%s\n", filein,  JPEG2000_EXT[opt_fileformat]);
						replace_ext(fileout, filein, JPEG2000_EXT[opt_fileformat]);
					}
					file_counter++;

					//printf("%s (%dx%d) -> %s\n", filein, img.xDim, img.yDim, fileout);

					mode = opt_mode;
					pcrd = opt_pcrd;
					quant_enable = opt_quant_enable;
					cb_dim = opt_cb_dim;
					fileformat = opt_fileformat;
					if(opt_pcrd==SIZE_ABS)
						j2k_filesize = opt_abs_size;
					else if(opt_pcrd==SIZE_REL)
						j2k_filesize = (int)(opt_size_factor * (float)(img.channels *
							img.xDim * img.yDim));

					if(in_filename_type == SINGLE_FILE) {
						first_to_find=1;
						no_more_files = fetch_next_filename(argc, argv, arg_i, &file_counter,
							&in_format, out_format, &in_filename_type, &out_filename_type,
							&opt_fileformat,
							&opt_mode, &opt_quant_enable,
							&opt_pcrd, &opt_abs_size, &opt_size_factor,
							&opt_cb_dim);
							//printf("in status 2 TETS no more file in= %d\n", no_more_files);
					}
					//printf("1\n");
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
			//printf("--------2\n");

			if(progress==DONE)
				break;
				//printf("3\n");
			// ***************** tiling(n�chstes) (without data transfer)
			//pic = .......
      printf("image.bps = %d\n", img.bps);
			tiling2 (&pic, &img, cb_dim);

			// *************** transfering data to device (originally in tiling)(memcpyasync host_to_device)
			size = img.xDim * img.yDim*sizeof(int);
      //copy int
			checkCudaErrors (cudaMemcpyAsync((void*)pic.tiles->imgData_d[0],(void*)img.imgData[0],3*size,cudaMemcpyHostToDevice,NO_STREAM));


		  //dcshift,dwt (aktuell)
			// *********** MCT akt *************
			dcshift_mct(&pic,mode, img.bps, NO_STREAM);
			// ********** DWT akt ***************
			for (int i=0; i<pic.tile_number;i++)
				DWT (&pic.tiles[i], 4, pic.xSize, mode, quant_enable, NO_STREAM, img.bps, wavelet_temp_d);

			//cudaStreamSynchronize(NO_STREAM);
			//cudaDeviceSynchronize();
			//getLastCudaError("~~~~~~~~~~~~~~~~~~~~~~~~~");
			//char * foo = NULL;
			//printf("!!!!!!!!!!sizeof foo = %d\n", sizeof(foo));

			// *************** Tier1 kernel akt ****************
			tier1_pre_and_kernel(&pic, &(t1pic), 64 /*threads_per_block*/,
								 mode, pcrd, NO_STREAM, opt_max_cb_per_kernel,
								 opt_use_local_mem, /*no timing*/0,NULL,NULL, img.bps);



			//nachtier1 (akt)
			tier1_post(&pic, &(t1pic), j2k_filesize, pcrd, NO_STREAM);
			//checkCudaErrors(cudaFree(pic.tiles[0].imgData_d[0]));

			//cudaStreamSynchronize(streams[prev]);
			// ************** Tier1 memcpy async prev *************

			//printf("cudaGetLastError ; %s\n",cudaGetErrorString(cudaGetLastError()));
			tier1_memcpy_async(&(t1pic), NO_STREAM, opt_use_local_mem);
			//printf("cudaGetLastError 2; %s\n",cudaGetErrorString(cudaGetLastError()));

			// must wait, copy_cbs_to_pic needs host data ready
			//printf("cudaErrorInvalidResourceHandle = %d , %lu\n", cudaErrorInvalidResourceHandle),
			//cudaStreamSynchronize(NO_STREAM);
			//checkCudaErrors(cudaStreamSynchronize(NO_STREAM));
			//checkCudaErrors(cudaDeviceSynchronize());
			cudaDeviceSynchronize();
			//getLastCudaError("~~~~~");

			copy_cbs_to_pic(&pic, t1pic.cbs_h, t1pic.mq_buf_size);

			InitializeBuffer(&buffer);
			encodeCodeStream(&buffer, &pic, &(t1pic), 4, mode, quant_enable, img.bps);
			//fp = fopen(fileout, "wb");                     //Nyriad: file output!!!!!!!!!!!!!!!!!!
			printf("DEBUG:~~~\n");
			fp = fmemopen((void *)*outputJ2k, buffer.ByteCounter + 85, "wb");
			*outSize = buffer.ByteCounter + 85;
      printf("DEBUG:~~~\n");
			//printf("DEBUG:~~~codestream.ByteCounter %d\n", buffer.ByteCounter);
			/*write_fileformat(fp, &pic);
			write_codestream_box(fp, &buffer);*/
			write_output_file(fp, &pic, &buffer, fileformat, img.bps);

			fclose(fp);
			//printf("FILESIZE: %d\n", buffer.ByteCounter+85);

			free(buffer.Data);
			total_files_compressed++;
		}
	}


	//tempspeicher f�r Wavlet freen
	checkCudaErrors(cudaFree(wavelet_temp_d));

	tiffFree(&(img));
	free_picture (&pic);
	tier1_free(&(t1pic));


	locend=(double)clock()/(double)CLOCKS_PER_SEC;
	printf("\n\n%d image(s) compressed, total Time:   %lf\n", total_files_compressed, locend-locstart);

	mj2_wrapper(opt_mj2);

	//system("pause");
	return 0;
}
