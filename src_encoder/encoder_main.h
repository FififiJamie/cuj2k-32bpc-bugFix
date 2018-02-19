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

// commandline functions and main function

#ifndef ENCODER_MAIN_H
#define ENCODER_MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#define CUJ2K_VERSION_STR "1.1"

#ifndef UNIX
//switch: enable filename wildcards handling for Windows?
#define ENABLE_WILDCARDS
#endif



#define BENCH_RUNS 10

//output size mode
#define NO_PCRD 0
#define SIZE_ABS 1
#define SIZE_REL 2

#define FILENAME_LEN 1000

//filename type
#define SINGLE_FILE   0
#define WILDCARDS     1  //filename with * or ?
#define FORMAT_STR    2  //c-style format string
#define NOT_SPECIFIED 3  //output filename not specified


extern const char JPEG2000_EXT[2][10];


//pipeline status
#define NOT_STARTED_YET 0
#define RUNNING 1
#define DONE 2



//get next filename from commandline and options before that
int fetch_next_filename(int argc, char **argv, int *arg_i, int *file_counter,
			char **in_format, char *out_format,
			int *in_filename_type, int *out_filename_type,
			int *opt_fileformat,
			int *opt_mode, int *opt_quant_enable, 
			int *opt_pcrd, int *opt_abs_size, float *opt_size_factor,
			int *opt_cb_dim);

void usage();

//replace filename extension, or append extenstion if there is none yet
//new_ext must include the dot.
void replace_ext(char *out_name, const char *old_name, const char *new_ext);

//cuts off everything after the last \\ or /
void cut_filename(char *filename);

//convenience for Windows users who start cuj2k by double-clicking:
//execute 'pause' so that window does not close immediately.
//linux users won't need this, so this function does nothing under linux
void win_pause();

//Set compression options according to opt_mj2:
//if NULL=> do nothing
//else   => set filetype to J2K and output filename to <opt_mj2>_%05.j2k
void mj2_set_options(char *opt_mj2, int *opt_fileformat, char *out_format,
					 int *out_filename_type);

//if opt_m2j != NULL  =>  call OpenJPEG's MJ2_Wrapper
void mj2_wrapper(char *opt_mj2);

//main functions with(out) streaming
int main_stream(int argc, char **argv, int *arg_i,
			int opt_device, int opt_bench, char *opt_bench_prefix, 
			int opt_use_local_mem, int opt_streaming, int opt_max_cb_per_kernel,
			char *opt_mj2);

int main_nostream(int argc, char **argv, int *arg_i,
			int opt_device, int opt_bench, char *opt_bench_prefix, 
			int opt_use_local_mem, int opt_streaming, int opt_max_cb_per_kernel,
			char *opt_mj2);

#ifdef __cplusplus
}
#endif

#endif
