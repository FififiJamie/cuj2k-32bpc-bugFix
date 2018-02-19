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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "device.h"
#include "bitmap.h"
#include "encoder_main.h"
#include "file_access.h"

#include "DataTypes.h"

const char JPEG2000_EXT[2][10] = { ".jp2", ".j2k" };

int Cuj2kMain(const unsigned char* src, size_t insize,
                    unsigned char** outputJ2k, size_t* outSize, int argc, char **argv) {
	// index for argv
	int arg_i = 1;

	// global options *************
	int opt_device = -1; //cuda device, default: autoselect
	int opt_bench = 0;   //boolean: enable benchmark?
	char opt_bench_prefix[100]; //prefix for timing files
	int opt_use_local_mem = -1; //default: depends on streaming
	int opt_streaming = 1; //boolean
	int opt_max_cb_per_kernel = 4*1024; //max. codeblocks per kernel
	char *opt_mj2 = NULL;

	if(argc <= 1) {
		usage();
		win_pause();
		return 0;
	}

	// parse global options.................................
	while(arg_i < argc) {
		if(strcmp(argv[arg_i], "-listdev") == 0) {
			list_devices();
			win_pause();
			return 0;
		}
		else if(strcmp(argv[arg_i], "-setdev") == 0) {
			arg_i++;
			if(!(arg_i < argc)) {
				printf("Error: missing argument for option -setdev\n");
				return 1;
			}
			opt_device = atoi(argv[arg_i]);
		}
		else if(strcmp(argv[arg_i], "-benchmark") == 0) {
			opt_bench = 1;
			printf("Prefix for benchmark filenames? ");
			gets(opt_bench_prefix);
		}
		else if(strcmp(argv[arg_i], "-local") == 0) {
			opt_use_local_mem=1;
		}
		else if(strcmp(argv[arg_i], "-global") == 0) {
			opt_use_local_mem=0;
		}
		else if(strcmp(argv[arg_i], "-nostream") == 0) {
			opt_streaming=0;
		}
		else if(strcmp(argv[arg_i], "-maxcb") == 0) {
			arg_i++;
			if(!(arg_i < argc)) {
				printf("Error: missing argument for option -maxcb\n");
				return 1;
			}
			opt_max_cb_per_kernel = atoi(argv[arg_i]);
		}
		else if(strcmp(argv[arg_i], "-mj2") == 0) {
			arg_i++;
			if(!(arg_i < argc)) {
				printf("Error: missing argument for option -mj2\n");
				return 1;
			}
			opt_mj2 = argv[arg_i];
		}
		else
			break; //no more global options => stop here
		arg_i++;
	}

	//option not set => depends on streaming
	if(opt_use_local_mem == (-1)) {
		if(opt_streaming)
			opt_use_local_mem = 0;
		else
			opt_use_local_mem = 1;
	}

	// set CUDA device......
	if(opt_device == (-1)) {
		int kernel_timeout_dummy; //not used
		//....auto-select
		if (choose_stream_gpu(&kernel_timeout_dummy) == 1)
			return 1;
	}
	else {
		//...user-select
		if(user_set_device(opt_device) != 0)
			return 1;
	}


	if(opt_streaming)
		return(main_stream(src, insize,
		                    outputJ2k, outSize, argc, argv, &arg_i,
			opt_device, opt_bench, opt_bench_prefix,
			opt_use_local_mem, opt_streaming, opt_max_cb_per_kernel,
			opt_mj2));
	else
		return(main_nostream(src, insize,
		                    outputJ2k, outSize, argc, argv, &arg_i,
			opt_device, opt_bench, opt_bench_prefix,
			opt_use_local_mem, opt_streaming, opt_max_cb_per_kernel,
			opt_mj2));
}



//return 0 if OK, or != 0 if there are no more file arguments
int fetch_next_filename(int argc, char **argv, int *arg_i, int *file_counter,
			char **in_format, char *out_format,
			int *in_filename_type, int *out_filename_type,
			int *opt_fileformat,
			int *opt_mode, int *opt_quant_enable,
			int *opt_pcrd, int *opt_abs_size, float *opt_size_factor,
			int *opt_cb_dim)
{
	int size_arg_given=0, in_arg_given=0, out_arg_given=0;

	//don't reset: for MJ2 creation continous counting is required
	//(*file_counter) = 0;

	while((!in_arg_given)  &&  (*arg_i) < argc) {
		//printf("%s\n", argv[(*arg_i)]);

		if(argv[(*arg_i)][0] != '-') {
			in_arg_given=1;
			*in_format = argv[(*arg_i)]; //copy address
		}

		else if(strcmp(argv[(*arg_i)], "-o") == 0) {
			out_arg_given=1;
			(*arg_i)++;
			if((*arg_i) == argc) {
				printf("file argument for '-o'-parameter missing.\n");
				return 1;
			}
			strcpy(out_format, argv[(*arg_i)]);
		}

		else if(strcmp(argv[(*arg_i)], "-rev") == 0) {
			(*opt_mode) = LOSSLESS;
			if(!size_arg_given)
				(*opt_pcrd) = NO_PCRD;
		}

		else if(strcmp(argv[(*arg_i)], "-irrev") == 0)
			(*opt_mode) = LOSSY;

		else if(strcmp(argv[(*arg_i)], "-nopcrd") == 0)
			(*opt_pcrd) = NO_PCRD;

		else if(strcmp(argv[(*arg_i)], "-size") == 0) {
			size_arg_given=1;
			(*arg_i)++;
			if((*arg_i) == argc) {
				printf("absolute size argument for '-size'-parameter missing.\n");
				return 1;
			}
			int last = strlen(argv[(*arg_i)]) - 1;
			char unit = argv[(*arg_i)][last];
			if(tolower(unit)=='k' || tolower(unit)=='m') {
				double mult;
				argv[(*arg_i)][last] = 0;
				(*opt_pcrd) = SIZE_ABS;
				switch(unit) {
					case 'k':        mult = 1000.0; break;
					case 'K':        mult = 1024.0; break;
					case 'm':        mult = 1000.0*1000.0; break;
					default /*'M'*/: mult = 1024.0*1024.0; break;
				}
				(*opt_abs_size) = (int)(atof(argv[(*arg_i)]) * mult);
			}
			else {
				(*opt_pcrd) = SIZE_ABS;
				(*opt_abs_size) = atoi(argv[(*arg_i)]);
			}
		}

		else if(strcmp(argv[(*arg_i)], "-sizerel") == 0) {
			size_arg_given=1;
			(*arg_i)++;
			if((*arg_i) == argc) {
				printf("relative size argument for '-sizerel'-parameter missing.\n");
				return 1;
			}
			(*opt_pcrd) = SIZE_REL;
			(*opt_size_factor) = (float)atof(argv[(*arg_i)]);
		}

		else if(strcmp(argv[(*arg_i)], "-ratio") == 0) {
			size_arg_given=1;
			(*arg_i)++;
			if((*arg_i) == argc) {
				printf("relative size argument for '-ratio'-parameter missing.\n");
				return 1;
			}
			(*opt_pcrd) = SIZE_REL;
			(*opt_size_factor) = (float)atof(argv[(*arg_i)]);
			if((*opt_size_factor) == 0.0f) {
				printf("argument for '-ratio'-parameter must not be 0.\n");
				return 1;
			}
			(*opt_size_factor) = 1.0f / (*opt_size_factor);
		}

		else if(strcmp(argv[(*arg_i)], "-starti")== 0) {
			(*arg_i)++;
			if((*arg_i) == argc) {
				printf("start index argument for option '-starti' missing.\n");
				return 1;
			}
			(*file_counter) = atoi(argv[(*arg_i)]);
		}

		else if(strcmp(argv[(*arg_i)], "-hq")== 0) {
			(*opt_quant_enable)=0;
		}

		else if(strcmp(argv[(*arg_i)], "-cb")== 0) {
			(*arg_i)++;
			if((*arg_i) == argc) {
				printf("argument (codeblock size; 16, 32, 64 or 0 (=automatic) allowed) for option '-cb' missing.\n");
				return 1;
			}
			(*opt_cb_dim) = atoi(argv[(*arg_i)]);
			switch(*opt_cb_dim) {
				case 0:  case 16: case 32: case 64:
					break;
				default:
					printf("allowed codeblock sizes are 16, 32, 64 or 0 (=automatic).\n");
					return 1;
			}
		}

		else if(strcmp(argv[(*arg_i)], "-format")== 0) {
			(*arg_i)++;
			if((*arg_i) == argc) {
				printf("argument (jp2 or j2k) for option '-format' missing.\n");
				return 1;
			}
			if(strcmp(argv[(*arg_i)], "jp2")== 0)
				(*opt_fileformat) = FORMAT_JP2;
			else if(strcmp(argv[(*arg_i)], "j2k")== 0)
				(*opt_fileformat) = FORMAT_J2K;
			else {
				printf("invalid argument for option '-format' (allowed: jp2 or j2k)\n");
				return 1;
			}
		}

		else {
			printf("unrecognized option '%s'\n", argv[(*arg_i)]);
			return 1;
		}
		(*arg_i)++;
	} //while((!in_arg_given)  &&  (*arg_i) < argc) {

	if(!in_arg_given)
		return 1;

	char *found = strchr(*in_format, '$');
	if(found != NULL)
		*found = '%'; //replace $ -> %

	if(strchr(*in_format, '%') != NULL)
		(*in_filename_type) = FORMAT_STR;
	else if((strchr(*in_format, '*') != NULL) || (strchr(*in_format, '?') != NULL)) {
#ifndef ENABLE_WILDCARDS
		printf("Wildcards must be enabled when compiling this program.\n"
			"(only supported for Windows)\n");
		return 1;
#else
		(*in_filename_type) = WILDCARDS;
		//first_to_find=1;
#endif
	}
	else
		(*in_filename_type) = SINGLE_FILE;

	if(!out_arg_given) {
		//only reset if no format string
		if((*out_filename_type) == SINGLE_FILE)
			(*out_filename_type) = NOT_SPECIFIED;
	}
	else {
		found = strchr(out_format, '$');
		if(found != NULL)
			*found = '%'; //replace $ -> %

		if(strchr(out_format, '%') != NULL)
			(*out_filename_type) = FORMAT_STR;
		else if((strchr(out_format, '*') != NULL) || (strchr(out_format, '?') != NULL)) {
			printf("Only format strings or single filenames are allowed for output\n");
			return 1;
		}
		else
			(*out_filename_type) = SINGLE_FILE;
	}

	printf("      options: ");
	if((*opt_fileformat) == FORMAT_J2K)
		printf("codestream-only(.J2K), ");

	if((*opt_mode)==LOSSLESS) {
		//lossless + pcrd is in fact not lossless
		if((*opt_pcrd) != NO_PCRD)
			printf("\"lossless\" compression, ");
		else
			printf("lossless compression, ");
	}
	else {
		printf("lossy compression, ");
		if(!(*opt_quant_enable))
			printf("no ");
		printf("quantization, ");
	}
	if((*opt_pcrd) == NO_PCRD)
		printf("no pcrd\n");
	else if((*opt_pcrd) == SIZE_REL) {
		if((*opt_size_factor) == 0.0f)
			printf("ratio=INF:1\n");
		else
			printf("ratio=%0.1f:1\n", 1.0f / (*opt_size_factor));
		//printf("size=%0.1f%%\n", (*opt_size_factor)*100.0f);
	}
	else
		printf("size=%0.1fk\n", (float)(*opt_abs_size) / 1000.0f);

	//printf("in-format: %s    out-format: %s\n", in_format, out_format);
	return 0;
}


void replace_ext(char *out_name, const char *old_name, const char *new_ext) {
	if(strlen(old_name) >= 4  &&  old_name[strlen(old_name)-4]=='.') {
		//replace file extension
		strcpy(out_name, old_name);
		strcpy(out_name + strlen(out_name) - 4, new_ext);
	}
	else {
		//append file extension
		strcpy(out_name, old_name);
		strcat(out_name, new_ext);
	}
}


void cut_filename(char *filename) //cuts off everything after the last \\ or /
{
	for(int i = strlen(filename)-1; i>=0; i--) {
		if(filename[i]=='/'  ||  filename[i]=='\\') {
			filename[i+1]=0;
			return;
		}
	}
	//no slash or backslash => delete whole string
	filename[0]=0;
}

void win_pause() {
#ifndef UNIX
	system("pause");
#endif
}

void mj2_set_options(char *opt_mj2, int *opt_fileformat, char *out_format,
					 int *out_filename_type)
{
	if(opt_mj2 != NULL) {
		*opt_fileformat = FORMAT_J2K;
		sprintf(out_format, "%s_%%05d.j2k", opt_mj2);
		*out_filename_type = FORMAT_STR;
	}
}

void mj2_wrapper(char *opt_mj2) {
	if(opt_mj2 != NULL) {
		char mj2_cmd[2200];
		sprintf(mj2_cmd, "mj2_wrapper \"%s\" \"%s.mj2\"", opt_mj2, opt_mj2);
		printf("Creating MJ2 video file:\n%s\n", mj2_cmd);
		int ret = system(mj2_cmd);
		if(ret != 0)
			printf("Error code %d.\n", ret);
	}
}



void usage() {
	const char u[] =

"CUJ2K " CUJ2K_VERSION_STR " - JPEG2000 Encoder on CUDA\n"
"http://cuj2k.sourceforge.net/\n"
"\n"
"Copyright (c) 2009 Norbert Fuerst, Martin Heide, Armin Weiss, Simon Papandreou, Ana Balevic\n"
"See copyright.txt for license.\n"
"*******************************************************************************\n"
"\n"
"INVOCATION: cuj2k [global-options] [file-options] input-file(s) [more file-options and input-files ...]\n"
#ifdef ENABLE_WILDCARDS
"\n"
"Input filenames may contain wildcards like *.bmp.\n"
#endif
"\n"
"*******************************************************************************\n"
"\n"
"\n"
"Available OPTIONS: (see PDF manual for further explanation + examples)\n"
"Note: <n> stands for an integer, <f> stands for a float (like 3.77)\n"
"\n"
"\n"
"************** GLOBAL options ************************\n"
"\n"
"-mj2 <mj2-file>  --  Create the Motion JPEG2000 file <mj2-file>.mj2 from all compressed pictures, using the intermediate files <mj2-file>_00000.j2k, ...00001.j2k for each frame. "
"This option requires OpenJPEG's mj2_wrapper. Note: You may not use the options '-o' or '-format' together with '-mj2'.\n"
"\n"
"-nostream  --  disable streaming. Streaming speeds up encoding for big pictures on high-end GPUs, but consumes lots of memory. So try to use this option if you get problems with big files.\n"
"\n"
"-setdev <n>  --  use CUDA device number <n> for computations on GPU. If omitted, a suitable device (compute capability >= 1.1) is automatically chosen.\n"
"\n"
"-listdev  --  list all CUDA devices and exit (no encoding).\n"
"\n"
"-benchmark  --  Do timing and print results to several text files.\n"
"\n"
"-maxcb <n>  --  process up to <n> codeblocks in one kernel call. Use this if Tier 1-kernel fails.\n"
"\n"
"-local  --  use local GPU memory. (default when streaming disabled)\n"
"-global  --  use global GPU memory. (default when streaming enabled)\n"
"\n"
"\n"
"\n"
"************** FILE options **************************\n"
"\n"
"-o <output-filename>  --  specifies the output filename. If omitted, each input filename with extension replaced to .jp2/.j2k is used.\n"
"\n"
"-rev  --  use reversible DWT and color transform (default). For lossless compression.\n"
"\n"
"-irrev  --  use irreversible DWT and color transform. For lossy compression.\n"
"\n"
"-hq  --  disable quantization (only has an effect with -irrev).\n"
"\n"
"-nopcrd  --  disable output filesize control => maximal filesize (default).\n"
"\n"
"-size <n> / <f>K / <f>M  --  set output filesize to <n> bytes / <f> kilobytes / <f> megabytes.\n"
"\n"
"-sizerel <f>  --  set output filesize to <f>*input-filesize, e.g. use 0.4 to set output filesize to 40% of the input filesize.\n"
"\n"
"-ratio <f>  --  set compression ratio to <f>:1, equal to '-sizerel <1/f>'.\n"
"\n"
"-format jp2/j2k  --  set file format to JP2 (full JPEG2000 file format; default) or J2K (codestream-only files, use this if you want to create MJ2 videos)\n"
"\n"
"-cb <n>  --  set codeblock size to <n>. Allowed values: 0 (default), 16, 32, 64. 0 means automatic, depending on image size.\n"
"\n";

	printf("%s", u);
}



///////////////////////////////////////////////
//Nyriad Integration
extern "C"{
    Response NyriadPush(void *src, SentMetadata *sMetadata)
    {
        printf("CuJ2k NyriadPush(%lu)\n", sMetadata->blockSize);
        Response response;
        size_t outSize = 0;
        unsigned char *output;

        size_t size = sMetadata->blockSize;
		output = (unsigned char*)malloc(size);
		unsigned char *src_temp = (unsigned char*)malloc(size);
		memcpy(src_temp, (void*)src, size);

        //default response (original file) if error occurs.
        outSize = sMetadata->blockSize;
        //output = (unsigned char *)src;
        int error = -1;

        response.output = (unsigned char *)malloc(outSize);
        //memcpy(response.output,output,outSize);
        response.outputSize = outSize;
        response.errorCode = error;


        Parameter *temp = sMetadata->parameters;
        printf("Parameter: %s\n", temp->parameter);
        temp = temp->next;
        int parameterCount = 0; //-c./
        while(temp)
        {
            parameterCount ++;
            temp = temp->next;
        }
		parameterCount++;   //put a fake command at first;
		parameterCount++;   //put a fake filename in the end

        temp = sMetadata->parameters;
        temp = temp->next;

        if(size > 0)
        {
            outSize = size;
            if(sMetadata->originalSize > 0 && sMetadata->originalSize != size)
            {
                  printf("use original size: %lu \n", sMetadata->originalSize);
                  outSize = sMetadata->originalSize;
            }
            output = (unsigned char*)malloc(outSize);

            if(strcmp("-c", sMetadata->parameters->parameter) == 0)
            {
                printf("ENCODE\n");
                sMetadata->originalSize = size;
                char *passedParams[parameterCount];
				passedParams[0] = "ignore";
                for(int s = 1; s < (parameterCount-1); s++){
                    passedParams[s] = (char *)temp->parameter;
                    temp = temp->next;
                }
				passedParams[parameterCount-1] = "temp.BMP";

                //printf("Debug from NyriadPush parameterCount: %d\n", parameterCount);
                error = Cuj2kMain((unsigned char*)src_temp, size, &output, &outSize, parameterCount, passedParams);
                //error = NvEncoderMain((unsigned char*)src, size, &output, &outSize, parameterCount, passedParams);
                printf("Error: %u\n", error);
                //printf("outSize from nv = %d\n",outSize);

                if(error!=0){
                    //set to origin file
                    outSize = sMetadata->blockSize;
                    output = (unsigned char *)malloc(sMetadata->blockSize);
                    //output = (unsigned char *)src;
					memcpy(output, (void *)src, outSize);
                }
            }
            else{
            	printf("Request type not supported\n");
                outSize = sMetadata->blockSize;
                output = (unsigned char *)malloc(sMetadata->blockSize);
                //output = (unsigned char *)src;
				memcpy(output, (void *)src, outSize);
            }
        }
        else
        {
			outSize = sMetadata->blockSize;
            output = NULL;
            error = 0;
        }


        response.output = (unsigned char *)malloc(outSize);
        memcpy(response.output,output,outSize);
        response.outputSize = outSize;
        response.errorCode = error;
        free(output);

        return response;

    }

    Response NyriadFlush(SentMetadata *meta)
    {
        printf("CuJ2k NyriadFlush()\n");
        Response response;
        response.errorCode = 0;
        response.output = NULL;
        response.outputSize = 0;
        return response;
    }
}
