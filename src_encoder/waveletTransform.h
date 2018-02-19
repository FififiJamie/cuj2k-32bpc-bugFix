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

#ifndef waveletTransform_h
#define waveletTransform_h
#include "bitmap.h"
#include <math.h>
//#include "driver_types.h"
//#include <cutil.h>
//#include <helper_cuda.h>
#include <cuda_runtime.h>
//#include <helper_math.h>
//#include <cutil_inline.h>



struct Codeblock;

typedef enum {LLsubdiv,LH,HL,HH,LLend} SubbandTyp; //LLend is an approximation of the original image, the other subband are the higher frequency parts
                                                   //LLsubdiv does not contain image data in itselft, but it contains pointers to the higher level subbands



typedef struct ggg{     //Type for output of wavelettransform
        SubbandTyp Typ;
        int Xdim;//width
        int Ydim;//height
		float fl_quantstep; //normal float value for quant.step
        unsigned short int quantstep; //quantizition step in 16bit floating-point-format (ready for QCD/QCC marker)
        int * daten_d; //image data(except in case of LLsubdiv-subband) on Device
		int * daten; //image data(except in case of LLsubdiv-subband)
        struct ggg* subbands[4]; //in case of LLsubdiv-subband: pointer to the 4 sub-subbands

		int K_max; // number of bit planes in this subband (defined after Tier 1 if Typ!=LLsubdiv)
		           // (when considering the coefficients in sign-magnitude form)

        int nCodeblocks, nCBx, nCBy; // number of codeblocks: total, columns, rows

        struct Codeblock *codeblocks; // contains compressed codeblocks after Tier 1
		// codeblocks[x + y*nCBx] is the codeblock @ position (x,y)

		int *K_msbs; // number of skipped bit planes in each codeblock

        }subband;

//in eingang.imgData[i] the pointer to the three Toplevel-Subbands of the channels are stored
int DWT(struct Tile *eingang,  int  max_Level,int line_length, int mode, int quant_enable, cudaStream_t stream, int bps, int* temp_d); 
extern "C" unsigned short int dwt_encode_stepsize(float stepsize, int numbps); //Transformation in exp/mant representation

#endif
