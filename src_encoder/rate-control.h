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

/*  Rate-control corresponded functions.
 *
 *  get_standard_qcd* is needed for tier 2
 *  get_quantstep needed for DWT and qcd
 *  calc_K_max is needed for Tier1/PCRD
 */

#ifndef RateControl_h
#define RateControl_h
#include "bitmap.h"
#include "waveletTransform.h"

//enable_pcrd is now a parameter for tier1_pic

/*Returns stand QCD sequence as array.
  Order is LL HL LH HH from highest to lowest level
  Needed for tier2*/
unsigned short int* get_standard_qcd (int dwt_levels, int quant_enable);
unsigned char* get_standard_qcd_lossless (int dwt_levels, int bps);


/*Function to get Quantization step for a subband.
  PARAMETERS:
  level: level of DWT
  max: max level
  subband: the subband type
  quant_enable: switch to enable quantization*/
extern "C" float get_quantstep(int level,int max, SubbandTyp subband, int quant_enable);


/* calaculates K_b_max, which is the maximum amount of bits needed for the
   decoder to perform the inverse dwt.
   PARAMETERS:
   quantstep: 16-bit exp.-mantissa encoded quantization step
   type: type of subband
   mode: lossy or lossless */
extern "C" int calc_K_max(unsigned short quantstep, SubbandTyp type, int mode, int bps);

#endif
