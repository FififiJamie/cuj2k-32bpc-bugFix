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
Rate-Control for JPEG2000

float get_quantstep(int level,int max, SubbandTyp subband, int quant_enable)
returns quantization step for given subband. If !quant_enable, returns
always 1.0f.

unsigned {char,short} * get_standard_qcd[lossless] (....)
Returns the QCD marker by encoding the quantization steps
calculated by get_quantstep.
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "rate-control.h"
#include "waveletTransform.h"
#include "tier1.h"
#include "misc.h"

/*Returns stand QCD sequence as array.
  Order is LL HL LH HH from highest to lowest level*/
unsigned short int* get_standard_qcd (int dwt_levels, int quant_enable) {
	unsigned short int* result;
	int level, counter;
	int bitplanes = 9;
    result = (unsigned short int*) malloc ((1+dwt_levels*3)* sizeof (unsigned short int));
	counter = 1;
	
	//LL is on level5 with highest qualitiy and lowest quantization
	result[0] = dwt_encode_stepsize((float) get_quantstep(dwt_levels,dwt_levels,LLend,quant_enable), bitplanes);
	
	//for (level=dwt_levels;level>0;level--){
	for (level=1;level<=dwt_levels;level++){
		//multiply by 2 or 4??
		//HL
		result[counter] = dwt_encode_stepsize(((float) get_quantstep(level,dwt_levels,HL,quant_enable)), bitplanes);
		counter++;
		//LH
		result[counter] = dwt_encode_stepsize(((float) get_quantstep(level,dwt_levels,LH,quant_enable)), bitplanes);
		counter++;
		//HH
		result[counter] = dwt_encode_stepsize(((float) get_quantstep(level,dwt_levels,HH,quant_enable)), bitplanes);
		counter++;
	}
    
	return result;
}

unsigned char* get_standard_qcd_lossless (int dwt_levels) {
	unsigned char* result;
	int level, counter;
	int bitplanes = 9;
    result = (unsigned char*) malloc ((1+dwt_levels*3)* sizeof (unsigned char));
	
	//LL is on level5 with highest qualitiy and lowest quantization
	result[0]=bitplanes<<3;
    for (counter=0;counter<(dwt_levels);counter++){
        //TODO: find right shiftsize?
        //printf("\n %x", dwt_encode_stepsize(1.0f, bitplanes));
        
        result[counter*3+1] = (bitplanes+1)<<3;
		result[counter*3+2] = (bitplanes+1)<<3;
		result[counter*3+3] = (bitplanes+2)<<3;
        
        //this would equal a 1 in the 16bit interpretation, but grey result 
        //result[counter]=0x9;
    }
	return result;
}



/* calaculates K_b_max, which is the maximum amount of bits needed for the
   decoder to perform the inverse dwt.
   quantstep: 16-bit exp.-mantissa encoded quantization step
   type: type of subband*/
int calc_K_max(unsigned short quantstep, SubbandTyp type, int mode) {
	if(mode==LOSSLESS){
		int X_b;
		if (type==LLend)
			X_b = 0;
		else if (type==LH || type==HL)
			X_b = 1;
		else /*HH*/
			X_b = 2;
			
		/* return B - 1 + X_b + G (where B is the bit depth of the original
			color transformed samples, is 8 for Y, [9 for Cr,Cb];  G=2 (guard bits)) */
		return(10+X_b);
	}else{
	/* K_b_max = max{0, epsilon_b+G-1}, where G=1 and epsilon_b 
	   are bits 15..11 of quantstep */
		return(((quantstep >> 11) & 0x1F) +1); /* need +1 because G=2 */
	}

}


/*Fixed values for the quantization steps
 (Highest level smallest quantization)
 internal function; is called by get_quantstep() below */
float get_quantstep_big(int level,int max, SubbandTyp subband){
	
	/*LL subband always no quantization*/
	if (subband == LLend) {
		return 1;
	}
	
	switch (max-level+1) {
		case 1:
			switch (subband){
				case LH:
					return 2;
				break;
				case HL:
					return 2;
				break;
				case HH:
					return 8;
				break;
				default:
					return 1;					
			}
		break;

		case 2:
			switch (subband){
				case LH:
					return 4;
				break;
				case HL:
					return 4;
				break;
				case HH:
					return 8;
				break;
				default:
					return 1;
			}
		break;

		case 3:
			switch (subband){
				case LH:
					return 16;
				break;
				case HL:
					return 16;
				break;
				case HH:
					return 16;
				break;
				default:
					return 1;
			}
		break;

		case 4: //quantize only on this level
			switch (subband){
				case LH:
					return 64;
				break;
				case HL:
					return 64;
				break;
				case HH:
					return 64;
				break;
				default:
					return 1;
			}
		break;

		//the following are currently never used....
		case 5:
			switch (subband){
				case LH:
					return 64;
				break;
				case HL:
					return 64;
				break;
				case HH:
					return 64;
				break;
				default:
					return 1;
			}
		break;
		
		
		case 6:
			switch (subband){
				case LH:
					return 64;
				break;
				case HL:
					return 64;
				break;
				case HH:
					return 64;
				break;
				default:
					return 1;
			}
		break;
		
		case 7:
			switch (subband){
				case LH:
					return 64;
				break;
				case HL:
					return 64;
				break;
				case HH:
					return 64;
				break;
				default:
					return 1;
			}
		break;
		
		
		
		default:
		return 1;	
	}
    return 1;
}
	
float get_quantstep(int level,int max, SubbandTyp subband, int quant_enable) {
	if(!quant_enable)
		return 1.0f;
	else {
		float step = get_quantstep_big(level,max, subband) / 32.0f;
		if(step < 1.0f)
			return 1.0f;  //step < 1 is nonsense and not encodeable
		else
			return step;
	}
}
	
	
	
	
	
	
	
	
	
	
	
	

