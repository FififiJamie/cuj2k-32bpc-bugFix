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
Tier 1 kernel and bit plane coding device functions; included by tier1.cu

__global__ void encode_cb_kernel_local/global_mem(...)
Kernel, each thread encodes one codeblock. Uses either local
or global memory.
- Quantizes DWT coefficients (only lossy mode). Conversion to sign-
  magnitude form.
- Does all coding passes
- Terminates the MQ codeword, calculates truncation points and slopes

__device__ int conv_sign_magn_lossy/lossless(....)
- Quantization, if lossy mode
- Conversion 2's complement -> sign-magnitude form
- Returns number of bitplanes that won't be skipped
  (number of lowest bitplanes with at least one 1-bit).
  Returns 1 if there are 0 bitplanes.

__device__ float spp(....)
__device__ float mrp(....)
__device__ float cpp(....)
Perform Significance Propagation Pass / Magnitude Refinement Pass /
Cleanup Pass on specified bit plane. Returns the distortion decrease
achieved by this coding pass.
*/




#define STRIPE_HEIGHT 4

//offsets for moving around in state array;
//L R U D = left right up down
#define S_SCANW (CB_MAX_XDIM+2)
#define s_scanw S_SCANW

//index of (0,0) position in state array
#define S_START (S_SCANW+1)
#define S_OFFSET_L  (-1)
#define S_OFFSET_R  (+1)
#define S_OFFSET_U  (-S_SCANW)
#define S_OFFSET_D  (+S_SCANW)
#define S_OFFSET_UL (-S_SCANW-1)
#define S_OFFSET_UR (-S_SCANW+1)
#define S_OFFSET_DL (+S_SCANW-1)
#define S_OFFSET_DR (+S_SCANW+1)


//how many bits to shift to get the sign bit
#define SHIFT_COEFF_SIGN 31
#define MASK_COEFF_SIGN 0x80000000u
#define MASK_COEFF_MAGN 0x7FFFFFFFu

// ****************** Bit access for state array ****************
#define ROW_SEP 16

// bit-region 1 (y is even)
#define MASK_HAS_SIGNIF_NEIGH_R1 (1u << 15)
#define MASK_REFINED_R1 (1u << 14)
#define MASK_SPP_CODED_R1 (1u << 13)
#define MASK_SIGNIF_R1 (1u << 12)

#define MASK_SIGN_L_R1 (1u << 11)
#define MASK_SIGN_R_R1 (1u << 10)
#define MASK_SIGN_U_R1 (1u << 9)
#define MASK_SIGN_D_R1 (1u << 8)

#define MASK_SN_L_R1 (1u << 7)
#define MASK_SN_R_R1 (1u << 6)
#define MASK_SN_U_R1 (1u << 5)
#define MASK_SN_D_R1 (1u << 4)
#define MASK_SN_UL_R1 (1u << 3)
#define MASK_SN_UR_R1 (1u << 2)
#define MASK_SN_DL_R1 (1u << 1)
#define MASK_SN_DR_R1 (1u << 0)

#define SHIFT_K_SIGN_R1 4
#define MASK_K_SIGN_R1 (0xFF0u)
#define SHIFT_K_SIG_R1 0
#define MASK_K_SIG_R1 (0xFFu)
#define SHIFT_K_MAG_R1 14
#define MASK_K_MAG_R1 (0xC000u)


// bit-region 2 (y is odd)
#define MASK_HAS_SIGNIF_NEIGH_R2 (MASK_HAS_SIGNIF_NEIGH_R1 << ROW_SEP)
#define MASK_REFINED_R2 (MASK_REFINED_R1 << ROW_SEP)
#define MASK_SPP_CODED_R2 (MASK_SPP_CODED_R1 << ROW_SEP)
#define MASK_SIGNIF_R2 (MASK_SIGNIF_R1 << ROW_SEP)

#define MASK_SIGN_L_R2 (MASK_SIGN_L_R1 << ROW_SEP)
#define MASK_SIGN_R_R2 (MASK_SIGN_R_R1 << ROW_SEP)
#define MASK_SIGN_U_R2 (MASK_SIGN_U_R1 << ROW_SEP)
#define MASK_SIGN_D_R2 (MASK_SIGN_D_R1 << ROW_SEP)

#define MASK_SN_L_R2 (MASK_SN_L_R1 << ROW_SEP)
#define MASK_SN_R_R2 (MASK_SN_R_R1 << ROW_SEP)
#define MASK_SN_U_R2 (MASK_SN_U_R1 << ROW_SEP)
#define MASK_SN_D_R2 (MASK_SN_D_R1 << ROW_SEP)
#define MASK_SN_UL_R2 (MASK_SN_UL_R1 << ROW_SEP)
#define MASK_SN_UR_R2 (MASK_SN_UR_R1 << ROW_SEP)
#define MASK_SN_DL_R2 (MASK_SN_DL_R1 << ROW_SEP)
#define MASK_SN_DR_R2 (MASK_SN_DR_R1 << ROW_SEP)

#define SHIFT_K_SIGN_R2 (SHIFT_K_SIGN_R1 + ROW_SEP)
#define MASK_K_SIGN_R2 (MASK_K_SIGN_R1 << ROW_SEP)
#define SHIFT_K_SIG_R2 (SHIFT_K_SIG_R1 + ROW_SEP)
#define MASK_K_SIG_R2 (MASK_K_SIG_R1 << ROW_SEP)
#define SHIFT_K_MAG_R2 (SHIFT_K_MAG_R1 + ROW_SEP)
#define MASK_K_MAG_R2 (MASK_K_MAG_R1 << ROW_SEP)


//some contexts for mq coder
#define K_RUN 9
#define K_UNI 18

//convert sign-magnitude format to normal 2's complement
__device__ int to_compl(sign_magn_t x) {
	int r = (int)(x & MASK_COEFF_MAGN);
	if(x & MASK_COEFF_SIGN)
		r = (-r);
	return r;
}



//#define encode_sign(id,cb,sign,lut_i) (MQenc_Encode((id), (cb), k_sign[(lut_i)], (sign) ^ sign_flip[(lut_i)]))

__device__ void encode_sign(int id, unsigned char *B, int *L, int sign, int lut_i) {
#ifdef _DEVICEEMU
	int s = (sign);
	my_dev_assert(s==0 || s==1);
	my_dev_assert(lut_i>=0 && lut_i<256);
	s = (sign) ^ sign_flip[(lut_i)];
	my_dev_assert(s==0 || s==1);
#endif
	MQenc_Encode((id), B, L, k_sign[(lut_i)], (sign) ^ sign_flip[(lut_i)]);
}

// returns the distortion decrease
__device__ float spp(sign_magn_t *v,
					 int v_scanw,
					 unsigned *states,
	 	             int id, //of MQ-Coder
					 char *k_sig,
	 	             unsigned char *B,
					 int *L,
					 int w,
					 int h,
	                 int bitplane)
{
	//s_i: index in state array
	int s_i, x, y, stripe_h = STRIPE_HEIGHT;
	// respect offset caused by padding on boundaries
	unsigned s1,s2; // current state
	//mean square error decrease per coded sample
	float d = square((float)(1<<bitplane));
	//distortion decrease; only changes when a '1' is encoded
	float dist_dec = 0.0f;

	// loop:
	// for stripe
	//     for x
	//         for y-pos in stripe (no loop, but hard-coded)
	for(y = 0; y < h; y += STRIPE_HEIGHT) {
		if(y + STRIPE_HEIGHT > h)
			stripe_h = h - y;
		s_i = (y>>1)/*y/2*/ * s_scanw + S_START; //states index
		for(x = 0; x < w; x++) {
			// 1st position in column
			s1 = states[s_i];
			if((s1&MASK_SIGNIF_R1)==0u && (s1&MASK_HAS_SIGNIF_NEIGH_R1)!=0u) {
				sign_magn_t coeff = v[x + v_scanw*y];
				int bit = (coeff>>bitplane) & 1;
				MQenc_Encode(id, B, L, k_sig[(s1&MASK_K_SIG_R1) >> SHIFT_K_SIG_R1], bit);
				if(bit == 1) {
					int sign = coeff >> SHIFT_COEFF_SIGN;
					//propagate significance to the 7 other neighbors
					states[s_i + S_OFFSET_UL] |= MASK_SN_DR_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
					states[s_i + S_OFFSET_UR] |= MASK_SN_DL_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
					//update X^
					if(sign) { //if negative => set X^ of the 4 neighbors to 1
						//make significant + propagate to bottom neighbor
						s1 |= MASK_SIGNIF_R1 | MASK_SN_U_R2 | MASK_HAS_SIGNIF_NEIGH_R2 |
						      MASK_SIGN_U_R2 | MASK_SPP_CODED_R1;
						states[s_i-1] |= MASK_SN_R_R1 | MASK_SN_UR_R2 | 
							             MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2 |
										 MASK_SIGN_R_R1;
						states[s_i+1] |= MASK_SN_L_R1 | MASK_SN_UL_R2 |
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2 |
										 MASK_SIGN_L_R1;
						states[s_i + S_OFFSET_U] |= MASK_SN_D_R2 | MASK_HAS_SIGNIF_NEIGH_R2 |
						                         MASK_SIGN_D_R2;
					}
					else {
						//make significant + propagate to bottom neighbor
						s1 |= MASK_SIGNIF_R1 | MASK_SN_U_R2 | MASK_HAS_SIGNIF_NEIGH_R2 |
							  MASK_SPP_CODED_R1;
						states[s_i-1] |= MASK_SN_R_R1 | MASK_SN_UR_R2 | 
							             MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i+1] |= MASK_SN_L_R1 | MASK_SN_UL_R2 |
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i + S_OFFSET_U] |= MASK_SN_D_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
					}
					encode_sign(id, B, L, sign, (s1&MASK_K_SIGN_R1) >> SHIFT_K_SIGN_R1);
					dist_dec += d;
				}
				else
					s1 |= MASK_SPP_CODED_R1;
			}
			else
				s1 &= (~MASK_SPP_CODED_R1);

			if(stripe_h < 2) { //shorter stripe (on the bottom)
				states[s_i++] = s1;
				continue;
			}

			s2 = states[s_i+S_OFFSET_D];

			//2nd position in column; value of s1 is still valid
			if((s1&MASK_SIGNIF_R2)==0u && (s1&MASK_HAS_SIGNIF_NEIGH_R2)!=0u) {
				sign_magn_t coeff = v[x + v_scanw*(y+1)];
				int bit = (coeff>>bitplane) & 1;
				MQenc_Encode(id, B, L, k_sig[(s1&MASK_K_SIG_R2) >> SHIFT_K_SIG_R2], bit);
				if(bit == 1) {
					int sign = coeff >> SHIFT_COEFF_SIGN;
					//propagate significance to the 7 other neighbors
					states[s_i + S_OFFSET_DL] |= MASK_SN_UR_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
					states[s_i + S_OFFSET_DR] |= MASK_SN_UL_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
					//update X^
					if(sign) { //if negative => set X^ of the 4 neighbors to 1
						//make significant + propagate to upper neighbor
						s1 |= MASK_SIGNIF_R2 | MASK_SN_D_R1 | MASK_HAS_SIGNIF_NEIGH_R1 |
						      MASK_SIGN_D_R1 | MASK_SPP_CODED_R2;
						states[s_i-1] |= MASK_SN_DR_R1 | MASK_SN_R_R2 | 
							             MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2 |
						                 MASK_SIGN_R_R2;
						states[s_i+1] |= MASK_SN_DL_R1 | MASK_SN_L_R2 |
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2 |
						                 MASK_SIGN_L_R2;
						s2 |= MASK_SN_U_R1 | MASK_HAS_SIGNIF_NEIGH_R1 |
						                         MASK_SIGN_U_R1;
					}
					else {
						//make significant + propagate to upper neighbor
						s1 |= MASK_SIGNIF_R2 | MASK_SN_D_R1 | MASK_HAS_SIGNIF_NEIGH_R1 |
						      MASK_SPP_CODED_R2;
						states[s_i-1] |= MASK_SN_DR_R1 | MASK_SN_R_R2 | 
							             MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i+1] |= MASK_SN_DL_R1 | MASK_SN_L_R2 |
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						s2 |= MASK_SN_U_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
					}
					encode_sign(id, B, L, sign, (s1&MASK_K_SIGN_R2) >> SHIFT_K_SIGN_R2);
					dist_dec += d;
				}
				else
					s1 |= MASK_SPP_CODED_R2;
			}
			else
				s1 &= (~MASK_SPP_CODED_R2);
			//states[s_i] = s; //next state word => save to array

			if(stripe_h < 3) { //shorter stripe (on the bottom)
				states[s_i+S_OFFSET_D] = s2;
				states[s_i++] = s1;
				continue;
			}

			s_i += S_OFFSET_D;

			// 3rd position in column
			//s = states[s_i];
			if((s2&MASK_SIGNIF_R1)==0u && (s2&MASK_HAS_SIGNIF_NEIGH_R1)!=0u) {
				sign_magn_t coeff = v[x + v_scanw*(y+2)];
				int bit = (coeff>>bitplane) & 1;
				MQenc_Encode(id, B, L, k_sig[(s2&MASK_K_SIG_R1) >> SHIFT_K_SIG_R1], bit);
				if(bit == 1) {
					int sign = coeff >> SHIFT_COEFF_SIGN;
					//propagate significance to the 7 other neighbors
					states[s_i + S_OFFSET_UL] |= MASK_SN_DR_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
					states[s_i + S_OFFSET_UR] |= MASK_SN_DL_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
					//update X^
					if(sign) { //if negative => set X^ of the 4 neighbors to 1
						//make significant + propagate to bottom neighbor
						s2 |= MASK_SIGNIF_R1 | MASK_SN_U_R2 | MASK_HAS_SIGNIF_NEIGH_R2 |
						      MASK_SIGN_U_R2 | MASK_SPP_CODED_R1;
						states[s_i-1] |= MASK_SN_R_R1 | MASK_SN_UR_R2 | 
							             MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2 |
						                 MASK_SIGN_R_R1;
						states[s_i+1] |= MASK_SN_L_R1 | MASK_SN_UL_R2 |
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2 |
						                 MASK_SIGN_L_R1;
						s1 |= MASK_SN_D_R2 | MASK_HAS_SIGNIF_NEIGH_R2 |
						                         MASK_SIGN_D_R2;
					}
					else {
						//make significant + propagate to bottom neighbor
						s2 |= MASK_SIGNIF_R1 | MASK_SN_U_R2 | MASK_HAS_SIGNIF_NEIGH_R2 |
							  MASK_SPP_CODED_R1;
						states[s_i-1] |= MASK_SN_R_R1 | MASK_SN_UR_R2 | 
							             MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i+1] |= MASK_SN_L_R1 | MASK_SN_UL_R2 |
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						s1 |= MASK_SN_D_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
					}
					encode_sign(id, B, L, sign, (s2&MASK_K_SIGN_R1) >> SHIFT_K_SIGN_R1);
					dist_dec += d;
				}
				else
					s2 |= MASK_SPP_CODED_R1;
			}
			else
				s2 &= (~MASK_SPP_CODED_R1);

			states[s_i+S_OFFSET_U] = s1; //needed no more => write back

			if(stripe_h < 4) { //shorter stripe (on the bottom)
				states[s_i] = s2;
				s_i += S_OFFSET_UR;
				continue;
			}
			//4th position in column; value of s is still valid
			if((s2&MASK_SIGNIF_R2)==0u && (s2&MASK_HAS_SIGNIF_NEIGH_R2)!=0u) {
				sign_magn_t coeff = v[x + v_scanw*(y+3)];
				int bit = (coeff>>bitplane) & 1;
				MQenc_Encode(id, B, L, k_sig[(s2&MASK_K_SIG_R2) >> SHIFT_K_SIG_R2], bit);
				if(bit == 1) {
					int sign = coeff >> SHIFT_COEFF_SIGN;
					//propagate significance to the 7 other neighbors
					states[s_i + S_OFFSET_DL] |= MASK_SN_UR_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
					states[s_i + S_OFFSET_DR] |= MASK_SN_UL_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
					//update X^
					if(sign) { //if negative => set X^ of the 4 neighbors to 1
						//make significant + propagate to upper neighbor
						s2 |= MASK_SIGNIF_R2 | MASK_SN_D_R1 | MASK_HAS_SIGNIF_NEIGH_R1 |
						      MASK_SIGN_D_R1 | MASK_SPP_CODED_R2;
						states[s_i-1] |= MASK_SN_DR_R1 | MASK_SN_R_R2 | 
							             MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2 |
						                 MASK_SIGN_R_R2;
						states[s_i+1] |= MASK_SN_DL_R1 | MASK_SN_L_R2 |
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2 |
						                 MASK_SIGN_L_R2;
						states[s_i + S_OFFSET_D] |= MASK_SN_U_R1 | MASK_HAS_SIGNIF_NEIGH_R1 |
						                         MASK_SIGN_U_R1;
					}
					else {
						//make significant + propagate to upper neighbor
						s2 |= MASK_SIGNIF_R2 | MASK_SN_D_R1 | MASK_HAS_SIGNIF_NEIGH_R1 |
							  MASK_SPP_CODED_R2;
						states[s_i-1] |= MASK_SN_DR_R1 | MASK_SN_R_R2 | 
							             MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i+1] |= MASK_SN_DL_R1 | MASK_SN_L_R2 |
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i + S_OFFSET_D] |= MASK_SN_U_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
					}
					encode_sign(id, B, L, sign, (s2&MASK_K_SIGN_R2) >> SHIFT_K_SIGN_R2);
					dist_dec += d;
				}
				else
					s2 |= MASK_SPP_CODED_R2;
			}
			else
				s2 &= (~MASK_SPP_CODED_R2);
			states[s_i] = s2; //next state word => save to array
			s_i += S_OFFSET_UR; //next stripe column => move up-right
		}
	}
	return dist_dec;
}





// returns the distortion decrease
__device__ float mrp(sign_magn_t *v,
					 int v_scanw,
					 unsigned *states,
	 	             int id, //of MQ-Coder
	 	             unsigned char *B,
					 int *L,
					 int w,
					 int h,
	                 int bitplane)
{
	//s_i: index in state array
	int v_i, s_i, x, y, stripe_h = STRIPE_HEIGHT;
	unsigned s; // current state
	int v_offset_next_col = 1 - 3*v_scanw;
	//mean square error decrease per coded sample
	float d = square((float)(1<<bitplane));
	//distortion decrease; only changes when a '1' is encoded
	float dist_dec = 0.0f;
	//int s_scanw = w+2;

	// loop:
	// for stripe
	//     for x
	//         for y-pos in stripe (no loop, but hard-coded)
	for(y = 0; y < h; y += STRIPE_HEIGHT) {
		if(y + stripe_h > h)
			stripe_h = h - y;
		s_i = (y>>1)/*y/2*/ * s_scanw + S_START; //states index
		v_i = y * v_scanw;
		for(x = 0; x < w; x++) {
			// 1st position in column
			s = states[s_i];
			// if significant, but not coded in SPP
			if( (s & (MASK_SIGNIF_R1|MASK_SPP_CODED_R1)) == MASK_SIGNIF_R1 ) {
				MQenc_Encode(id, B, L, k_mag[(s&MASK_K_MAG_R1)>>SHIFT_K_MAG_R1],
					(v[v_i] >> bitplane) & 1);
				s |= MASK_REFINED_R1;
				dist_dec += d;
			}
			if(stripe_h < 2) {
				states[s_i++] = s;
				v_i++;
				continue;
			}
			v_i += v_scanw;

			// 2nd position in column (s still valid)
			// if significant, but not coded in SPP
			if( (s & (MASK_SIGNIF_R2|MASK_SPP_CODED_R2)) == MASK_SIGNIF_R2 ) {
				MQenc_Encode(id, B, L, k_mag[(s&MASK_K_MAG_R2)>>SHIFT_K_MAG_R2],
					(v[v_i] >> bitplane) & 1);
				s |= MASK_REFINED_R2;
				dist_dec += d;
			}
			states[s_i] = s; //write-back state word
			if(stripe_h < 3) {
				s_i++;
				v_i += 1 - v_scanw;
				continue;
			}
			s_i += S_OFFSET_D;
			v_i += v_scanw;

			// 3rd position in column
			s = states[s_i];
			// if significant, but not coded in SPP
			if( (s & (MASK_SIGNIF_R1|MASK_SPP_CODED_R1)) == MASK_SIGNIF_R1 ) {
				MQenc_Encode(id, B, L, k_mag[(s&MASK_K_MAG_R1)>>SHIFT_K_MAG_R1],
					(v[v_i] >> bitplane) & 1);
				s |= MASK_REFINED_R1;
				dist_dec += d;
			}
			if(stripe_h < 4) {
				states[s_i] = s;
				s_i += S_OFFSET_UR;
				v_i += 1 - 2*v_scanw;
				continue;
			}
			v_i += v_scanw;

			// 4th position in column (s still valid)
			// if significant, but not coded in SPP
			if( (s & (MASK_SIGNIF_R2|MASK_SPP_CODED_R2)) == MASK_SIGNIF_R2 ) {
				MQenc_Encode(id, B, L, k_mag[(s&MASK_K_MAG_R2)>>SHIFT_K_MAG_R2],
					(v[v_i] >> bitplane) & 1);
				s |= MASK_REFINED_R2;
				dist_dec += d;
			}
			states[s_i] = s; //write-back state word
			s_i += S_OFFSET_UR; //next stripe column
			v_i += v_offset_next_col;
		}
	}
	return dist_dec;
}




__device__ float cup(sign_magn_t *v,
					 int v_scanw,
					 unsigned *states,
	 	             int id, //of MQ-Coder
					 char *k_sig,
	 	             unsigned char *B,
					 int *L,
					 int w,
					 int h,
	                 int bitplane)
{
	//s_i: index in state array
	int s_i, x, y, stripe_h = STRIPE_HEIGHT;
	//mean square error decrease per coded sample
	float d = square((float)(1<<bitplane));
	//distortion decrease; only changes when a '1' is encoded
	float dist_dec = 0.0f;
	unsigned s1,s2;
	//int s_scanw = w+2;

	// loop:
	// for stripe
	//     for x
	//         for y-pos in stripe (no loop, but hard-coded)
	for(y = 0; y < h; y += STRIPE_HEIGHT) {
		if(y + STRIPE_HEIGHT > h)
			stripe_h = h - y;
		s_i = (y>>1)/*y/2*/ * s_scanw + S_START; //states index
		for(x = 0; x < w; x++) {
			//rl_coded==1 if RL coding was the last operation in this column
			int r=0, rl_coded=0, bit;
			// run-length coding only on full stripe columns
			// which only have insignificant neighbors

			s1 = states[s_i           ];
			s2 = states[s_i+S_OFFSET_D];
			if((stripe_h == STRIPE_HEIGHT) &&
			   ((s1 & (MASK_HAS_SIGNIF_NEIGH_R1|MASK_HAS_SIGNIF_NEIGH_R2))==0u) &&
			   ((s2 & (MASK_HAS_SIGNIF_NEIGH_R1|MASK_HAS_SIGNIF_NEIGH_R2))==0u))
			{
				while(r<4 && (v[x + v_scanw*(y+r)] & (1<<bitplane))==0)
					r++;
				if(r==4) {
					MQenc_Encode(id, B, L, K_RUN, 0);
					s_i++;
					continue; //no 1-bit => done for this column
				}
				else {
					MQenc_Encode(id, B, L, K_RUN, 1);
					MQenc_Encode(id, B, L, K_UNI, r>>1); //bit 1
					MQenc_Encode(id, B, L, K_UNI, r&1);  //bit 0
					rl_coded = bit = 1;
				}
			}

			if(r==0) {
				if((s1&(MASK_SIGNIF_R1|MASK_SPP_CODED_R1)) == 0) {
					//zero coding if no RL coding used
					if(!rl_coded) {
						bit = (v[x + v_scanw*y] >> bitplane) & 1;
						MQenc_Encode(id, B, L, k_sig[(s1&MASK_K_SIG_R1)>>SHIFT_K_SIG_R1], bit);
					}
					else
						rl_coded = 0;
					if(bit) {
						int sign = v[x + v_scanw*y] >> SHIFT_COEFF_SIGN;
						//make significant + propagate to bottom neighbor
						s1 |= MASK_SIGNIF_R1 | MASK_SN_U_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
						//propagate significance to the 7 other neighbors
						states[s_i-1] |= MASK_SN_R_R1 | MASK_SN_UR_R2 | 
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i+1] |= MASK_SN_L_R1 | MASK_SN_UL_R2 |
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i + S_OFFSET_U]  |= MASK_SN_D_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i + S_OFFSET_UL] |= MASK_SN_DR_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i + S_OFFSET_UR] |= MASK_SN_DL_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
						//update X^
						if(sign) { //if negative => set X^ of the 4 neighbors to 1
							states[s_i-1] |= MASK_SIGN_R_R1;
							states[s_i+1] |= MASK_SIGN_L_R1;
							states[s_i + S_OFFSET_U] |= MASK_SIGN_D_R2;
							s1 |= MASK_SIGN_U_R2;
						}
						encode_sign(id, B, L, sign, (s1&MASK_K_SIGN_R1) >> SHIFT_K_SIGN_R1);
						dist_dec += d;
					}
				}
				r++;
			}

			if(stripe_h < 2) {
				states[s_i++] = s1;
				continue;
			}
			if(r==1) {
				if((s1&(MASK_SIGNIF_R2|MASK_SPP_CODED_R2)) == 0) {
					//zero coding if no RL coding used
					if(!rl_coded) {
						bit = (v[x + v_scanw*(y+1)] >> bitplane) & 1;
						MQenc_Encode(id, B, L, k_sig[(s1&MASK_K_SIG_R2)>>SHIFT_K_SIG_R2], bit);
					}
					else
						rl_coded = 0;
					if(bit) {
						int sign = v[x + v_scanw*(y+1)] >> SHIFT_COEFF_SIGN;
						//make significant + propagate to upper neighbor
						s1 |= MASK_SIGNIF_R2 | MASK_SN_D_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
						//propagate significance to the 7 other neighbors
						states[s_i-1] |= MASK_SN_DR_R1 | MASK_SN_R_R2 | 
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i+1] |= MASK_SN_DL_R1 | MASK_SN_L_R2 |
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						s2  |= MASK_SN_U_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
						states[s_i + S_OFFSET_DL] |= MASK_SN_UR_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
						states[s_i + S_OFFSET_DR] |= MASK_SN_UL_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
						//update X^
						if(sign) { //if negative => set X^ of the 4 neighbors to 1
							states[s_i-1] |= MASK_SIGN_R_R2;
							states[s_i+1] |= MASK_SIGN_L_R2;
							s2 |= MASK_SIGN_U_R1;
							s1 |= MASK_SIGN_D_R1;
						}
						encode_sign(id, B, L, sign, (s1&MASK_K_SIGN_R2) >> SHIFT_K_SIGN_R2);
						dist_dec += d;
					}
				}
				r++;
			}

			if(stripe_h < 3) {
				states[s_i + S_OFFSET_D] = s2; 
				states[s_i++] = s1;
				continue;
			}
			s_i += S_OFFSET_D;
			if(r==2) {
				if((s2&(MASK_SIGNIF_R1|MASK_SPP_CODED_R1)) == 0) {
					//zero coding if no RL coding used
					if(!rl_coded) {
						bit = (v[x + v_scanw*(y+2)] >> bitplane) & 1;
						MQenc_Encode(id, B, L, k_sig[(s2&MASK_K_SIG_R1)>>SHIFT_K_SIG_R1], bit);
					}
					else
						rl_coded = 0;
					if(bit) {
						int sign = v[x + v_scanw*(y+2)] >> SHIFT_COEFF_SIGN;
						//make significant + propagate to bottom neighbor
						s2 |= MASK_SIGNIF_R1 | MASK_SN_U_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
						//propagate significance to the 7 other neighbors
						states[s_i-1] |= MASK_SN_R_R1 | MASK_SN_UR_R2 | 
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i+1] |= MASK_SN_L_R1 | MASK_SN_UL_R2 |
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						s1 |= MASK_SN_D_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i + S_OFFSET_UL] |= MASK_SN_DR_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i + S_OFFSET_UR] |= MASK_SN_DL_R2 | MASK_HAS_SIGNIF_NEIGH_R2;
						//update X^
						if(sign) { //if negative => set X^ of the 4 neighbors to 1
							states[s_i-1] |= MASK_SIGN_R_R1;
							states[s_i+1] |= MASK_SIGN_L_R1;
							s1 |= MASK_SIGN_D_R2;
							s2 |= MASK_SIGN_U_R2;
						}
						encode_sign(id, B, L, sign, (s2&MASK_K_SIGN_R1) >> SHIFT_K_SIGN_R1);
						dist_dec += d;
					}
				}
				r++;
			}

			states[s_i + S_OFFSET_U] = s1; //needed no more

			if(stripe_h < 4) {
				states[s_i] = s2; 	
				s_i += S_OFFSET_UR;
				continue;
			}
			if(r==3) {
				if((s2&(MASK_SIGNIF_R2|MASK_SPP_CODED_R2)) == 0) {
					//zero coding if no RL coding used
					if(!rl_coded) {
						bit = (v[x + v_scanw*(y+3)] >> bitplane) & 1;
						MQenc_Encode(id, B, L, k_sig[(s2&MASK_K_SIG_R2)>>SHIFT_K_SIG_R2], bit);
					}
					if(bit) {
						int sign = v[x + v_scanw*(y+3)] >> SHIFT_COEFF_SIGN;
						//make significant + propagate to upper neighbor
						s2 |= MASK_SIGNIF_R2 | MASK_SN_D_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
						//propagate significance to the 7 other neighbors
						states[s_i-1] |= MASK_SN_DR_R1 | MASK_SN_R_R2 | 
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i+1] |= MASK_SN_DL_R1 | MASK_SN_L_R2 |
										 MASK_HAS_SIGNIF_NEIGH_R1 | MASK_HAS_SIGNIF_NEIGH_R2;
						states[s_i + S_OFFSET_D] |= MASK_SN_U_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
						states[s_i + S_OFFSET_DL] |= MASK_SN_UR_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
						states[s_i + S_OFFSET_DR] |= MASK_SN_UL_R1 | MASK_HAS_SIGNIF_NEIGH_R1;
						//update X^
						if(sign) { //if negative => set X^ of the 4 neighbors to 1
							states[s_i-1] |= MASK_SIGN_R_R2;
							states[s_i+1] |= MASK_SIGN_L_R2;
							states[s_i + S_OFFSET_D] |= MASK_SIGN_U_R1;
							s2 |= MASK_SIGN_D_R1;
						}
						encode_sign(id, B, L, sign, (s2&MASK_K_SIGN_R2) >> SHIFT_K_SIGN_R2);
						dist_dec += d;
					}
				}
			}
			states[s_i] = s2; 
			s_i += S_OFFSET_UR;
		}
	}
	return dist_dec;
}






// in-place converting from 2's complement to sign-magnitude form
// reversible: int -> int, no quantization
// irreversible: float -> quantizazion -> int
//RETURNS: #bitplanes
__device__ int conv_sign_magn_lossless(sign_magn_t *v, //__global__
									   int w, int h, 
									   int scanw) 
{
	int co;
	int x,y,bitplanes;
	int skipw = scanw - w; //coeffs to skip after each line
	int maximum = 0;

	for(y = 0; y < h; y++) {
		for(x = 0; x < w; x++) 
		{
			co = *((int*)v);
			//printf("%d,", co);

			if(co < 0) {
				co = abs(co);
				maximum |= co;
				*(v++) /*v[i++]*/ = (sign_magn_t)co | MASK_COEFF_SIGN;
			}
			else {
				maximum |= co;
				*(v++) /*v[i++]*/ = (sign_magn_t)co;
			}
		}
		v += skipw;
	}
	//printf("\n");

	bitplanes = 0;
	while(maximum > 0) {
		bitplanes++;
		maximum >>= 1;
	}
	/* if 0 bitplanes => pretend to have 1 bitplane, doesn't matter */
	return(max(1, bitplanes));
}


__device__ int conv_sign_magn_lossy(sign_magn_t *v, //__global__
									int w, int h, 
									int scanw, 
									float quantstep) 
{
	int co;
	int x,y,bitplanes;
	int skipw = scanw - w; //coeffs to skip after each line
	int maximum = 0;

	for(y = 0; y < h; y++) {
		for(x = 0; x < w; x++) 
		{
			co = (int)(*((float*)v) / quantstep);
			//printf("%d,", co);

			if(co < 0) {
				co = abs(co);
				maximum |= co;
				*(v++) /*v[i++]*/ = (sign_magn_t)co | MASK_COEFF_SIGN;
			}
			else {
				maximum |= co;
				*(v++) /*v[i++]*/ = (sign_magn_t)co;
			}
		}
		v += skipw;
	}
	//printf("\n");

	bitplanes = 0;
	while(maximum > 0) {
		bitplanes++;
		maximum >>= 1;
	}
	/* if 0 bitplanes => pretend to have 1 bitplane, doesn't matter */
	return(max(1, bitplanes));
}


#define cb_idx (threadIdx.x + blockIdx.x*blockDim.x)


// Uses global memory => suitable for streaming
__global__ void encode_cb_kernel_global_mem
	(struct Raw_CB *rawcbs, 
     struct Codeblock *cbs, 
     int n_cbs,
	 float *slope_max,
	 int bmp_width,
	 int mode,
	 int enable_pcrd,
	 unsigned char *global_buf_d, unsigned char *global_buf_h, int buffer_size,
	 unsigned *global_state_arr_d,
	 float *global_dist_d, uint4 *global_saved_states_d) 
{
	//int cb_i = threadIdx.x + blockIdx.x*blockDim.x;
	//int id = threadIdx.x;
	//int i;
	struct Codeblock *cb = &(cbs[cb_idx]);
	struct Raw_CB *rawcb = &(rawcbs[cb_idx]);

	float tot_dist = 0.0f;
	/*int trunc_len[MAX_PASSES+1];
	float dist[MAX_PASSES+1];
	uint4 saved_states[MAX_PASSES+1];*/
	//int *trunc_len =      global_trunc_len_d +    cb_idx*PCRD_TEMP_ARRAY_SIZE;
	float *dist =         global_dist_d +         cb_idx*PCRD_TEMP_ARRAY_SIZE;
	uint4 *saved_states = global_saved_states_d + cb_idx*PCRD_TEMP_ARRAY_SIZE;
	//TODO: test if define macros are faster^^^^

	if(cb_idx < n_cbs) {
		int codingpass, p;
		//sign_magn_t *v = rawcb->v;
		//int w = rawcb->w, h = rawcb->h;
		//int v_scanw = rawcb->scanw;
		unsigned *states = global_state_arr_d + cb_idx*MAX_STATE_ARRAY_SIZE;
		//unsigned states[MAX_STATE_ARRAY_SIZE];

		char *k_sig;
		//unsigned char *buf_B = cb->B;
		int buf_L;

		dist[0] = tot_dist;
		cb->trunc_len[0] = 0;

		//printf("conv\n");
		//convert to sign-magnitude, plus quantization if irreversible
		if(mode == LOSSLESS)
			cb->nBitplanes = conv_sign_magn_lossless
				(rawcb->v, rawcb->w, rawcb->h, bmp_width);
		else
			cb->nBitplanes = conv_sign_magn_lossy
				(rawcb->v, rawcb->w, rawcb->h, bmp_width, rawcb->subb_quantstep);

		//cb->nBitplanes = quant_and_conv_sign_magn(rawcb->v, rawcb->w, rawcb->h, bmp_width, rawcb->subb_quantstep);

		if(rawcb->subbType == HH)
			k_sig = k_sig_hh;
		else if(rawcb->subbType == HL)
			k_sig = k_sig_hl;
		else
			k_sig = k_sig_lx;

#ifdef PRINT_COEFF
		{
			int x,y;
			printf("CB #%d\n", cb_idx);
			for(y=0; y < rawcb->h; y++) {
				for(x=0; x < rawcb->w; x++) {
					printf("%4d,", to_compl(rawcb->v[x + y*bmp_width]));
				}
				printf("\n");
			}
			printf("\n");
		}
#endif

		cb->Xdim = rawcb->w;
		cb->Ydim = rawcb->h;
		cb->nCodingPasses = cb->nBitplanes*3 - 2; //bitplanes is always >= 1
		cb->B_d = global_buf_d +  cb_idx * buffer_size;
		cb->B_h = global_buf_h +  cb_idx * buffer_size;
		//printf("%d bitplanes\n", cb->nBitplanes);

		//printf("states init\n");
		//for(p = rawcb->w+2; p < ((rawcb->h+1)/2+1)*(rawcb->w+2); p++)
		for(p = S_SCANW; p < ((rawcb->h+1)/2+1)*S_SCANW; p++)
			states[p] = 0u;

		//printf("mqenc init\n");
		/* init MQ coder */
		MQenc_Init(threadIdx.x);
		buf_L = 0;

		uint4 *mqenc_reg = (uint4*)(s_data + threadIdx.x*ALIGNED_MQENC_SIZE);

		//printf("coding\n");
		/* ************** do encoding ************** */
		codingpass = 0;
		//check_states(rawcb, bpc);
		for(p = cb->nBitplanes-1; p >= 0; p--) {
			//like this, the file compiles about 5 times faster (compared to
			//separate handling of the MSB). it even seems to run faster?!
			if(p != cb->nBitplanes-1) {
#ifdef PRINT_CX_D
				printf("bitplane %d: spp ", p);
#endif
				//printf("spp\n");
				tot_dist -= spp(rawcb->v, bmp_width, states, threadIdx.x, k_sig, cb->B_d, &buf_L, 
					rawcb->w, rawcb->h, p);
				dist      [++codingpass] = tot_dist;
				cb->trunc_len   [codingpass] = buf_L;
				saved_states[codingpass] = *mqenc_reg;

#ifdef PRINT_CX_D
				printf("mrp ");
#endif
				//printf("mrp\n");
				tot_dist -= mrp(rawcb->v, bmp_width, states, threadIdx.x, cb->B_d, &buf_L, 
					rawcb->w, rawcb->h, p);
				dist      [++codingpass] = tot_dist;
				cb->trunc_len   [codingpass] = buf_L;
				saved_states[codingpass] = *mqenc_reg;
			}

#ifdef PRINT_CX_D
			printf("cup ");
#endif
			//printf("cup\n");
			tot_dist -= cup(rawcb->v, bmp_width, states, threadIdx.x, k_sig, cb->B_d, &buf_L, 
				rawcb->w, rawcb->h, p);
			//printf("ok\n");
			dist      [++codingpass] = tot_dist;
			cb->trunc_len   [codingpass] = buf_L;
			saved_states[codingpass] = *mqenc_reg;
		}

		//printf("terminate\n");
		/* terminate MQ codeword correctly */
		MQenc_Terminate(threadIdx.x, cb->B_d, &buf_L);
		//MQenc_Easy_Truncation(buf_B,buf_L);
		//printf("trunc\n");

		//MQenc_Easy_Truncation(cb->B_d, &buf_L);

		//no PCRD: only truncation length after last pass needed
		MQenc_CalcTruncation(saved_states, cb->trunc_len, cb->B_d, &buf_L, 
			/*start index*/ (enable_pcrd ? 1 : codingpass), 
			/*stop index*/  codingpass);
		cb->L = buf_L;

		if(enable_pcrd) {
			//printf("slopes\n");
			//store length after 1st coding pass; needed if pcrd wants to
			//discard all coding passes which isn't encodable at the moment.
			cb->trunc_len_1 = cb->trunc_len[1];
			//calculate feasible truncation points and slope values
			pcrd_calc_slopes(cb, dist, rawcb->subbType, 
				rawcb->dwt_level, rawcb->color_channel, 
				rawcb->subb_quantstep, slope_max, mode);
		}
		//subtract 1 , since we started with 0 instead of -1
		if(cb->L > 0)
			(cb->L)--;
	}
}


// Uses local memory for state array, dist and saved_states
// => for GPUs where local mem is faster than global mem
__global__ void encode_cb_kernel_local_mem
	(struct Raw_CB *rawcbs, 
     struct Codeblock *cbs, 
     int n_cbs,
	 float *slope_max,
	 int bmp_width,
	 int mode,
	 int enable_pcrd,
	 unsigned char *global_buf_d, unsigned char *global_buf_h, int buffer_size) 
{
	//int cb_i = threadIdx.x + blockIdx.x*blockDim.x;
	//int id = threadIdx.x;
	//int i;
	struct Codeblock *cb = &(cbs[cb_idx]);
	struct Raw_CB *rawcb = &(rawcbs[cb_idx]);

	float tot_dist = 0.0f;
	//int trunc_len[MAX_PASSES+1];
	float dist[MAX_PASSES+1];
	uint4 saved_states[MAX_PASSES+1];
	//int *trunc_len =      global_trunc_len_d +    cb_idx*PCRD_TEMP_ARRAY_SIZE;
	/*float *dist =         global_dist_d +         cb_idx*PCRD_TEMP_ARRAY_SIZE;
	uint4 *saved_states = global_saved_states_d + cb_idx*PCRD_TEMP_ARRAY_SIZE;*/
	//TODO: test if define macros are faster^^^^

	if(cb_idx < n_cbs) {
		int codingpass, p;
		//sign_magn_t *v = rawcb->v;
		//int w = rawcb->w, h = rawcb->h;
		//int v_scanw = rawcb->scanw;
		//unsigned *states = global_state_arr_d + cb_idx*MAX_STATE_ARRAY_SIZE;
		unsigned states[MAX_STATE_ARRAY_SIZE];

		char *k_sig;
		//unsigned char *buf_B = cb->B;
		int buf_L;

		dist[0] = tot_dist;
		cb->trunc_len[0] = 0;

		//printf("conv\n");
		//convert to sign-magnitude, plus quantization if irreversible
		if(mode == LOSSLESS)
			cb->nBitplanes = conv_sign_magn_lossless
				(rawcb->v, rawcb->w, rawcb->h, bmp_width);
		else
			cb->nBitplanes = conv_sign_magn_lossy
				(rawcb->v, rawcb->w, rawcb->h, bmp_width, rawcb->subb_quantstep);

		//cb->nBitplanes = quant_and_conv_sign_magn(rawcb->v, rawcb->w, rawcb->h, bmp_width, rawcb->subb_quantstep);

		if(rawcb->subbType == HH)
			k_sig = k_sig_hh;
		else if(rawcb->subbType == HL)
			k_sig = k_sig_hl;
		else
			k_sig = k_sig_lx;

#ifdef PRINT_COEFF
		{
			int x,y;
			printf("CB #%d\n", cb_idx);
			for(y=0; y < rawcb->h; y++) {
				for(x=0; x < rawcb->w; x++) {
					printf("%4d,", to_compl(rawcb->v[x + y*bmp_width]));
				}
				printf("\n");
			}
			printf("\n");
		}
#endif

		cb->Xdim = rawcb->w;
		cb->Ydim = rawcb->h;
		cb->nCodingPasses = cb->nBitplanes*3 - 2; //bitplanes is always >= 1
		cb->B_d = global_buf_d +  cb_idx * buffer_size;
		cb->B_h = global_buf_h +  cb_idx * buffer_size;
		//printf("%d bitplanes\n", cb->nBitplanes);

		//printf("states init\n");
		//for(p = rawcb->w+2; p < ((rawcb->h+1)/2+1)*(rawcb->w+2); p++)
		for(p = S_SCANW; p < ((rawcb->h+1)/2+1)*S_SCANW; p++)
			states[p] = 0u;

		//printf("mqenc init\n");
		/* init MQ coder */
		MQenc_Init(threadIdx.x);
		buf_L = 0;

		uint4 *mqenc_reg = (uint4*)(s_data + threadIdx.x*ALIGNED_MQENC_SIZE);

		//printf("coding\n");
		/* ************** do encoding ************** */
		codingpass = 0;
		//check_states(rawcb, bpc);
		for(p = cb->nBitplanes-1; p >= 0; p--) {
			//like this, the file compiles about 5 times faster (compared to
			//separate handling of the MSB). it even seems to run faster?!
			if(p != cb->nBitplanes-1) {
#ifdef PRINT_CX_D
				printf("bitplane %d: spp ", p);
#endif
				//printf("spp\n");
				tot_dist -= spp(rawcb->v, bmp_width, states, threadIdx.x, k_sig, cb->B_d, &buf_L, 
					rawcb->w, rawcb->h, p);
				dist      [++codingpass] = tot_dist;
				cb->trunc_len   [codingpass] = buf_L;
				saved_states[codingpass] = *mqenc_reg;

#ifdef PRINT_CX_D
				printf("mrp ");
#endif
				//printf("mrp\n");
				tot_dist -= mrp(rawcb->v, bmp_width, states, threadIdx.x, cb->B_d, &buf_L, 
					rawcb->w, rawcb->h, p);
				dist      [++codingpass] = tot_dist;
				cb->trunc_len   [codingpass] = buf_L;
				saved_states[codingpass] = *mqenc_reg;
			}

#ifdef PRINT_CX_D
			printf("cup ");
#endif
			//printf("cup\n");
			tot_dist -= cup(rawcb->v, bmp_width, states, threadIdx.x, k_sig, cb->B_d, &buf_L, 
				rawcb->w, rawcb->h, p);
			//printf("ok\n");
			dist      [++codingpass] = tot_dist;
			cb->trunc_len   [codingpass] = buf_L;
			saved_states[codingpass] = *mqenc_reg;
		}

		//printf("terminate\n");
		/* terminate MQ codeword correctly */
		MQenc_Terminate(threadIdx.x, cb->B_d, &buf_L);
		//MQenc_Easy_Truncation(buf_B,buf_L);
		//printf("trunc\n");

		//MQenc_Easy_Truncation(cb->B_d, &buf_L);

		//no PCRD: only truncation length after last pass needed
		MQenc_CalcTruncation(saved_states, cb->trunc_len, cb->B_d, &buf_L, 
			/*start index*/ (enable_pcrd ? 1 : codingpass), 
			/*stop index*/  codingpass);
		cb->L = buf_L;

		if(enable_pcrd) {
			//printf("slopes\n");
			//store length after 1st coding pass; needed if pcrd wants to
			//discard all coding passes which isn't encodable at the moment.
			cb->trunc_len_1 = cb->trunc_len[1];
			//calculate feasible truncation points and slope values
			pcrd_calc_slopes(cb, dist, rawcb->subbType, 
				rawcb->dwt_level, rawcb->color_channel, 
				rawcb->subb_quantstep, slope_max, mode);
		}
		//subtract 1 , since we started with 0 instead of -1
		if(cb->L > 0)
			(cb->L)--;
	}
}

#undef cb_idx 
