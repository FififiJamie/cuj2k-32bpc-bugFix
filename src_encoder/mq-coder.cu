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
MQ coder device functions; included by tier1.cu

__device__ void MQenc_Encode(int id, unsigned char *B, int *L, int cx, int d)
Encode one (context,decision) pair for MQ coder given by 'id'.

__device__ void MQenc_Terminate(int id, unsigned char *B, int *L)
Terminates MQ segment; truncation must be done afterwards.

__device__ void MQenc_CalcTruncation(....)
Calculate legal truncation points after each coding passes
in the range from 'pass_begin' to 'pass_end', so that all
symbols can be decoded correctly up to each coding pass
*/

#define FAST_MQ_ENCODE //choose which MQenc_Encode to use


typedef long long int64;


/* Lookup-tables for MQ-coder (stolen from jj2000) */
/* The data structures containing the probabilities for the LPS;
   LPS probability in context cx is = Qe[I[cx]] / (0.708 * 2^16) */
__constant__
    unsigned short Qe[]={0x5601, 0x3401, 0x1801, 0x0ac1, 0x0521, 0x0221, 0x5601,
              0x5401, 0x4801, 0x3801, 0x3001, 0x2401, 0x1c01, 0x1601,
              0x5601, 0x5401, 0x5101, 0x4801, 0x3801, 0x3401, 0x3001,
              0x2801, 0x2401, 0x2201, 0x1c01, 0x1801, 0x1601, 0x1401,
              0x1201, 0x1101, 0x0ac1, 0x09c1, 0x08a1, 0x0521, 0x0441,
              0x02a1, 0x0221, 0x0141, 0x0111, 0x0085, 0x0049, 0x0025,
              0x0015, 0x0009, 0x0005, 0x0001, 0x5601 };

//next index I[cx] if MPS is coded in context cx
//???don't know why, but this has to be 32bit. doesn't work with
//signed/unsigned char/int
__constant__
    unsigned NMPS[]={ 1 , 2, 3, 4, 5,38, 7, 8, 9,10,11,12,13,29,15,16,17,
                 18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
                 35,36,37,38,39,40,41,42,43,44,45,45,46 };

/* next index I[cx] if LPS is coded in context cx */
__constant__
    unsigned char NLPS[]={ 1 , 6, 9,12,29,33, 6,14,14,14,17,18,20,21,14,14,15,
                 16,17,18,19,19,20,21,22,23,24,25,26,27,28,29,30,31,
                 32,33,34,35,36,37,38,39,40,41,42,43,46 };

/* Whether LPS and MPS should be switched */
__constant__       /* at indices 0, 6, and 14 we switch */
    unsigned char SWITCH[]={ 1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

//use defines instead of variables: it's faster
#define enc_start ((uint4*)(s_data + id*ALIGNED_MQENC_SIZE))
#define enc_MPS   ((unsigned char*)(s_data + id*ALIGNED_MQENC_SIZE + sizeof(uint4)))
#define enc_I     ((unsigned char*)(enc_MPS + N_CONTEXTS))


#define enc_A (enc_start->x)
#define enc_C (enc_start->y)
#define enc_T (enc_start->z)
#define enc_t (enc_start->w)
/*#define enc_A (*((unsigned*)(s_data + id*ALIGNED_MQENC_SIZE)))
#define enc_C (*((unsigned*)(s_data + id*ALIGNED_MQENC_SIZE +   sizeof(unsigned))))
#define enc_T (*((unsigned*)(s_data + id*ALIGNED_MQENC_SIZE + 2*sizeof(unsigned))))
#define enc_t (*((unsigned*)(s_data + id*ALIGNED_MQENC_SIZE + 3*sizeof(unsigned))))*/

	/* C_carry   = bit 27
	   C_msbs    = bits 20..27
	   C_partial = bits 19..26 (contains partial codeword)
	   C_active  = bits 0..15 (contains lower interval bound)
	   (space bits = bits 16..18) */
/* read access to several bit areas in C */
#define C_msbs (((enc_C) >> 20) & 0xFF)
#define C_carry (((enc_C) >> 27) & 1)
#define C_partial (((enc_C) >> 19) & 0xFF)




/* reset MPS[cx], I[cx] arrays (for en- and decoder) */
__device__ void MQ_Reset(unsigned char *MPS, unsigned char *I) {
	/* init MPS[cx] with 0 for all cx */
	int i;
	for(i = 0; i < N_CONTEXTS; i++)
		MPS[i] = I[i] = 0;

	/* set some special values (different in wiley<->marcellin!)*/
	I[0] = 4; /*Ksig*/
	//I[17] = 3; /*Krun <- wiley*/
	I[9] = 3; /*Krun <- marcellin*/
	I[18] = 46; /*Kuni*/
}


/* reset A,C,t,T */
__device__ inline void MQenc_Restart(int id) {
	//uint4 *enc_start = (uint4*)(s_data + id * ALIGNED_MQENC_SIZE);
	(enc_A) = 0x8000;
	(enc_C) = 0;
	(enc_t) = 12; //12 before
	(enc_T) = 0;


    //char *enc_start_temp = s_data + id*ALIGNED_MQENC_SIZE;
    //uint4 * enc_start_temp_2 = (uint4*)enc_start_temp;
    //enc_start_temp_2->x = 0x8000;
    //enc_start_temp_2->y = 0;
    //enc_start_temp_2->w = 12;
    //enc_start_temp_2->z = 0;

}



/* initialize entire MQenc structure, including buffer */
__device__ void MQenc_Init(int id) {
	//uint4 *enc_start = (uint4*)(s_data  +  id*ALIGNED_MQENC_SIZE);
	//char *enc_MPS = s_data + id*ALIGNED_MQENC_SIZE + sizeof(uint4);
	//char *enc_I   = enc_MPS + N_CONTEXTS;

    char *MPS_temp = s_data + id*ALIGNED_MQENC_SIZE + sizeof(uint4);
    char *enc_I_temp = MPS_temp + N_CONTEXTS;

    MQ_Reset((unsigned char*)MPS_temp, (unsigned char*)enc_I_temp);

    //printf("~~~~~s_data = %d\n", s_data[1] - s_data[0]);

	/* skip 1st byte */
	//MQ_Reset(enc_MPS, enc_I);
	MQenc_Restart(id);
}



__device__ void putByte(int id, unsigned char *B, int *L) {
	//uint4 *enc_start = (uint4*)(s_data + id*ALIGNED_MQENC_SIZE);

	//don't respect max. buffer size
	//if((*L) < MAX_LEN_MQ)
		B[(*L)++] = enc_T;

	/*if((*L) >= MAX_LEN_MQ) {
		//buf_too_small_d = 1;
		return;
	}
	B[(*L)++] = enc_T;*/
/*	if((*L) >= 0)
		B[*L] = (enc_T);
	(*L)++;*/
}



/* transfer from C to T */
__device__ void transferByte(int id, unsigned char *B, int *L) {
	//uint4 *enc_start = (uint4*)(s_data + id*ALIGNED_MQENC_SIZE);

	if((enc_T) == 0xFF) { /* need bit stuff */
		putByte(id, B, L);
		(enc_T) = C_msbs;
		(enc_C) &= 0xFFFFF; /* clear C_msbs = keep bits 0..19 */
		(enc_t) = 7; /* transfer 7 bits + carry */
	}
	else {
		(enc_T) += C_carry; /* propagate carry from C to T */
		(enc_C) &= 0x7FFFFFF; /* clear C_carry = keep bits 0..26 */

		putByte(id, B, L);

		if((enc_T) == 0xFF) {
			(enc_T) = C_msbs;
			(enc_C) &= 0xFFFFF; /* clear C_msbs = keep bits 0..19 */
			(enc_t) = 7; /* transfer 7 bits + carry */
		}
		else {
			(enc_T) = C_partial;
			(enc_C) &= 0x807FFFF; /*clear C_partial = keep bits 0..18 + 27*/
			(enc_t) = 8; /* transfer full byte */
		}
	}
}



#ifdef FAST_MQ_ENCODE //defined on top of file
/*fast version: chap.17.1.1 (p.646) in Marcellin's book
  only slightly faster than normal version*/
__device__ void MQenc_Encode(int id, unsigned char *B, int *L, int cx, int d) {

	/*uint4 *enc_start = (uint4*)(s_data + id*ALIGNED_MQENC_SIZE);
	char *enc_MPS = s_data + id*ALIGNED_MQENC_SIZE + sizeof(uint4);
	char *enc_I   = enc_MPS + N_CONTEXTS;*/

	//static int counter=1;
	unsigned p; //LPS prob. in context cx
	my_dev_assert((d == 0) || (d == 1));
	my_dev_assert((cx >= 0) && (cx < N_CONTEXTS));

#ifdef PRINT_CX_D
	//dbgTier1("%2d. CX:%2d D:%d\n", counter++, cx, d);
	printf("(%d,%d)", cx, d);
#endif

	p = Qe[enc_I[cx]];

	(enc_A) -= p;
	if(d == (int)(enc_MPS[cx])) { //coding an MPS
		if((enc_A) >= (1<<15)) //no renormalization and hence no cond. exchange
			(enc_C) += p;
		else {
			if((enc_A) < p) //cond. exchange
				(enc_A) = p;
			else
				(enc_C) += p;
			enc_I[cx] = NMPS[enc_I[cx]];
			do { //perform renormalization shift
				(enc_A) <<= 1;
				(enc_C) <<= 1;
				if((--(enc_t)) == 0) // --!
					transferByte(id, B, L);
			} while((enc_A) < (1<<15));
		}
	}
	else { //coding an LPS; renormalization is inevitable
		if((enc_A) < p) //conditional exchange
			(enc_C) += p;
		else
			(enc_A) = p;
		enc_MPS[cx] ^=/*XOR*/ (SWITCH[enc_I[cx]]);
		enc_I[cx] = (NLPS[enc_I[cx]]);
		do { //perform renormalization shift (yes, the same as above)
			(enc_A) <<= 1;
			(enc_C) <<= 1;
			if((--(enc_t)) == 0) // --!
				transferByte(id, B, L);
		} while((enc_A) < (1<<15));
	}
}
#else

//doesn't work yet, TODO: adjust to use shared mem, and determine which one is faster
__device__ void MQenc_Encode(struct MQ_Enc_Info *enc, struct Codeblock *cb, int cx, int d) {
	unsigned s = enc->MPS[cx];
	unsigned p = Qe [enc->I [cx]];

	/*assert((d == 0) || (d == 1));
	assert((cx >= 0) && (cx < N_CONTEXTS));*/

	enc->A -= p;
	if(enc->A < p)
		s = 1 - s; /* conditional exchange MPS/LPS; only local effect (intended?)*/

	if(d == s)
		enc->C += p; /* assign MPS the upper sub-interval*/
	else
		enc->A = p; /* assign LPS the lower sub-interval */

	if(enc->A < (1<<15) ) {
		if(d == enc->MPS[cx]) /*symbol was real MPS (use old value)*/
			enc->I[cx]  =  NMPS[ enc->I[cx] ];
		else { /*symbol was real LPS*/
			/*switch MPS/LPS if SWITCH=1*/
			enc->MPS[cx] ^= /*XOR*/ SWITCH[ enc->I[cx] ];
			//assert((enc->MPS[cx]==0) || (enc->MPS[cx]==1));
			enc->I[cx] = NLPS[ enc->I[cx] ];
		}
	}

	while(enc->A < (1<<15) ) { /*perform renormalization shift*/
		(enc->A) <<= 1; /*double values*/
		(enc->C) <<= 1;
		(enc->t)--;
		if(enc->t == 0)
			transferByte(enc, cb);
	}
}
#endif


/* terminate segment, currently only "easy termination" */
__device__ void MQenc_Terminate(int id, unsigned char *B, int *L) {

	//uint4 *enc_start = (uint4*)(s_data + id*ALIGNED_MQENC_SIZE);

	int nBits = 27 - 15 - (enc_t); /* the number of bits we need to flush out of C*/
	/* move the next 8 available bits into the partial byte */

	//TODO: direct shift!
	(enc_C) <<= (enc_t); /* C = 2^t * C */
	//(enc_C) *= 1 << (enc_t); /* C = 2^t * C */

	while(nBits > 0) {
		transferByte(id,B,L);
		nBits -= (enc_t); /*new value of t is the number of bits just transferred*/
		/* move bits into available positions for next transfer*/

		//TODO: direct shift!
		(enc_C) <<= (enc_t); /* C = 2^t * C */
		//(enc_C) *= 1 << (enc_t); /* C = 2^t * C */
	}

	transferByte(id,B,L);
}


__device__ void MQenc_Easy_Truncation(unsigned char *B, int *L) {
	if((*L) < 0)
		(*L) = 0;

	/* 2nd phase: make truncation point "shorter" if possible */
	/* this happens very rarely, but when deliberately introducing 0xFFs,
	   the truncation works. */
	/* trunc FF at the end once if possible */
	if((*L) >= 1  &&  B[(*L) - 1] == 0xFF)
		(*L)--;
	/* trunc FF 7F at the end as often as possible */
	while((*L) >= 2  &&  B[(*L) - 2] == 0xFF &&
	                     B[(*L) - 1] == 0x7F)
	{
		(*L) -= 2;
	}
}



#define sav_A (saved[pass].x)
#define sav_C (saved[pass].y)
#define sav_T (saved[pass].z)
#define sav_t (saved[pass].w)

/* algorithm: p.500 Marcellin */
/* note: <<, >> have lower precedences than +,-,*,/ */
__device__ void MQenc_CalcTruncation(uint4 *saved,
									 int *trunc_len_array,
			          			     unsigned char *buf,
									 int *L,
						             int pass_begin, int pass_end)
{
	// stores copies out of enc->saved, should be faster
	int pre_trunc_len, pass, trunc_len, shortened;
	int64 Cr, Ar, RF, s, SF, SF_pow2, F; /* TODO: probably some could be 32bit */
	unsigned char c; /* current byte in buffer */

	/* calculate optimal truncation length for each coding pass */
	for(pass=pass_begin; pass <= pass_end; pass++) {
		//avoid too small index
		/*if(trunc_len_array[pass] < 0)
			trunc_len_array[pass] = 0;*/

		pre_trunc_len = trunc_len_array[pass];

		Cr = ((int64)sav_T << (int64)27) + ((int64)sav_C << (int64)sav_t);
		Ar = (int64)sav_A << (int64)sav_t;
		RF = 0;
		s = 8;
		SF = 35;
		F = 0;
		SF_pow2 = ((int64)1)<<SF; /* value is used often => pre-calculate it */

		/* 1st phase: make truncation point "longer" if necessary */
		      /*length-check necessary for some special cases */
		while(pre_trunc_len + F < (*L)  && //don't exceed data length
			  F < 5  &&  ((RF + SF_pow2 - 1 < Cr) || (RF + SF_pow2 - 1 >= Cr + Ar)))
		{
			F++;
			if(F <= 4) { /* maximum for F is 5, anyway */
				SF -= s;
				SF_pow2 = ((int64)1)<<SF;
				/*RF += SF_pow2 * ((int64)enc->buf.b[sav->buf_len + F - 1]); (slow?)*/
				/*if(buf_len + F - 1 < 0) {
					printf("pass=%d n_passes=%d\n", pass, nCodingPasses);
				}
				assert(buf_len + F - 1 >= 0);
				assert(buf_len + F - 1 < L);*/
				c = buf[pre_trunc_len + F - 1];
				RF += (int64)c << SF;
				if(c == 0xFFu)
					s = 7;
				else
					s = 8;
			}
		}
		/* now F==Fmin (as mentioned in the book) */

		trunc_len = pre_trunc_len + (int)F;
		shortened = 0;

		dbgTier1b("pass %d: %d+%d = %d bytes. ", pass, pre_trunc_len, (int)F, trunc_len);

		/* 2nd phase: make truncation point "shorter" if possible */
		/* this happens very rarely, but when deliberately introducing 0xFFs,
		   the truncation works. */
		/* trunc FF at the end once if possible */
		if(trunc_len >= 1 && buf[trunc_len-1]==0xFF) {
			trunc_len--;
			shortened = 1;
		}
		/* trunc FF 7F at the end as often as possible */
		while(trunc_len >= 2 && buf[trunc_len-2]==0xFF &&
		                        buf[trunc_len-1]==0x7F)
		{
			trunc_len -= 2;
			shortened = 1;
		}
		//subtract 1 because lengths are always 1 byte too big,
		//because L starts with 0 instead of -1
		trunc_len_array[pass] = max(trunc_len-1, 0);
		//assert(trunc_len_array[pass] <= L);

		if(shortened)
			dbgTier1b("further trunc to %d bytes.\n", trunc_len);
		else
			dbgTier1b("\n");
	}

	dbgTier1b("Buffer size was %d bytes\n", cb->L);
	/* if there is truncation after the last coding pass,
	   make the buffer size smaller to save memory later */
	if(pass_end > 0)
		(*L) = trunc_len; //use last value of trunc_len
	/*if((*L) < 0)
		(*L) = 0;*/
	dbgTier1b("Buffer size is  %d bytes\n", cb->L);
}
