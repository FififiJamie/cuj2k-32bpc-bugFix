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
Lookup tables for bit plane coder; included by tier1.cu
Array elements are of type char to maximize cache hits.

Also contains (commented out) a main program to calculate
these lookup tables.
*/


//char is faster than int, probably because of more cache hits
__constant__ char k_sig_lx[256] = {
	0,1,1,2,1,2,2,2,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
	3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
	5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
	7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
	5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
	7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
	8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
	8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8
};
__constant__ char k_sig_hl[256] = {
	0,1,1,2,1,2,2,2,1,2,2,2,2,2,2,2,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
	5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
	3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
	7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
	3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
	7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
	4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
	7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8
};
__constant__ char k_sig_hh[256] = {
	0,3,3,6,3,6,6,8,3,6,6,8,6,8,8,8,1,4,4,7,4,7,7,8,4,7,7,8,7,8,8,8,
	1,4,4,7,4,7,7,8,4,7,7,8,7,8,8,8,2,5,5,7,5,7,7,8,5,7,7,8,7,8,8,8,
	1,4,4,7,4,7,7,8,4,7,7,8,7,8,8,8,2,5,5,7,5,7,7,8,5,7,7,8,7,8,8,8,
	2,5,5,7,5,7,7,8,5,7,7,8,7,8,8,8,2,5,5,7,5,7,7,8,5,7,7,8,7,8,8,8,
	1,4,4,7,4,7,7,8,4,7,7,8,7,8,8,8,2,5,5,7,5,7,7,8,5,7,7,8,7,8,8,8,
	2,5,5,7,5,7,7,8,5,7,7,8,7,8,8,8,2,5,5,7,5,7,7,8,5,7,7,8,7,8,8,8,
	2,5,5,7,5,7,7,8,5,7,7,8,7,8,8,8,2,5,5,7,5,7,7,8,5,7,7,8,7,8,8,8,
	2,5,5,7,5,7,7,8,5,7,7,8,7,8,8,8,2,5,5,7,5,7,7,8,5,7,7,8,7,8,8,8
};
__constant__ char k_mag[4] = {
	15,17,16,17
};
__constant__ char k_sign[256] = {
	10,11,11,11,13,14,14,14,13,14,14,14,13,14,14,14,11,11,10,10,12,12,13,13,12,12,13,13,12,12,13,13,
	11,10,11,10,12,13,12,13,12,13,12,13,12,13,12,13,11,11,11,11,12,12,12,12,12,12,12,12,12,12,12,12,
	13,12,12,12,13,12,12,12,10,11,11,11,10,11,11,11,14,14,13,13,14,14,13,13,11,11,10,10,11,11,10,10,
	14,13,14,13,14,13,14,13,11,10,11,10,11,10,11,10,14,14,14,14,14,14,14,14,11,11,11,11,11,11,11,11,
	13,12,12,12,10,11,11,11,13,12,12,12,10,11,11,11,14,14,13,13,11,11,10,10,14,14,13,13,11,11,10,10,
	14,13,14,13,11,10,11,10,14,13,14,13,11,10,11,10,14,14,14,14,11,11,11,11,14,14,14,14,11,11,11,11,
	13,12,12,12,13,12,12,12,13,12,12,12,13,12,12,12,14,14,13,13,14,14,13,13,14,14,13,13,14,14,13,13,
	14,13,14,13,14,13,14,13,14,13,14,13,14,13,14,13,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14
};
__constant__ char sign_flip[256] = {
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
	1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,
	1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,
	1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};


#if 0
// *********************************** LUT calculation
int k_sig_lx_h[256], k_sig_hl_h[256], k_sig_hh_h[256];
int k_mag_h[4];
int k_sign_h[256], sign_flip_h[256];


int make_number(int a7, int a6, int a5, int a4, int a3, int a2, int a1, int a0) {
	return (a7<<7)+(a6<<6)+(a5<<5)+(a4<<4)+(a3<<3)+(a2<<2)+(a1<<1)+a0;
}

int calc_k_sig_lx(int k_h, int k_v, int k_d) {
	if(k_h == 2)          return 8;
	else if(k_h == 1) {
		if(k_v >= 1)      return 7;
		else if(k_d >= 1) return 6;
		else              return 5;
	}
	else {
		if(k_v == 2)      return 4;
		else if(k_v == 1) return 3;
		else if(k_d >= 2) return 2;
		else if(k_d == 1) return 1;
		else              return 0;
	}
}

int calc_k_sig_hh(int k_h, int k_v, int k_d) {
	k_h += k_v;
	if(k_d >= 3)          return 8;
	else if(k_d == 2) {
		if(k_h >= 1)      return 7;
		else              return 6;
	}
	else if(k_d == 1) {
		if(k_h >= 2)      return 5;
		else if(k_h == 1) return 4;
		else              return 3;
	}
	else {
		if(k_h >= 2)      return 2;
		else if(k_h == 1) return 1;
		else              return 0;
	}
}

// called once before tier 1: initializes k_sig_xx, k_sign,....
void bpc_init_luts() {
	int l,r,u,d,ul,ur,dl,dr;
	int x_l,x_r,x_u,x_d;
	int sigma_l,sigma_r,sigma_u,sigma_d;
	//int k_sig_i;
	int has_signif_nb, refined;

	// k_sig LUTs
	for(l = 0; l <= 1; l++)
	for(r = 0; r <= 1; r++)
	for(u = 0; u <= 1; u++)
	for(d = 0; d <= 1; d++) 
	for(ul = 0; ul <= 1; ul++)
	for(ur = 0; ur <= 1; ur++)
	for(dl = 0; dl <= 1; dl++)
	for(dr = 0; dr <= 1; dr++) {
		int i = make_number(l,r,u,d,ul,ur,dl,dr);
		int k_h = l+r;
		int k_v = u+d;
		int k_d = ul+ur+dl+dr;
		k_sig_lx_h[i] = calc_k_sig_lx(k_h,k_v,k_d);
		k_sig_hl_h[i] = calc_k_sig_lx(k_v,k_h,k_d); //just switch h and v
		k_sig_hh_h[i] = calc_k_sig_hh(k_h,k_v,k_d);
	}

	for(has_signif_nb=0; has_signif_nb<=1; has_signif_nb++)
		for(refined=0; refined<=1; refined++) {
			if(refined)
				k_mag_h[(has_signif_nb<<1) + refined] = 17;
			else if(has_signif_nb)
				k_mag_h[(has_signif_nb<<1) + refined] = 16;
			else
				k_mag_h[(has_signif_nb<<1) + refined] = 15;
		}

	// sign LUTs
	for(x_l = 0; x_l <= 1; x_l++)
	for(x_r = 0; x_r <= 1; x_r++)
	for(x_u = 0; x_u <= 1; x_u++)
	for(x_d = 0; x_d <= 1; x_d++)
	for(sigma_l=0; sigma_l<=1; sigma_l++)
	for(sigma_r=0; sigma_r<=1; sigma_r++)
	for(sigma_u=0; sigma_u<=1; sigma_u++)
	for(sigma_d=0; sigma_d<=1; sigma_d++) {
		int i = make_number(x_l,x_r,x_u,x_d,sigma_l,sigma_r,sigma_u,sigma_d);
		int x_h=0,x_v=0,product;

		if(x_l==1) product = -1;
		else if(sigma_l==1) product=1;
		else product = 0;
		x_h += product;
		if(x_r==1) product = -1;
		else if(sigma_r==1) product=1;
		else product = 0;
		x_h += product;
		x_h = min(1, x_h);
		x_h = max(-1, x_h); //cut down to range [-1,+1]

		if(x_u==1) product = -1;
		else if(sigma_u==1) product=1;
		else product = 0;
		x_v += product;
		if(x_d==1) product = -1;
		else if(sigma_d==1) product=1;
		else product = 0;
		x_v += product;
		x_v = min(1, x_v);
		x_v = max(-1, x_v); //cut down to range [-1,+1]

		     if(x_h== 1 && x_v== 1) { k_sign_h[i]=14; sign_flip_h[i]=0; }
		else if(x_h== 1 && x_v== 0) { k_sign_h[i]=13; sign_flip_h[i]=0; }
		else if(x_h== 1 && x_v==-1) { k_sign_h[i]=12; sign_flip_h[i]=0; }
		else if(x_h== 0 && x_v== 1) { k_sign_h[i]=11; sign_flip_h[i]=0; }
		else if(x_h== 0 && x_v== 0) { k_sign_h[i]=10; sign_flip_h[i]=0; }
		else if(x_h== 0 && x_v==-1) { k_sign_h[i]=11; sign_flip_h[i]=1; }
		else if(x_h==-1 && x_v== 1) { k_sign_h[i]=12; sign_flip_h[i]=1; }
		else if(x_h==-1 && x_v== 0) { k_sign_h[i]=13; sign_flip_h[i]=1; }
		else if(x_h==-1 && x_v==-1) { k_sign_h[i]=14; sign_flip_h[i]=1; }
		else { assert(0); }
	}
	//copy LUTs to device (constant memory)
	/*cutilSafeCall(cudaMemcpyToSymbol(k_sig_lx, k_sig_lx_h, sizeof(k_sig_lx_h)));
	cutilSafeCall(cudaMemcpyToSymbol(k_sig_hl, k_sig_hl_h, sizeof(k_sig_hl_h)));
	cutilSafeCall(cudaMemcpyToSymbol(k_sig_hh, k_sig_hh_h, sizeof(k_sig_hh_h)));
	cutilSafeCall(cudaMemcpyToSymbol(k_mag, k_mag_h, sizeof(k_mag_h)));
	cutilSafeCall(cudaMemcpyToSymbol(k_sign, k_sign_h, sizeof(k_sign_h)));
	cutilSafeCall(cudaMemcpyToSymbol(sign_flip, sign_flip_h, sizeof(sign_flip_h)));*/
}

void print_array(FILE *fp, char *name, int *arr, int n) {
	int i;
	fprintf(fp, "__constant__ int %s[%d] = {", name, n);
	for(i=0; i<n; i++) {
		if((i%32)==0)
			fprintf(fp, "\n\t");
		fprintf(fp, "%d", arr[i]);
		if(i < (n-1))
			fprintf(fp, ",");
	}
	fprintf(fp, "\n};\n");
}

int main() {
	FILE *fp = fopen("bpc_luts.txt", "w");
	bpc_init_luts();
	print_array(fp, "k_sig_lx", k_sig_lx_h, 256);
	print_array(fp, "k_sig_hl", k_sig_hl_h, 256);
	print_array(fp, "k_sig_hh", k_sig_hh_h, 256);
	print_array(fp, "k_mag", k_mag_h, 4);	
	print_array(fp, "k_sign", k_sign_h, 256);
	print_array(fp, "sign_flip", sign_flip_h, 256);
	fclose(fp);

	/*int x, s=0;

	x = sizeof(struct MQ_Enc_Info); s += x;
	printf("mq: %d\n", x);
	x = sizeof(struct BPC_Info); s += x;
	printf("bpc: %d\n", x);
	x = sizeof(struct Codeblock); s += x;
	printf("cb: %d\n", x);
	x = sizeof(struct Raw_CB); s += x;
	printf("raw-cb: %d\n", x);

	printf("--------------\ntotal: %d\n", s);
	printf("%0.4f fit in 16k shared mem\n", 16384.0f / (float)s );*/

	return 0;
}
#endif //#if 0
