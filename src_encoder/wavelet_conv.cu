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
unsigned short int dwt_encode_stepsize(float stepsize, int numbps)
converts a 32-bit float to a 16-bit floatingpoint type ready for QCD-marker. This is used
to encode the quantization-stepsize.


int DWT(struct Tile *eingang,  int  max_Level,int line_length, int mode, int quant_enable,
		cudaStream_t stream, int* temp_d)
dwt-main-function. Performs the dwt on a whole tile for all its channels.
It calls a recursive function which performs the dwt for each level.
The dwt is implemented as a lifting implementation. The kernels are in the
5_3_wavelet_kernels.cu (for reversible 5-3-Le Gall-Wavelet) and 9_7_wavelet_kernels.cu
(for irreversible 9-7-Daubechies-Wavelet) files.


*/




#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"
#include "waveletTransform.h"
#include "rate-control.h"
#include <math.h>
//#include <cutil.h>
#include <helper_cuda.h>

//#define debug
#define temp_xdim 1024

#include "5_3_wavelet_kernels.cu"
#include "9_7_wavelet_kernels.cu"


//converts a 32-bit float (quantization-step) to a 16-bit floatingpoint type ready for QCD-marker
unsigned short int dwt_encode_stepsize(float stepsize, int numbps) {
	int p; //floorlog(stepsize)
	unsigned short int erg;
	int * pointer;
	p=0;

	//calculate log of stepsize
	while (stepsize<1){
		p--;
		stepsize*=2;
	}

	while (stepsize>=2){
		p++;
		stepsize/=2;
	}
	pointer= (int*) &stepsize; //read stepsize as integer, to extract the first 11 bits of the mantissa

	erg = ((*pointer)>>12)&0x7ff;
	erg = erg|((numbps - p)<<11); //11 before

	return erg;
}







//function to call the horizontal 5-3-wavelet kernel
void wavelet_step_hor_rev(int* bild_d,int*temp_d,int xdim, int ydim,int line_length , cudaStream_t stream){

	if (xdim%8==0 && xdim>63){

		dim3 dimBlock(xdim/4);
		dim3 dimGrid(1,ydim);
		size_t shared_mem_size=(xdim+1)*sizeof(float);

		// Launch the device computation threads
		dwt_5_3_Kernel_hor_0mod4_shared<<<dimGrid, dimBlock,shared_mem_size, stream>>>(temp_d,bild_d,line_length );

	}else if (xdim%2==0){

		dim3 dimBlock(xdim/2);
		dim3 dimGrid(1, ydim);
		size_t shared_mem_size=xdim*sizeof(int);

		// Launch the device computation threads
		dwt_5_3_Kernel_hor_even_shared<<<dimGrid, dimBlock,shared_mem_size, stream>>>(temp_d,bild_d,xdim,line_length );

	}else if (xdim==1){

		dim3 dimBlock(1);
		dim3 dimGrid(1,ydim);

		// Launch the device computation threads
		dwt_one_pixel_hor_rev<<<dimGrid, dimBlock>>>(temp_d,bild_d,line_length);

	}else{

		dim3 dimBlock((xdim+1)/2);
		dim3 dimGrid(1, ydim);
		size_t shared_mem_size=xdim*sizeof(int);

		// Launch the device computation threads
		dwt_5_3_Kernel_hor_odd_shared<<<dimGrid, dimBlock,shared_mem_size, stream>>>(temp_d,bild_d,xdim,line_length );

	}
}



//function to choose the vertical 5-3-wavelet kernel
void wavelet_step_ver_rev(int* bild_d,int*temp_d,int xdim, int ydim,int line_length, cudaStream_t stream ){

	if (ydim%4==0&&xdim%16==0&&ydim>75&&((ydim+1)%56>4)){

		dim3 dimBlock(16 ,16);
		dim3 dimGrid(xdim/16,(ydim+55)/56);
		size_t shared_mem_size=16*16*2*sizeof(int);

		// Launch the device computation threads
		dwt_5_3_Kernel_0mod4_x16_even_shared<<<dimGrid, dimBlock,shared_mem_size,  stream>>>(bild_d,temp_d,  ydim, line_length);

	} else if (xdim%16==0&&ydim>60&&((ydim+1)%56>4)){

		dim3 dimBlock(16 ,16);
		dim3 dimGrid(xdim/16,(ydim+55)/56);
		size_t shared_mem_size=16*16*2*sizeof(int);

		// Launch the device computation threads
		dwt_5_3_Kernel_56pix_x16_even_shared<<<dimGrid, dimBlock,shared_mem_size,  stream>>>(bild_d,temp_d, ydim, line_length);

	} else if (ydim>60&&((ydim+1)%56>4)){

		dim3 dimBlock(16 ,16);
		dim3 dimGrid((xdim+15)/16,(ydim+55)/56);
		size_t shared_mem_size=16*16*2*sizeof(int);

		// Launch the device computation threads
		dwt_5_3_Kernel_56pix_notx16_even_shared<<<dimGrid, dimBlock,shared_mem_size,  stream>>>(bild_d,temp_d,xdim,  ydim, line_length);

	}else if (xdim%32==0&&ydim>30&&((ydim+1)%24>4)){

		dim3 dimBlock(32 ,8);
		dim3 dimGrid(xdim/32,(ydim+23)/24);
		size_t shared_mem_size=32*8*2*sizeof(int);

		// Launch the device computation threads
		dwt_5_3_Kernel_56pix_x16_even_shared<<<dimGrid, dimBlock,shared_mem_size,  stream>>>(bild_d,temp_d,  ydim, line_length);

	}else if (ydim>30&&((ydim+1)%24>4)){

		dim3 dimBlock(32 ,8);
		dim3 dimGrid((xdim+31)/32,(ydim+23)/24);
		size_t shared_mem_size=32*8*2*sizeof(int);

		// Launch the device computation threads
		dwt_5_3_Kernel_56pix_notx16_even_shared<<<dimGrid, dimBlock,shared_mem_size,  stream>>>(bild_d,temp_d,xdim,  ydim, line_length);

	}else if (ydim%2==0){

		dim3 dimBlock(1,ydim/2);
		dim3 dimGrid(xdim);
		size_t shared_mem_size=ydim*sizeof(int);

		// Launch the device computation threads
		dwt_5_3_Kernel_ver_even_shared<<<dimGrid, dimBlock,shared_mem_size, stream>>>(bild_d,temp_d,line_length );

	}else if (ydim==1){

		dim3 dimBlock(1);
		dim3 dimGrid(xdim);

		// Launch the device computation threads
		dwt_one_pixel_ver_rev<<<dimGrid, dimBlock>>>(bild_d,temp_d);

	}else{

		dim3 dimBlock(1,(ydim+1)/2);
		dim3 dimGrid(xdim);
		size_t shared_mem_size=ydim*sizeof(int);

		// Launch the device computation threads
		dwt_5_3_Kernel_ver_odd_shared<<<dimGrid, dimBlock,shared_mem_size, stream>>>(bild_d,temp_d,line_length );

	}
}



//function to choose the horizontal 9-7-wavelet kernel
void wavelet_step_hor(float* bild_d,float*temp_d,int xdim, int ydim,int line_length, cudaStream_t stream ){

	if (xdim%8==0 && xdim>63){

		dim3 dimBlock(xdim/4);
		dim3 dimGrid(1,ydim);
		size_t shared_mem_size=(xdim+1)*sizeof(float);

		// Launch the device computation threads
		dwt_9_7_Kernel_hor_0mod4_shared<<<dimGrid, dimBlock,shared_mem_size, stream>>>(temp_d,bild_d,line_length );

	}else if (xdim%2==0){

		dim3 dimBlock(xdim/2);
		dim3 dimGrid(1, ydim);
		size_t shared_mem_size=xdim*sizeof(float);

		// Launch the device computation threads
		dwt_9_7_Kernel_hor_even_shared<<<dimGrid, dimBlock,shared_mem_size, stream>>>(temp_d,bild_d,xdim,line_length );

	}else if (xdim==1){

		dim3 dimBlock(1);
		dim3 dimGrid(1,ydim);

		dwt_one_pixel_hor<<<dimGrid, dimBlock>>>(temp_d,bild_d,line_length );

	}else{

		dim3 dimBlock((xdim+1)/2);
		dim3 dimGrid(1, ydim);
		size_t shared_mem_size=xdim*sizeof(float);

		// Launch the device computation threads
		dwt_9_7_Kernel_hor_odd_shared<<<dimGrid, dimBlock,shared_mem_size, stream>>>(temp_d,bild_d,xdim,line_length );

	}

}



//function to choose the vertical 9-7-wavelet kernel
void wavelet_step_ver(float* bild_d,float*temp_d,int xdim, int ydim,int line_length, cudaStream_t stream ){

	if (ydim%4==0&&xdim%16==0&&ydim>75&&((ydim+1)%56>4)){

		dim3 dimBlock(16 ,16);
		dim3 dimGrid(xdim/16,(ydim+55)/56);
		size_t shared_mem_size=16*16*2*sizeof(int);

		// Launch the device computation threads
		dwt_9_7_Kernel_0mod4_x16_even_shared<<<dimGrid, dimBlock,shared_mem_size,  stream>>>(bild_d,temp_d, ydim, line_length);

	} else if (xdim%16==0&&ydim>75&&((ydim+1)%56>4)){

		dim3 dimBlock(16 ,16);
		dim3 dimGrid(xdim/16,(ydim+55)/56);
		size_t shared_mem_size=16*16*2*sizeof(int);

		// Launch the device computation threads
		dwt_9_7_Kernel_56pix_x16_even_shared<<<dimGrid, dimBlock,shared_mem_size,  stream>>>(bild_d,temp_d,  ydim, line_length);

	} else if (ydim>75&&((ydim+1)%56>4)){

		dim3 dimBlock(16 ,16);
		dim3 dimGrid((xdim+15)/16,(ydim+55)/56);
		size_t shared_mem_size=16*16*2*sizeof(int);

		// Launch the device computation threads
		dwt_9_7_Kernel_56pix_notx16_even_shared<<<dimGrid, dimBlock,shared_mem_size,  stream>>>(bild_d,temp_d,xdim,  ydim, line_length);

	}else if (ydim%4==0&&xdim%32==0&&ydim>18&&((ydim+1)%24>4)){

		dim3 dimBlock(32 ,8);
		dim3 dimGrid(xdim/32,(ydim+23)/24);
		size_t shared_mem_size=32*8*2*sizeof(int);

		// Launch the device computation threads
		dwt_9_7_Kernel_0mod4_x16_even_shared<<<dimGrid, dimBlock,shared_mem_size,  stream>>>(bild_d,temp_d,  ydim, line_length);

	}else  if (xdim%32==0&&ydim>18&&((ydim+1)%24>4)){

		dim3 dimBlock(32 ,8);
		dim3 dimGrid(xdim/32,(ydim+23)/24);
		size_t shared_mem_size=32*8*2*sizeof(int);

		// Launch the device computation threads
		dwt_9_7_Kernel_56pix_x16_even_shared<<<dimGrid, dimBlock,shared_mem_size, stream>>>(bild_d,temp_d, ydim, line_length);

	}else if (ydim>18&&((ydim+1)%24>4)){

		dim3 dimBlock(32 ,8);
		dim3 dimGrid((xdim+31)/32,(ydim+23)/24);
		size_t shared_mem_size=32*8*2*sizeof(int);

		// Launch the device computation threads
		dwt_9_7_Kernel_56pix_notx16_even_shared<<<dimGrid, dimBlock,shared_mem_size, stream>>>(bild_d,temp_d,xdim,  ydim, line_length);

	}else if (ydim%2==0){

		dim3 dimBlock(1,ydim/2);
		dim3 dimGrid(xdim);
		size_t shared_mem_size=ydim*sizeof(float);

		// Launch the device computation threads
		dwt_9_7_Kernel_ver_even_shared<<<dimGrid, dimBlock, shared_mem_size, stream>>>(bild_d,temp_d,ydim,line_length );

	}else if (ydim==1){

		dim3 dimBlock(1);
		dim3 dimGrid(xdim);

		// Launch the device computation threads
		dwt_one_pixel_ver<<<dimGrid, dimBlock>>>(bild_d,temp_d);

	}else{

		dim3 dimBlock(1,(ydim+1)/2);
		dim3 dimGrid(xdim);
		size_t shared_mem_size=ydim*sizeof(float);

		// Launch the device computation threads
		dwt_9_7_Kernel_ver_odd_shared<<<dimGrid, dimBlock, shared_mem_size, stream>>>(bild_d,temp_d,ydim,line_length );

	}
}



void _2D_dwt(int* bild_d,int * temp_d,int** erg,int xdim, int ydim,int line_length,
			 int mode, cudaStream_t stream){//erg 4dim array of int* to store the result subbands

#ifdef debug
	 int m=ydim;
	 int n = xdim;
	 int i,j;
	 int nElem=256;

	float* bild= (float*) malloc(sizeof(float)*16*1024);
	printf("start, xdim: %d, ydim:%d\n ",xdim,ydim);
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilSafeCall(cudaMemcpy(bild, bild_d, nElem*sizeof(float), cudaMemcpyDeviceToHost, stream));
	cutilSafeCall(cudaStreamSynchronize(stream));

	printf("vor der dwt\n");
	for(i=0;i<16;i++){
		for(j=0;j<16;j++){
			printf("%d ",(int)(bild[i*16+j]));
		}
		printf("\n");
	}

	getchar();

	printf("\n");



#endif



//vertical wavelet step
if(mode==LOSSLESS)
	wavelet_step_ver_rev(bild_d,temp_d,xdim ,ydim,line_length, stream );
else
	wavelet_step_ver((float*)bild_d,(float*)temp_d,xdim ,ydim,line_length, stream );





#ifdef debug
	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilSafeCall(cudaMemcpy(bild, temp_d, 16*1024*sizeof(float), cudaMemcpyDeviceToHost));

	printf("mitten in dwt\n");
	for(i=0;i<16;i++){
		for(j=0;j<16;j++){
			printf("%d ",(int)(bild[i*1024+j]));
		}
		printf("\n");
	}

	getchar();

	printf("\n");
#endif


//horizontal wavelet step
if(mode==LOSSLESS)
	wavelet_step_hor_rev(bild_d,temp_d,xdim ,ydim,line_length ,  stream);
else
	wavelet_step_hor((float*)bild_d,(float*)temp_d,xdim ,ydim,line_length ,  stream);



#ifdef debug

	cutilSafeCall(cudaStreamSynchronize(stream));
	cutilSafeCall(cudaMemcpy(bild, bild_d, nElem*sizeof(float), cudaMemcpyDeviceToHost));
	printf("nach dwt\n");
	for(i=0;i<16;i++){
		for(j=0;j<16;j++){
			printf("%d ",(int)(bild[i*16+j]));
		}
		printf("\n");
	}



	getchar();
#endif

//calculate starting adresses for the subbands
	erg[0]= &(bild_d[0]);
	erg[1]= &(bild_d[((xdim+1)/2)]);
	erg[2]= &(bild_d[line_length*((ydim+1)/2)]);
	erg[3]= &(bild_d[line_length*((ydim+1)/2)+((xdim+1)/2)]);


return;



}



//recursive function to perform dwt-steps
subband* DWT_rec(int * bild_d, int * temp_d,int  m, int n,  int  level,
				 int max_Level, int line_length, int mode, int quant_enable,
				 cudaStream_t stream, int bps){ //typically  m=height, n=width, level = number of recursions to do

	subband* ergebnis;
	float quantstepHH;
	float quantstepHL;
	float quantstepLH;
	float quantstepLL;

	typedef int* pointerTyp;
	pointerTyp daten[4];


	ergebnis= (subband*) malloc(sizeof(subband));
	ergebnis->Xdim=n;
	ergebnis->Ydim=m;

	//printf("DEBUG: DWT bps = %d\n", bps);
	if (level <= max_Level){

		//perform 2D dwt
		_2D_dwt(bild_d, temp_d,daten,n, m,line_length, mode,  stream);


		//get quantization-stepsize
		if (mode==LOSSY){
			quantstepHH=get_quantstep(level,max_Level,HH,quant_enable);
			quantstepHL=get_quantstep(level,max_Level,HL,quant_enable);
			quantstepLH=get_quantstep(level,max_Level,LH,quant_enable);
		}
		else{
			quantstepHH=1;
			quantstepHL=1;
			quantstepLH=1;
		}

    	ergebnis->Typ=LLsubdiv;

    	//allocating the structs for the result-subbands
    	ergebnis->subbands[1]=(subband*) malloc(sizeof(subband));
    	ergebnis->subbands[2]=(subband*) malloc(sizeof(subband));
    	ergebnis->subbands[3]=(subband*) malloc(sizeof(subband));


		ergebnis->subbands[1]->daten_d=daten[1];
        ergebnis->subbands[1]->Typ=HL;
        ergebnis->subbands[1]->Xdim=n/2;
        ergebnis->subbands[1]->Ydim=(m+1)/2;
		ergebnis->subbands[1]->fl_quantstep = quantstepHL;
        ergebnis->subbands[1]->quantstep=dwt_encode_stepsize(quantstepHL,bps);
        ergebnis->subbands[1]->K_max = calc_K_max(ergebnis->subbands[1]->quantstep, HL, mode, bps);


        ergebnis->subbands[2]->daten_d=daten[2];
        ergebnis->subbands[2]->Typ=LH;
        ergebnis->subbands[2]->Xdim=(n+1)/2;
        ergebnis->subbands[2]->Ydim=m/2;
		ergebnis->subbands[2]->fl_quantstep = quantstepLH;
        ergebnis->subbands[2]->quantstep=dwt_encode_stepsize(quantstepLH,bps);
        ergebnis->subbands[2]->K_max = calc_K_max(ergebnis->subbands[2]->quantstep, LH, mode, bps);


        ergebnis->subbands[3]->daten_d=daten[3];
        ergebnis->subbands[3]->Typ=HH;
        ergebnis->subbands[3]->Xdim=n/2;
        ergebnis->subbands[3]->Ydim=m/2;
		ergebnis->subbands[3]->fl_quantstep = quantstepHH;
        ergebnis->subbands[3]->quantstep=dwt_encode_stepsize(quantstepHH,bps);
        ergebnis->subbands[3]->K_max = calc_K_max(ergebnis->subbands[3]->quantstep, HH, mode, bps);



		//recursive call for the LL-subband
        ergebnis->subbands[0]=DWT_rec(bild_d, temp_d,(m+1)/2,(n+1)/2,level+1,max_Level,line_length, mode, quant_enable, stream, bps);

    }else{//last step of recursion
          quantstepLL=get_quantstep(level-1,max_Level,LLend,quant_enable);

          ergebnis->daten_d=bild_d;
          ergebnis->Typ=LLend;
          ergebnis->subbands[3]=NULL;
          ergebnis->subbands[2]=NULL;
          ergebnis->subbands[1]=NULL;
          ergebnis->subbands[0]=NULL;
		  ergebnis->fl_quantstep = quantstepLL;
          ergebnis->quantstep=dwt_encode_stepsize(quantstepLL,bps);
          ergebnis->K_max = calc_K_max(ergebnis->quantstep, LLend, mode, bps);

   	}
    return ergebnis;
}



//main DWT-function
int DWT(struct Tile *eingang,  int  max_Level,int line_length, int mode, int quant_enable,
		cudaStream_t stream, int bps, int* temp_d){
	int m,n,ch;
	subband ** subbands;
	int * bild_d;

	subbands= (subband**) malloc(eingang->channels * sizeof(subband*));
	m=eingang->yDim;
	n=eingang->xDim;

	if(mode==LOSSLESS)
		eingang->QS=0x40;
	else
		eingang->QS=0x42;

	//perform dwt for each channel of a tile
	for(ch=0;ch<eingang->channels;ch++)
	{
		bild_d =((int**)eingang->imgData_d)[ch];
		subbands[ch]=(subband*) DWT_rec(bild_d,temp_d,m,n,1,max_Level, line_length,mode, quant_enable, stream, bps);
	}
	eingang->imgData= (void **) subbands;
	return 0;
}
