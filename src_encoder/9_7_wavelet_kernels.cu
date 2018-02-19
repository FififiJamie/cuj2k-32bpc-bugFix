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

/* this file contains all the kernels to calculate the 9-7-wavelet transform.
They are separated into horizontal and vertical kernels (because they have a different access to the data).
For each direction there are various kernels which are called depending on the dimensions of the image.
Each kernel in this file refers to one in 5_3_wavelet_dernels.cu but they have different coefficients.
*/

#define a (-1.586134342f)
#define b (-0.05298011854f)
#define c 0.8829110762f
#define d 0.4435686522f
#define K  1.230174105f

///////////////////////////////////////////////////////////////////////////////
//horizontal 9-7-wavelet kernels



extern __shared__ float shared_bild[];
__global__ void dwt_9_7_Kernel_hor_odd_shared(float *bild,float*target, int xdim,int line_length){
	int i =threadIdx.x;
	int j=blockIdx.y;
	int temp_ind;
	int upper_ind=(i<blockDim.x-1?i:i-1);
	int lower_ind=(i>0?(i-1):0);
	float * shared_upper=&shared_bild[(xdim+1)/2];

	temp_ind=j*temp_xdim+(i<<1);
	shared_bild[i]= bild[temp_ind];
	if (i<blockDim.x-1)shared_upper[i]= bild[1+temp_ind];

	__syncthreads();

	if (i<blockDim.x-1) shared_upper[i]+=a*(shared_bild[i]+ shared_bild[i+1]);
	__syncthreads();
	shared_bild[i]+=b*(shared_upper[lower_ind]+ shared_upper[upper_ind]);
	__syncthreads();
	if (i<blockDim.x-1) shared_upper[i]+=c*(shared_bild[(i)]+ shared_bild[i+1]);
	__syncthreads();
	
	temp_ind=i+j*line_length;
	target[temp_ind]=(shared_bild[i]+d*(shared_upper[lower_ind]+shared_upper[upper_ind]))/K;
	if (i<blockDim.x-1)target[(xdim+1)/2+temp_ind]=shared_upper[i]*K/2;	
}


extern __shared__ float shared_bild[];
__global__ void dwt_9_7_Kernel_hor_even_shared(float *bild,float*target, int xdim,int line_length){
	int i =threadIdx.x;
	int j=blockIdx.y;
	int temp_ind;
	int upper_ind=(i<blockDim.x-1?i+1:i);
	int lower_ind=(i>0?(i-1):0);
	float * shared_upper=&shared_bild[xdim/2];

	temp_ind=j*temp_xdim+(i<<1);
	shared_bild[i]= bild[temp_ind];
	shared_upper[i]= bild[1+temp_ind];
	__syncthreads();

	shared_upper[i]+=a*(  shared_bild[i]+  shared_bild[upper_ind]);
	__syncthreads();
	shared_bild[i]+=b*(  shared_upper[lower_ind]+  shared_upper[i]);
	__syncthreads();
	shared_upper[i]+=c*(  shared_bild[(i)]+  shared_bild[upper_ind]);
	__syncthreads();

	temp_ind=i+j*line_length;
	target[temp_ind]=(shared_bild[i]+d*(  shared_upper[lower_ind]+ shared_upper[i]))/K;
	target[xdim/2+temp_ind]=shared_upper[i]*K/2;	
}



extern __shared__ float shared_bild[];
__global__ void dwt_9_7_Kernel_hor_0mod4_shared(float*temp,float *bild, int line_length){
//	int i =threadIdx.x<<1;
//	int j=blockIdx.y;
	int temp_ind;
	int shared_temp_ind;
	float *shared_lower=&shared_bild[threadIdx.x<<1];
	float *shared_upper=&shared_lower[blockDim.x*2+1];


	//read from global memory
	temp_ind=blockIdx.y*temp_xdim+threadIdx.x;
	shared_temp_ind=((threadIdx.x>>1)+(blockDim.x*2+1)*(threadIdx.x&1));
	shared_bild[shared_temp_ind]= temp[temp_ind];
	temp_ind+=blockDim.x;
	shared_temp_ind+=blockDim.x>>1;

	shared_bild[shared_temp_ind]= temp[temp_ind];
	temp_ind+=blockDim.x;
	shared_temp_ind+=blockDim.x>>1;

	shared_bild[shared_temp_ind]= temp[temp_ind];
	temp_ind+=blockDim.x;
	shared_temp_ind+=(blockDim.x>>1);

	shared_bild[shared_temp_ind]= temp[temp_ind];

	
	__syncthreads();

	//wavelet step
	(*shared_upper)+=a*((*shared_lower) + shared_lower[1]);
	if (threadIdx.x<blockDim.x-1)shared_upper[1]+=a*( shared_lower[1]+  shared_lower[2]);
	else shared_upper[1]+=a*2*shared_lower[1];
	__syncthreads();

	if (threadIdx.x>0) (*shared_lower)+=b*(  shared_upper[-1]+  (*shared_upper));
	else (*shared_lower)+=b*2*(*shared_upper);
	shared_lower[1]+=b*( (*shared_upper)+ shared_upper[1]);
	__syncthreads();

	(*shared_upper)+=c*((*shared_lower) + shared_lower[1]);
	if (threadIdx.x<blockDim.x-1)shared_upper[1]+=c*( shared_lower[1]+  shared_lower[2]);
	else shared_upper[1]+=c*2*shared_lower[1];
	
	__syncthreads();
	
	if (threadIdx.x>0) (*shared_lower)+=d*(  shared_upper[-1]+ (*shared_upper));
	else (*shared_lower)+=d*2*(*shared_upper);
	shared_lower[1]+=d*(  (*shared_upper)+ shared_upper[1]);

	__syncthreads();


	//write back
	temp_ind=blockIdx.y*line_length+threadIdx.x;
	shared_temp_ind=threadIdx.x;
	bild[temp_ind]=shared_bild[shared_temp_ind]/K;

	temp_ind+=blockDim.x;
	shared_temp_ind+=blockDim.x;
	bild[temp_ind]=shared_bild[shared_temp_ind]/K;

	temp_ind+=blockDim.x;
	shared_temp_ind+=blockDim.x+1;
	bild[temp_ind]=shared_bild[shared_temp_ind]*K/2;

	temp_ind+=blockDim.x;
	shared_temp_ind+=blockDim.x;
	bild[temp_ind]=shared_bild[shared_temp_ind]*K/2;
		
}


__global__ void dwt_one_pixel_hor(float *bild,float*target, int line_length){
	target[blockIdx.y*line_length]=bild[blockIdx.y*temp_xdim];
}



///////////////////////////////////////////////////////////////////////////////
//vertical 9-7-wavelet kernels

extern __shared__ float shared_bild[];
__global__ void dwt_9_7_Kernel_ver_odd_shared(float *bild,float*target, int ydim,int line_length){
	int i =threadIdx.y;
	int j=blockIdx.x;
	int temp_ind;
	int upper_ind=(i<blockDim.y-1?i:i-1);
	int lower_ind=(i>0?(i-1):0);
	float * shared_upper=&shared_bild[(ydim+1)/2];

	temp_ind=i*(line_length<<1)+j;
	shared_bild[i]= bild[temp_ind];
	if (i<blockDim.y-1) shared_upper[i]= bild[line_length+temp_ind];

	__syncthreads();

	if (i<blockDim.y-1) shared_upper[i]+=a*(  shared_bild[i]+  shared_bild[i+1]);
	__syncthreads();
	shared_bild[i]+=b*(  shared_upper[lower_ind]+  shared_upper[upper_ind]);
	__syncthreads();
	if (i<blockDim.y-1)shared_upper[i]+=c*(  shared_bild[(i)]+  shared_bild[i+1]);
	__syncthreads();


	target[i*temp_xdim+j]=(shared_bild[i]+d*(  shared_upper[lower_ind]+ shared_upper[upper_ind]))/K;	
	if (i<blockDim.y-1)target[(i+(ydim+1)/2)*temp_xdim+j]=shared_upper[i]*K/2;	
}




extern __shared__ float shared_bild[];
__global__ void dwt_9_7_Kernel_56pix_x16_even_shared(float *bild,float*temp, int ydim,int line_length){
	int line_num =threadIdx.y+blockIdx.y*(blockDim.y-2)-1; //y-coordinate/4
	int j=threadIdx.x+blockIdx.x*blockDim.x;			   //x-coordinate
	int temp_ind;
	float ma,mb;										//data which doesnt need to be shared between threads
	float *shared_lower=&shared_bild[threadIdx.y*blockDim.x+threadIdx.x];
	float * shared_upper=&shared_lower[blockDim.y*blockDim.x]; //for H subb

	//load from global memory
	if (blockIdx.y>0&&blockIdx.y<gridDim.y-1)//normal threads in the inside
	{
			temp_ind=line_num*(line_length<<2)+j;
			(*shared_lower)= bild[temp_ind];
			temp_ind+=line_length;
			mb=bild[temp_ind];
			temp_ind+=line_length;
			ma=bild[temp_ind];
			temp_ind+=line_length;
			(*shared_upper)= bild[temp_ind];
	}else{
		if(threadIdx.y==0&&blockIdx.y==0)	//top border (is mirrored)
		{
			temp_ind=(line_num+2)*(line_length<<2)+j;
			(*shared_lower)= bild[temp_ind];
			temp_ind-=line_length;
			mb=bild[temp_ind];
			temp_ind-=line_length;
			ma=bild[temp_ind];
			temp_ind-=line_length;
			(*shared_upper)= bild[temp_ind];
		}
		else if (blockIdx.y==gridDim.y-1){	//blocks at bottom border
			if((line_num<<2)<ydim-3)	//normal threads close to bottom border
			{
				temp_ind=line_num*(line_length<<2)+j;
				(*shared_lower)= bild[temp_ind];
				temp_ind+=line_length;
				mb=bild[temp_ind];
				temp_ind+=line_length;
				ma=bild[temp_ind];
				temp_ind+=line_length;
				(*shared_upper)= bild[temp_ind];
			}else if((line_num<<2)<ydim+4)	//bottom border(with test if mirroring is necessary)
			{
				temp_ind=line_num<<2;//temp_ind is the line number in tile
				(*shared_lower)= bild[((temp_ind<ydim)?temp_ind:2*(ydim-1)-temp_ind)*line_length+j];
				temp_ind++;
				mb= bild[((temp_ind<ydim)?temp_ind:2*(ydim-1)-temp_ind)*line_length+j];
				temp_ind++;
				ma= bild[((temp_ind<ydim)?temp_ind:2*(ydim-1)-temp_ind)*line_length+j];
				temp_ind++;
				(*shared_upper)= bild[((temp_ind<ydim)?temp_ind:2*(ydim-1)-temp_ind)*line_length+j];
			}else{//just to do something
				temp_ind=(blockIdx.y*(blockDim.y-2))*(line_length<<2)+j;
				(*shared_lower)= bild[temp_ind];
				temp_ind+=line_length;
				mb=bild[temp_ind];
				temp_ind+=line_length;
				ma=bild[temp_ind];
				temp_ind+=line_length;
				(*shared_upper)= bild[temp_ind];
			}

		}else{									//normal threads close to top border
			temp_ind=line_num*(line_length<<2)+j;
			(*shared_lower)= bild[temp_ind];
			temp_ind+=line_length;
			mb=bild[temp_ind];
			temp_ind+=line_length;
			ma=bild[temp_ind];
			temp_ind+=line_length;
			(*shared_upper)= bild[temp_ind];
		}
	}
		__syncthreads();


		//wavelet-step
		mb+=a*((*shared_lower) + ma);
		(*shared_upper)+=a*(ma+  shared_lower[blockDim.x]);

		__syncthreads();

		(*shared_lower)+=b*(shared_upper[-(int)blockDim.x]+  mb);
		ma+=b*(mb+ (*shared_upper));

		__syncthreads();
		mb+=c*((*shared_lower) + ma);
		(*shared_upper)+=c*(ma+  shared_lower[blockDim.x]);

		__syncthreads();

		(*shared_lower)+=d*(shared_upper[-(int)blockDim.x]+  mb);
		ma+=d*(mb+ (*shared_upper));

		__syncthreads();
		(*shared_lower)/=K;
		ma/=K;
		mb*=K/2;
		(*shared_upper)*=K/2;



		if (blockIdx.y<gridDim.y-1){
			//write back to global memory (except threads at boundary)
			if(threadIdx.y!=0&&threadIdx.y!=blockDim.y-1)
			{
				//L-subb
				temp_ind=line_num*(temp_xdim<<1)+j;
				temp[temp_ind]=(*shared_lower);
				temp_ind+=temp_xdim;
				temp[temp_ind]=ma;

				//H-subb
				temp_ind+=temp_xdim*((ydim+1)/2);
				temp[temp_ind]=(*shared_upper);	
				temp_ind-=temp_xdim;
				temp[temp_ind]=mb;	
			}
		}else if((line_num<<2)<ydim&&threadIdx.y!=0&&threadIdx.y!=blockDim.y-1)
		{
			if((line_num<<2)<ydim-3)
			{
				//L-subb
				temp_ind=line_num*(temp_xdim<<1)+j;
				temp[temp_ind]=(*shared_lower);
				temp_ind+=temp_xdim;
				temp[temp_ind]=ma;

				//H-subb
				temp_ind+=temp_xdim*((ydim+1)/2);
				temp[temp_ind]=(*shared_upper);	
				temp_ind-=temp_xdim;
				temp[temp_ind]=mb;	
			}else{
				//L-subb
				temp_ind=line_num*(temp_xdim<<1)+j;
				temp[temp_ind]=(*shared_lower);				//(line_num<<2)<ydim was already checked above
				temp_ind+=temp_xdim;
				if ((line_num<<2)+2<ydim)temp[temp_ind]=ma;

				//H-subb
				temp_ind+=temp_xdim*((ydim+1)/2);
				if ((line_num<<2)+3<ydim)temp[temp_ind]=(*shared_upper);	
				temp_ind-=temp_xdim;
				if ((line_num<<2)+1<ydim)temp[temp_ind]=mb;	
			}
		}
		

}


extern __shared__ float shared_bild[];
__global__ void dwt_9_7_Kernel_0mod4_x16_even_shared(float *bild,float*temp, int ydim,int line_length){
	int line_num =threadIdx.y+blockIdx.y*(blockDim.y-2)-1; //y-coordinate/4
	int j=threadIdx.x+blockIdx.x*blockDim.x;			   //x-coordinate
	int temp_ind;
	float ma,mb;										//data which doesnt need to be shared between threads
	float *shared_lower=&shared_bild[threadIdx.y*blockDim.x+threadIdx.x];
	float * shared_upper=&shared_lower[blockDim.y*blockDim.x]; //for H subb

	//load from global memory
	if (blockIdx.y>0&&blockIdx.y<gridDim.y-1)//normal threads in the inside
	{
			temp_ind=line_num*(line_length<<2)+j;
			(*shared_lower)= bild[temp_ind];
			temp_ind+=line_length;
			mb=bild[temp_ind];
			temp_ind+=line_length;
			ma=bild[temp_ind];
			temp_ind+=line_length;
			(*shared_upper)= bild[temp_ind];
	}else{
		if(threadIdx.y==0&&blockIdx.y==0)	//top border(is mirrored)
		{
			temp_ind=(line_num+2)*(line_length<<2)+j;
			(*shared_lower)= bild[temp_ind];
			temp_ind-=line_length;
			mb=bild[temp_ind];
			temp_ind-=line_length;
			ma=bild[temp_ind];
			temp_ind-=line_length;
			(*shared_upper)= bild[temp_ind];
		}
		else if (blockIdx.y==gridDim.y-1){	//blocks at bottom border
			if((line_num<<2)<ydim-3)	//normal threads close to bottom border
			{
				temp_ind=line_num*(line_length<<2)+j;
				(*shared_lower)= bild[temp_ind];
				temp_ind+=line_length;
				mb=bild[temp_ind];
				temp_ind+=line_length;
				ma=bild[temp_ind];
				temp_ind+=line_length;
				(*shared_upper)= bild[temp_ind];
			}else if((line_num<<2)<ydim+4)	//bottom border(with test if mirroring is necessary)
			{
				temp_ind=((line_num<<2)-2)*line_length+j;
				(*shared_lower)= bild[temp_ind];
				temp_ind-=line_length;
				mb=bild[temp_ind];
				temp_ind-=line_length;
				ma=bild[temp_ind];
				temp_ind-=line_length;
				(*shared_upper)= bild[temp_ind];
			}else{//just to do something
				//temp_ind=(line_num-blockDim.y*4+8)*(line_length<<2)+j;
				temp_ind=(blockIdx.y*(blockDim.y-2))*(line_length<<2)+j;
				(*shared_lower)= bild[temp_ind];
				temp_ind+=line_length;
				mb=bild[temp_ind];
				temp_ind+=line_length;
				ma=bild[temp_ind];
				temp_ind+=line_length;
				(*shared_upper)= bild[temp_ind];
			}

		}else{									//normal threads close to top border
			temp_ind=line_num*(line_length<<2)+j;
			(*shared_lower)= bild[temp_ind];
			temp_ind+=line_length;
			mb=bild[temp_ind];
			temp_ind+=line_length;
			ma=bild[temp_ind];
			temp_ind+=line_length;
			(*shared_upper)= bild[temp_ind];
		}
	}
		__syncthreads();


		//wavelet-step
		mb+=a*((*shared_lower) + ma);
		(*shared_upper)+=a*(ma+  shared_lower[blockDim.x]);

		__syncthreads();

		(*shared_lower)+=b*(shared_upper[-(int)blockDim.x]+  mb);
		ma+=b*(mb+ (*shared_upper));

		__syncthreads();
		mb+=c*((*shared_lower) + ma);
		(*shared_upper)+=c*(ma+  shared_lower[blockDim.x]);

		__syncthreads();

		(*shared_lower)+=d*(shared_upper[-(int)blockDim.x]+  mb);
		ma+=d*(mb+ (*shared_upper));

		__syncthreads();
		(*shared_lower)/=K;
		ma/=K;
		mb*=K/2;
		(*shared_upper)*=K/2;

	
		//write back to global memory (except threads at boundary)
		if(threadIdx.y!=0&&threadIdx.y!=blockDim.y-1&&(line_num<<2)<ydim)
		{
			//L-subb
			temp_ind=line_num*(temp_xdim<<1)+j;
			temp[temp_ind]=(*shared_lower);
			temp_ind+=temp_xdim;
			temp[temp_ind]=ma;

			//H-subb
			temp_ind+=temp_xdim*((ydim+1)/2);
			temp[temp_ind]=(*shared_upper);	
			temp_ind-=temp_xdim;
			temp[temp_ind]=mb;	
		}
}


extern __shared__ float shared_bild[];
__global__ void dwt_9_7_Kernel_56pix_notx16_even_shared(float *bild,float*temp,int xdim, int ydim,int line_length){
	int line_num =threadIdx.y+blockIdx.y*(blockDim.y-2)-1; //y-coordinate/4
	int j=threadIdx.x+blockIdx.x*blockDim.x;			   //x-coordinate
	int temp_ind;
	float ma,mb;										//data which doesnt need to be shared between threads
	float *shared_lower=&shared_bild[threadIdx.y*blockDim.x+threadIdx.x];
	float* shared_upper=&shared_lower[blockDim.y*blockDim.x]; //for H subb

//load from global memory
	if(j<xdim){		
		if (blockIdx.y>0&&blockIdx.y<gridDim.y-1)//normal threads in the inside
		{
				temp_ind=line_num*(line_length<<2)+j;
				(*shared_lower)= bild[temp_ind];
				temp_ind+=line_length;
				mb=bild[temp_ind];
				temp_ind+=line_length;
				ma=bild[temp_ind];
				temp_ind+=line_length;
				(*shared_upper)= bild[temp_ind];
		}else{
			if(threadIdx.y==0&&blockIdx.y==0)	//top border(is mirrored)
			{
				temp_ind=(line_num+2)*(line_length<<2)+j;
				(*shared_lower)= bild[temp_ind];
				temp_ind-=line_length;
				mb=bild[temp_ind];
				temp_ind-=line_length;
				ma=bild[temp_ind];
				temp_ind-=line_length;
				(*shared_upper)= bild[temp_ind];
			}
			else if (blockIdx.y==gridDim.y-1){	//blocks at bottom border
				if((line_num<<2)<ydim-3)	//normal threads close to bottom border
				{
					temp_ind=line_num*(line_length<<2)+j;
					(*shared_lower)= bild[temp_ind];
					temp_ind+=line_length;
					mb=bild[temp_ind];
					temp_ind+=line_length;
					ma=bild[temp_ind];
					temp_ind+=line_length;
					(*shared_upper)= bild[temp_ind];
				}else if((line_num<<2)<ydim+4)	//bottom border(with test if mirroring is necessary)
				{
					temp_ind=line_num<<2;//temp_ind is the line number in tile
					(*shared_lower)= bild[((temp_ind<ydim)?temp_ind:2*(ydim-1)-temp_ind)*line_length+j];
					temp_ind++;
					mb= bild[((temp_ind<ydim)?temp_ind:2*(ydim-1)-temp_ind)*line_length+j];
					temp_ind++;
					ma= bild[((temp_ind<ydim)?temp_ind:2*(ydim-1)-temp_ind)*line_length+j];
					temp_ind++;
					(*shared_upper)= bild[((temp_ind<ydim)?temp_ind:2*(ydim-1)-temp_ind)*line_length+j];
				}else{//just to do something
					//temp_ind=(line_num-blockDim.y*4+8)*(line_length<<2)+j;
					temp_ind=(blockIdx.y*(blockDim.y-2))*(line_length<<2)+j;
					(*shared_lower)= bild[temp_ind];
					temp_ind+=line_length;
					mb=bild[temp_ind];
					temp_ind+=line_length;
					ma=bild[temp_ind];
					temp_ind+=line_length;
					(*shared_upper)= bild[temp_ind];
				}

			}else{									//normal threads close to top border
				temp_ind=line_num*(line_length<<2)+j;
				(*shared_lower)= bild[temp_ind];
				temp_ind+=line_length;
				mb=bild[temp_ind];
				temp_ind+=line_length;
				ma=bild[temp_ind];
				temp_ind+=line_length;
				(*shared_upper)= bild[temp_ind];
			}
		}
	}
		__syncthreads();


//wavelet-step
		mb+=a*((*shared_lower) + ma);
		(*shared_upper)+=a*(ma+  shared_lower[blockDim.x]);

		__syncthreads();

		(*shared_lower)+=b*(shared_upper[-(int)blockDim.x]+  mb);
		ma+=b*(mb+ (*shared_upper));

		__syncthreads();
		mb+=c*((*shared_lower) + ma);
		(*shared_upper)+=c*(ma+  shared_lower[blockDim.x]);

		__syncthreads();

		(*shared_lower)+=d*(shared_upper[-(int)blockDim.x]+  mb);
		ma+=d*(mb+ (*shared_upper));

		__syncthreads();
		(*shared_lower)/=K;
		ma/=K;
		mb*=K/2;
		(*shared_upper)*=K/2;



//write back to global memory (except threads at boundary)
		if(j<xdim){
			if (blockIdx.y<gridDim.y-1){
				
				if(threadIdx.y!=0&&threadIdx.y!=blockDim.y-1)
				{
					//L-subb
					temp_ind=line_num*(temp_xdim<<1)+j;
					temp[temp_ind]=(*shared_lower);
					temp_ind+=temp_xdim;
					temp[temp_ind]=ma;

					//H-subb
					temp_ind+=temp_xdim*((ydim+1)/2);
					temp[temp_ind]=(*shared_upper);	
					temp_ind-=temp_xdim;
					temp[temp_ind]=mb;	
				}
			}else if((line_num<<2)<ydim&&threadIdx.y!=0&&threadIdx.y!=blockDim.y-1)
			{
				if((line_num<<2)<ydim-3)
				{
					//L-subb
					temp_ind=line_num*(temp_xdim<<1)+j;
					temp[temp_ind]=(*shared_lower);
					temp_ind+=temp_xdim;
					temp[temp_ind]=ma;

					//H-subb
					temp_ind+=temp_xdim*((ydim+1)/2);
					temp[temp_ind]=(*shared_upper);	
					temp_ind-=temp_xdim;
					temp[temp_ind]=mb;	
				}else{
					//L-subb
					temp_ind=line_num*(temp_xdim<<1)+j;
					temp[temp_ind]=(*shared_lower);				//(line_num<<4)<ydim was already checked above
					temp_ind+=temp_xdim;
					if ((line_num<<2)+2<ydim)temp[temp_ind]=ma;

					//H-subb
					temp_ind+=temp_xdim*((ydim+1)/2);
					if ((line_num<<2)+3<ydim)temp[temp_ind]=(*shared_upper);	
					temp_ind-=temp_xdim;
					if ((line_num<<2)+1<ydim)temp[temp_ind]=mb;	
				}
			}
		
		}
}


extern __shared__ float shared_bild[];
__global__ void dwt_9_7_Kernel_ver_even_shared(float *bild,float*target,int ydim,int line_length){
	int i =threadIdx.y;
	int j=blockIdx.x;
	int temp_ind;
	int upper_ind=(i<blockDim.y-1?i+1:i);
	int lower_ind=(i>0?(i-1):0);
	float * shared_upper=&shared_bild[ydim/2];

	temp_ind=i*(line_length<<1)+j;
	shared_bild[i]= bild[temp_ind];
	shared_upper[i]= bild[line_length+temp_ind];
	__syncthreads();

	shared_upper[i]+=a*(  shared_bild[i]+  shared_bild[upper_ind]);
	__syncthreads();
	shared_bild[i]+=b*(  shared_upper[lower_ind]+  shared_upper[i]);
	__syncthreads();
	shared_upper[i]+=c*(  shared_bild[(i)]+  shared_bild[upper_ind]);
	__syncthreads();


	target[i*temp_xdim+j]=(shared_bild[i]+d*(  shared_upper[lower_ind]+ shared_upper[i]))/K;
	target[(i+ydim/2)*temp_xdim+j]=shared_upper[i]*K/2;	
}



__global__ void dwt_one_pixel_ver(float *bild,float*target){
	target[blockIdx.x]=bild[blockIdx.x];
}