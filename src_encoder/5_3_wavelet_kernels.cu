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

/* this file contains all the kernels to calculate the 5-3-wavelet transform.
They are separated into horizontal and vertical kernels (because they have a different access to the data).
For each direction there are various kernels which are called depending on the dimensions of the image.
*/



///////////////////////////////////////////////////////////////////////////////
//horizontal 5-3-wavelet kernels


extern __shared__ int shared_bild_rev[];
__global__ void dwt_5_3_Kernel_hor_0mod4_shared(int*temp,int *bild, int line_length){
	int temp_ind;
	int shared_temp_ind;
	int *shared_lower=&shared_bild_rev[threadIdx.x<<1];
	int *shared_upper=&shared_lower[blockDim.x*2+1];



	//read from global memory
	temp_ind=blockIdx.y*temp_xdim+threadIdx.x;
	shared_temp_ind=((threadIdx.x>>1)+(blockDim.x*2+1)*(threadIdx.x&1));
	shared_bild_rev[shared_temp_ind]= temp[temp_ind];
	temp_ind+=blockDim.x;
	shared_temp_ind+=blockDim.x>>1;

	shared_bild_rev[shared_temp_ind]= temp[temp_ind];
	temp_ind+=blockDim.x;
	shared_temp_ind+=blockDim.x>>1;

	shared_bild_rev[shared_temp_ind]= temp[temp_ind];
	temp_ind+=blockDim.x;
	shared_temp_ind+=(blockDim.x>>1);

	shared_bild_rev[shared_temp_ind]= temp[temp_ind];


	__syncthreads();

	//wavelet step
	(*shared_upper)-=((*shared_lower) + shared_lower[1])>>1;
	if (threadIdx.x<blockDim.x-1)shared_upper[1]-=( shared_lower[1]+  shared_lower[2])>>1;
	else shared_upper[1]-=shared_lower[1];
	__syncthreads();

	if (threadIdx.x>0) (*shared_lower)+=(  shared_upper[-1]+  (*shared_upper)+2)>>2;
	else (*shared_lower)+=((*shared_upper)+1)>>1;
	shared_lower[1]+=( (*shared_upper)+ shared_upper[1]+2)>>2;

	__syncthreads();

	//write back
	temp_ind=blockIdx.y*line_length+threadIdx.x;
	shared_temp_ind=threadIdx.x;
	bild[temp_ind]=shared_bild_rev[shared_temp_ind];

	temp_ind+=blockDim.x;
	shared_temp_ind+=blockDim.x;
	bild[temp_ind]=shared_bild_rev[shared_temp_ind];

	temp_ind+=blockDim.x;
	shared_temp_ind+=blockDim.x+1;

	bild[temp_ind]=shared_bild_rev[shared_temp_ind];
	temp_ind+=blockDim.x;
	shared_temp_ind+=blockDim.x;
	bild[temp_ind]=shared_bild_rev[shared_temp_ind];

		
}


extern __shared__ int shared_bild_rev[];
__global__ void dwt_5_3_Kernel_hor_even_shared(int *bild,int*target, int xdim,int line_length){
	
	int i =threadIdx.x;
	int j=blockIdx.y;
	int temp_ind;
	int upper_ind=(i<blockDim.x-1?i+1:i);
	int lower_ind=(i>0?(i-1):0);
	int * shared_upper=&shared_bild_rev[xdim/2];

	temp_ind=j*temp_xdim+(i<<1);
	shared_bild_rev[i]= bild[temp_ind];
	shared_upper[i]= bild[1+temp_ind];

	__syncthreads();

	shared_upper[i]-=(shared_bild_rev[i]+  shared_bild_rev[upper_ind])>>1;
	__syncthreads();

	temp_ind=i+j*line_length;
	target[temp_ind]=shared_bild_rev[i]+(( shared_upper[lower_ind]+  shared_upper[i]+2)>>2);		
	target[xdim/2+temp_ind]=shared_upper[i];
		
}


extern __shared__ int shared_bild_rev[];
__global__ void dwt_5_3_Kernel_hor_odd_shared(int *bild,int*target, int xdim,int line_length){
	int i =threadIdx.x;
	int j=blockIdx.y;
	int temp_ind;
	int upper_ind=(i<blockDim.x-1?i:i-1);
	int lower_ind=(i>0?(i-1):0);
	int * shared_upper=&shared_bild_rev[(xdim+1)/2];

	temp_ind=j*temp_xdim+(i<<1);
	shared_bild_rev[i]= bild[temp_ind];
	if (i<blockDim.x-1)shared_upper[i]= bild[1+temp_ind];

	__syncthreads();

	if (i<blockDim.x-1) shared_upper[i]-=(shared_bild_rev[i]+ shared_bild_rev[i+1])>>1;
	__syncthreads();


	temp_ind=i+j*line_length;
	target[temp_ind]=shared_bild_rev[i]+((shared_upper[lower_ind]+shared_upper[upper_ind]+2)>>2);

	if (i<blockDim.x-1)target[(xdim+1)/2+temp_ind]=shared_upper[i];	
}


//for only one pixel there is nothing to do
__global__ void dwt_one_pixel_hor_rev(int *bild,int*target, int line_length){
	target[blockIdx.y*line_length]=bild[blockIdx.y*temp_xdim];
}


///////////////////////////////////////////////////////////////////////////////
//vertical 5-3-wavelet kernels

extern __shared__ int shared_bild_rev[];
__global__ void dwt_5_3_Kernel_ver_odd_shared(int *bild,int*target, int line_length){
	int i =threadIdx.y;
	int j=blockIdx.x;
	int temp_ind;
	int upper_ind=(i<blockDim.y-1?i:i-1);
	int lower_ind=(i>0?(i-1):0);
	int * shared_upper=&shared_bild_rev[blockDim.y];

	temp_ind=i*(line_length<<1)+j;
	shared_bild_rev[i]= bild[temp_ind];
	if (i<blockDim.y-1) shared_upper[i]= bild[line_length+temp_ind];

	__syncthreads();

	if (i<blockDim.y-1) shared_upper[i]-=(  shared_bild_rev[i]+  shared_bild_rev[i+1])>>1;
	__syncthreads();

	target[i*temp_xdim+j]=shared_bild_rev[i]+((  shared_upper[lower_ind]+ shared_upper[upper_ind]+2)>>2);	
	if (i<blockDim.y-1)target[(i+blockDim.y)*temp_xdim+j]=shared_upper[i];	
}


extern __shared__ int shared_bild_rev[];
__global__ void dwt_5_3_Kernel_ver_even_shared(int *bild,int*target, int line_length){
	int i =threadIdx.y;
	int j=blockIdx.x;
	int temp_ind;
	int upper_ind=(i<blockDim.y-1?i+1:i);
	int lower_ind=(i>0?(i-1):0);
	int * shared_upper=&shared_bild_rev[blockDim.y];

	temp_ind=i*(line_length<<1)+j;
	shared_bild_rev[i]= bild[temp_ind];
	shared_upper[i]= bild[line_length+temp_ind];


	__syncthreads();

	shared_upper[i]-=(  shared_bild_rev[i]+  shared_bild_rev[upper_ind])>>1;
	__syncthreads();

	target[i*temp_xdim+j]=(shared_bild_rev[i]+((  shared_upper[lower_ind]+ shared_upper[i]+2)>>2));
	target[(i+blockDim.y)*temp_xdim+j]=shared_upper[i];	
}




extern __shared__ int shared_bild_rev[];
__global__ void dwt_5_3_Kernel_56pix_x16_even_shared(int *bild,int*temp, int ydim,int line_length){
	int line_num =threadIdx.y+blockIdx.y*(blockDim.y-2)-1; //y-coordinate/4
	int j=threadIdx.x+blockIdx.x*blockDim.x;			   //x-coordinate
	int temp_ind;
	int ma,mb;										//data which doesnt need to be shared between threads
	int *shared_lower=&shared_bild_rev[threadIdx.y*blockDim.x+threadIdx.x];
	int * shared_upper=&shared_lower[blockDim.y*blockDim.x]; //for H subb

	//load from global memory
	if (blockIdx.y>0&&blockIdx.y<gridDim.y-1)//totally normal threads in the inside
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
				temp_ind=line_num<<2;//temp_ind is line number in tile
				(*shared_lower)= bild[((temp_ind<ydim)?temp_ind:2*(ydim-1)-temp_ind)*line_length+j];
				temp_ind++;
				mb= bild[((temp_ind<ydim)?temp_ind:2*(ydim-1)-temp_ind)*line_length+j];
				temp_ind++;
				ma= bild[((temp_ind<ydim)?temp_ind:2*(ydim-1)-temp_ind)*line_length+j];
				temp_ind++;
				(*shared_upper)= bild[((temp_ind<ydim)?temp_ind:2*(ydim-1)-temp_ind)*line_length+j];
			}else{//just to do something
				temp_ind=(blockIdx.y*(blockDim.y-2))*(line_length<<2)+j;
				//temp_ind=(line_num-blockDim.y*4+8)*(line_length<<2)+j;
				(*shared_lower)= bild[temp_ind];
				temp_ind+=line_length;
				mb=bild[temp_ind];
				temp_ind+=line_length;
				ma=bild[temp_ind];
				temp_ind+=line_length;
				(*shared_upper)= bild[temp_ind];
			}

		}else{									//normal threads close to top bottom
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
		mb-=((*shared_lower) + ma)>>1;
		(*shared_upper)-=(ma+  shared_lower[blockDim.x])>>1;

		__syncthreads();

		(*shared_lower)+=(shared_upper[- (int)blockDim.x]+  mb+2)>>2;
		ma+=(mb+ (*shared_upper)+2)>>2;
		


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
				temp[temp_ind]=(*shared_lower);				//(line_num<<4)<ydim was checked above
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


extern __shared__ int shared_bild_rev[];
__global__ void dwt_5_3_Kernel_0mod4_x16_even_shared(int *bild,int*temp, int ydim,int line_length){
	int line_num =threadIdx.y+blockIdx.y*(blockDim.y-2)-1; //y-coordinate/4
	int j=threadIdx.x+blockIdx.x*blockDim.x;			   //x-coordinate
	int temp_ind;
	int ma,mb;										//data which doesnt need to be shared between threads
	int *shared_lower=&shared_bild_rev[threadIdx.y*blockDim.x+threadIdx.x];
	int * shared_upper=&shared_lower[blockDim.y*blockDim.x]; //for H subb

	//load from global memory
	if (blockIdx.y>0&&blockIdx.y<gridDim.y-1)//completely normal threads in the inside
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
				temp_ind=((line_num<<2)-2)*line_length+j;
				(*shared_lower)= bild[temp_ind];
				temp_ind-=line_length;
				mb=bild[temp_ind];
				temp_ind-=line_length;
				ma=bild[temp_ind];
				temp_ind-=line_length;
				(*shared_upper)= bild[temp_ind];
			}else{//just do do something
				temp_ind=(blockIdx.y*(blockDim.y-2))*(line_length<<2)+j;
				//temp_ind=(line_num-blockDim.y*4+8)*(line_length<<2)+j;
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
		mb-=((*shared_lower) + ma)>>1;
		(*shared_upper)-=(ma+  shared_lower[blockDim.x])>>1;

		__syncthreads();

		(*shared_lower)+=(shared_upper[- (int)blockDim.x]+  mb+2)>>2;
		ma+=(mb+ (*shared_upper)+2)>>2;


	
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


extern __shared__ int shared_bild_rev[];
__global__ void dwt_5_3_Kernel_56pix_notx16_even_shared(int *bild,int*temp,int xdim, int ydim,int line_length){
	int line_num =threadIdx.y+blockIdx.y*(blockDim.y-2)-1; //y-coordinate/4
	int j=threadIdx.x+blockIdx.x*blockDim.x;			   //x-coordinate
	int temp_ind;
	int ma,mb;										//data which doesnt need to be shared between threads
	int *shared_lower=&shared_bild_rev[threadIdx.y*blockDim.x+threadIdx.x];
	int * shared_upper=&shared_lower[blockDim.y*blockDim.x]; //for H subb
	if(j<xdim){
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
					//temp_ind=(line_num-blockDim.y*4+8)*(line_length<<2)+j;
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
		mb-=((*shared_lower) + ma)>>1;
		(*shared_upper)-=(ma+  shared_lower[blockDim.x])>>1;

		__syncthreads();

		(*shared_lower)+=(shared_upper[- (int)blockDim.x]+  mb+2)>>2;
		ma+=(mb+ (*shared_upper)+2)>>2;

		__syncthreads();


		if(j<xdim){
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


//for only one pixel there is nothing to do
__global__ void dwt_one_pixel_ver_rev(int *bild,int*target){
	target[blockIdx.x]=bild[blockIdx.x];
}
