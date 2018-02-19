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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cutil.h>
#include <cutil_inline.h>
#include "device.h"
#include "encoder_main.h"


//sets the device as working device with best computing capability
void choose_highest_capability (){
	int deviceCount, dev;
	int major=0, minor=0, best;
	cudaDeviceProp deviceProp;

	cutilSafeCall(cudaGetDeviceCount(&deviceCount));
    
	if (deviceCount == 0){
        printf("ERROR - There is no device supporting CUDA\n");
		return;
	}
	
	if (deviceCount == 1){
		cutilSafeCall(cudaGetDeviceProperties(&deviceProp, 0));
		printf("***   CUJ2K running on \"%s\"   ***\n\n", deviceProp.name);
		return;
	}
    
	for (dev = 0; dev < deviceCount; ++dev) {
        cutilSafeCall(cudaGetDeviceProperties(&deviceProp, dev));
		
		if (major < deviceProp.major){
			major = deviceProp.major;
			best = dev;
		}

		if (minor < deviceProp.minor){
			minor = deviceProp.minor;
			best = dev;
		}
	}
	cutilSafeCall (cudaSetDevice(best));
	cutilSafeCall(cudaGetDeviceProperties(&deviceProp, best));
	printf("***   CUJ2K running on \"%s\"   ***\n\n", deviceProp.name);
}

void choose_fastest_gpu(){
	int deviceCount, dev;
	int fastest=0, best;
	cudaDeviceProp deviceProp;

	cutilSafeCall(cudaGetDeviceCount(&deviceCount));
    
	if (deviceCount == 0){
        printf("There is no device supporting CUDA\n");
		return;
	}
	
	if (deviceCount == 1){
		cutilSafeCall(cudaGetDeviceProperties(&deviceProp, 0));
		printf("***   CUJ2K running on \"%s\"   ***\n\n", deviceProp.name);
		return;
	}
    
	for (dev = 0; dev < deviceCount; ++dev) {
        cutilSafeCall(cudaGetDeviceProperties(&deviceProp, dev));
		
		if (fastest < deviceProp.clockRate){
			fastest = deviceProp.clockRate;
			best = dev;
		}
	}
	cutilSafeCall (cudaSetDevice(best));
	cutilSafeCall(cudaGetDeviceProperties(&deviceProp, best));
	printf("***   CUJ2K running on \"%s\"   ***\n\n", deviceProp.name);
}

void choose_biggest_memory(){
	int deviceCount, dev;
	int best;
	unsigned int biggest=0;
	cudaDeviceProp deviceProp;

	cutilSafeCall(cudaGetDeviceCount(&deviceCount));
    
	if (deviceCount == 0){
        printf("There is no device supporting CUDA\n");
		return;
	}
	
	if (deviceCount == 1){
		cutilSafeCall(cudaGetDeviceProperties(&deviceProp, 0));
		printf("***   CUJ2K running on \"%s\"   ***\n\n", deviceProp.name);
		return;
	}
    
	for (dev = 0; dev < deviceCount; ++dev) {
        cutilSafeCall(cudaGetDeviceProperties(&deviceProp, dev));
		
		if (biggest < deviceProp.totalGlobalMem){
			biggest = deviceProp.totalGlobalMem;
			best = dev;
		}
	}
	cutilSafeCall (cudaSetDevice(best));
	cutilSafeCall(cudaGetDeviceProperties(&deviceProp, best));
	printf("***   CUJ2K running on \"%s\"   ***\n\n", deviceProp.name);
}

/*chooses first gpu with compute capability 1.1 or higher.
RETURNS:0 - stream compatible gpu available
		1 - no streaming compatible gpu*/
int choose_stream_gpu (int *timeout){
#ifdef NO_DEVICE_PROP
	//cudaGetDeviceProperties() not implemented => be happy with current GPU
	printf("***   CUJ2K %s   ***\n\n", CUJ2K_VERSION_STR);
	return 0;
#else
	int deviceCount, dev;
	int major=0, minor=0, best;
	cudaDeviceProp deviceProp;

	cutilSafeCall(cudaGetDeviceCount(&deviceCount));
    
	if (deviceCount == 0){
        printf("ERROR - There is no device supporting CUDA\n");
		return 1;
	}
	
	for (dev = 0; dev < deviceCount; ++dev) {
        cutilSafeCall(cudaGetDeviceProperties(&deviceProp, dev));
		
		if (major < deviceProp.major){
			major = deviceProp.major;
			best = dev;
		}

		if (minor < deviceProp.minor){
			minor = deviceProp.minor;
			best = dev;
		}
	}
	
	if ((major >= 1) && (minor >= 1)){
		cutilSafeCall (cudaSetDevice(best));	
		cutilSafeCall(cudaGetDeviceProperties(&deviceProp, best));
		*timeout = deviceProp.kernelExecTimeoutEnabled;
		//printf("timeout=%d\n", *timeout);

		printf("***   CUJ2K %s running on \"%s\"   ***\n\n", CUJ2K_VERSION_STR,
			deviceProp.name);
		return 0;
	}
	else{
		cutilSafeCall(cudaGetDeviceProperties(&deviceProp, 0));
		printf("Error: \"%s\" is NOT streaming-compatible.\nGPU with compute capability 1.1 or higher needed\n\n", deviceProp.name);
		return 1;
	}
#endif
}

int user_set_device(int device) {
#ifdef NO_DEVICE_PROP
	//no checking, we can't determine compute capability
	printf("***   CUJ2K %s   ***\n\n", CUJ2K_VERSION_STR);
	cutilSafeCall (cudaSetDevice(device));
	return 0;
#else
	int deviceCount;
	cudaDeviceProp deviceProp;

	cutilSafeCall(cudaGetDeviceCount(&deviceCount));
	if(device < 0  ||  device >= deviceCount) {
		printf("Error: device number out of range.\n\n");
		list_devices();
		return 1;
	}

	cutilSafeCall(cudaGetDeviceProperties(&deviceProp, device));
	if(deviceProp.major==1  &&  deviceProp.minor < 1) {
		printf("Error: device \"%s\" is NOT streaming-compatible.\nGPU with compute capability 1.1 or higher needed\n\n", deviceProp.name);
		list_devices();
		return 1;
	}

	cutilSafeCall (cudaSetDevice(device));
	printf("***   CUJ2K %s running on \"%s\"   ***\n\n", CUJ2K_VERSION_STR,
		deviceProp.name);
	return 0;
#endif
}

void list_devices() {
	int deviceCount;
	cudaDeviceProp deviceProp;

	printf("CUDA devices:\n");
	cutilSafeCall(cudaGetDeviceCount(&deviceCount));
	if(deviceCount == 0)
		printf("No CUDA-enabled GPU was found.\n\n");
	else {
		for(int i = 0; i < deviceCount; i++) {
			cutilSafeCall(cudaGetDeviceProperties(&deviceProp, i));
			printf("device #%d: \"%s\", compute capability %d.%d\n", i, deviceProp.name, deviceProp.major, deviceProp.minor);
		}
		printf("\nNote: compute capability >= 1.1 is required for this program.\n\n");
	}
}
