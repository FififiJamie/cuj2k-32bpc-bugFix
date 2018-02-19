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

#include <stdlib.h>
#include "BufferedWriter.h"
#include "file_access.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#define STANDARD_BUFFER_SIZE 512



const unsigned char ZeroMasks[] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
const unsigned char OneMasks[] = {0x7F, 0xBF, 0xDF, 0xEF, 0xF7, 0xFB, 0xFD, 0xFE};



/* Initialize Buffer with standard size*/
void InitializeBuffer(struct Buffer *data){
	data->Data = (unsigned char*) malloc(STANDARD_BUFFER_SIZE);
	data->ByteCounter = 0;
	data->BitCounter = 0;
	data->size = STANDARD_BUFFER_SIZE;
	data->temp = 0x00;
	
}

/*returns a copy of the char pointer of the given buffer */
unsigned char *CopyBuffer(struct Buffer* data){
	unsigned char *output = (unsigned char*) malloc(data->ByteCounter);
	memcpy(output, data->Data, data->ByteCounter);
	return output;
}

// enlarges Buffer to specified size
void EnlargeBuffer(struct Buffer *Data, int new_size) {
	unsigned char *temp;
	temp = Data->Data;
	Data->Data = (unsigned char*) malloc(new_size);
	memcpy(Data->Data, temp, Data->size);
	Data->size = new_size;
	free(temp);
}

/* doubles Buffer if its smaller then needed */
/*void DoubleBuffer(struct Buffer *Data){
	EnlargeBuffer(Data, Data->size * 2);
}*/

/* writes one Byte to the Buffer, works only with empty temp */
void BufferByte(struct Buffer *Data, unsigned char value){
	assert(Data->BitCounter == 0);
	Data->Data[Data->ByteCounter] = value;
	Data->ByteCounter++;
	if(Data->ByteCounter == Data->size){
		EnlargeBuffer(Data, Data->size * 2);
	}
}

void BufferShort(struct Buffer *data, unsigned short value){
	BufferByte(data, value >> 8);
	BufferByte(data, value);
}

void BufferInt(struct Buffer *data, unsigned int value){
	BufferByte(data, value >> 24);
	BufferByte(data, value >> 16);
	BufferByte(data, value >> 8);
	BufferByte(data, value);
}

/* writes one ore zero to the buffers temp and writes the temp if 1 byte is reached */
void BufferZero(struct Buffer *Data){
	Data->BitCounter++;
	if(Data->BitCounter == 8){
		Data->BitCounter = 0;
		BufferByte(Data, Data->temp);
		if(Data->temp == 0xFF){
			Data->BitCounter = 1;
		}
		Data->temp = 0x00;
	}
}
void BufferOne(struct Buffer *Data){
	Data->temp |= 1<<(7 - Data->BitCounter);
	Data->BitCounter++;
	if(Data->BitCounter == 8){
		Data->BitCounter = 0;
		BufferByte(Data, Data->temp);
		if(Data->temp == 0xFF){
			Data->BitCounter = 1;
		}
		Data->temp = 0x00;
	}
}

/* stuffs the temp with zero bits and writes it to the buffer */
void StuffTempZero(struct Buffer *Data){
	if(Data->BitCounter != 0){
		Data->BitCounter = 0;
		BufferByte(Data, Data->temp);
		Data->temp = 0x00;
	}
}
/* same with one */
void StuffTempOne(struct Buffer *Data){

	while(Data->BitCounter > 0){
		BufferOne(Data);
	}
	Data->temp = 0x00;
}
/* writes the buffer to given file */
void WriteBuffer(struct Buffer *Data, FILE *fp){
	int i;
	for(i = 0; i < Data->ByteCounter;i++){
		write_byte(fp, Data->Data[i]);

	}
	
}


//appends an array to the buffer; enlarges the buffer if necessary
void BufferAppendArray(struct Buffer *Data, unsigned char *arr, int arr_len) {
	assert(Data->BitCounter == 0);
	//ensures sufficent size for the array
	if(Data->ByteCounter + arr_len >= Data->size)
		EnlargeBuffer(Data, Data->size * 2 + arr_len);
	memcpy(Data->Data + Data->ByteCounter, arr, arr_len);
	Data->ByteCounter += arr_len;
}

