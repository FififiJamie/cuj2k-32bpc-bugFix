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

#include "file_access.h"
#include "Boxes.h"
#include <stdio.h>
#include "BufferedWriter.h"
#include "bitmap.h"
/* Provides functionalities for file access*/

/* writes a byte to given file*/
int write_byte(FILE *fp, unsigned char value){
	
	if(fputc(value, fp) == EOF){
		printf("Error while writing file!");
		return(-1);
	}
	else
		return 0;
}

/*writes an int(4byte) to given file */
int write_int(FILE *fp, unsigned int value){
	write_byte(fp, value >> 24);
	write_byte(fp, value >> 16);
	write_byte(fp, value >> 8);
	write_byte(fp, value);
	return 0;
}

/*writes a short to given file(2byte) */

int write_short(FILE *fp, unsigned short value){
	write_byte(fp, value >> 8);
	write_byte(fp, value);
    return 0;
}




/*writes JP2 Header Box, which is a superbox that contains several other boxes.
So some side information is needed to calculate the length of this box.
@param: variousBpc: if the number of Bpc varies another box is needed
		nc: if so the number of components must be known
*/


int write_JP2header(FILE *fp, int variousBpc, int nc){

	/*calculate length */
	//brauchen wir glaub ich nicht
	if(variousBpc == 1)
		write_int(fp, 8 + IHB_LENGTH + CSB_LENGTH + BPC_LENGTH + nc);
	else
		write_int(fp, 8+ IHB_LENGTH + CSB_LENGTH);
	
	
	write_int(fp, JP2_HEADER_BOX);
	return 0;
}
/* write image header box.*/

int write_imageheader(FILE *fp, struct Picture *pic){
	// L field
	write_int(fp, IHB_LENGTH);
	// T field
	write_int(fp, IMAGE_HEADER_BOX);
	// Height and Width
	write_int(fp, pic->ySize);
	write_int(fp, pic->xSize);
	// number of components
	write_short(fp, 0x0003);
	// bit depth (only 24 bit yet)
	write_byte(fp, 0x07);
	// compression type, JPEG2000 allows only the value 7
	write_byte(fp, 0x07);
	// UC = unknown colorspace
	write_byte(fp, 0x00);
	// IP field, no intellectual property present
	write_byte(fp, 0x00);
	return 0;
}

/*writes the colorbox */

int write_colorbox(FILE *fp, int nc){
	write_int(fp, CSB_LENGTH);
	write_int(fp, COLOR_BOX);
	// color space is singalled by ECS parameter(enumerated)
	write_byte(fp, 0x01);
	// should be set to zero
	write_byte(fp, 0x00);
	write_byte(fp, 0x00);
	// sRGB color space
	write_int(fp, 16);
	
	return 0;
}



/* writes codestreambox. the buffer has to contain a jpeg2000 codestream, which is the output of tier 2 */
int write_codestream_box(FILE *fp, struct Buffer *codestream, int format){
	if(format == FORMAT_JP2) {
		write_int(fp, 0x00000000);
		write_int(fp, CODE_STREAM_BOX);
	}
	fwrite(codestream->Data, 1, codestream->ByteCounter, fp);
	return 0;
}

int write_fileformat (FILE *fp, struct Picture *pic){
	
	/* writing the fixed length 12 bytes JPEG2000 signature-box */
	write_int(fp, 0x0000000C);
	write_int(fp, JP2_SIGNATURE_BOX);
	// brand
	write_int(fp, 0x0d0a870a);
	// writing file type box. 
	write_int(fp, 0x00000014);
	write_int(fp, JP2_FILETYPE_BOX);

	write_int(fp, 0x6a703220);
	write_int(fp, 0);
	write_int(fp, 0x6a703220);
	// header box
	write_JP2header(fp, 0, 0);
	// image header box
	write_imageheader(fp, pic);
	// color box
	write_colorbox(fp, 0);

	return 0;
}


int write_output_file(FILE *fp, struct Picture *pic, struct Buffer *codestream, 
					  int format)
{
	if(format==FORMAT_JP2)
		write_fileformat(fp, pic);
	write_codestream_box(fp, codestream, format);

	return 0;
}
