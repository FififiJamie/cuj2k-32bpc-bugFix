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

#ifndef FILE_ACCESS_H
#define FILE_ACCESS_H
#include <stdio.h>
#include "bitmap.h"
#include "BufferedWriter.h"

// output formats
#define FORMAT_JP2 0  //full jpeg2000 file
#define FORMAT_J2K 1  //codestream only (e.g. for MotionJpeg2000)

/* functions for writing values of different size to a file */
int write_int(FILE *fp, unsigned int value);
int write_short(FILE *fp,unsigned short value);
int write_byte(FILE *fp, unsigned char value);
extern "C" int write_fileformat (FILE *fp, struct Picture *pic);
int write_JP2header(FILE *fp, int variousBpc, int nc);
int write_imageheader(FILE *fp, struct Picture *pic);
int write_colorbox(FILE *fp, int nc);
extern "C" int write_codestream_box(FILE *fp, struct Buffer *codestream, int format);
extern "C" int write_output_file(FILE *fp, struct Picture *pic, struct Buffer *codestream, 
								 int format);

#endif
