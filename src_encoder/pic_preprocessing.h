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

/* Picture Preprocessing, has to be done before DWT.
 * Order:
 * Tiling 
 * Dc-Shift/MCT (RCT:int,ICT:float)
 */

#ifndef PIC_Preprocessing
#define PIC_Preprocessing
#include "bitmap.h"


/*Tiling, divides the picture in several tiles
  PARAMETERS:
  *pic - the picture container
  *img - the original bitmap
  cb_dim - dimension of the codeblocks, usually 64

  RETURNS:
  struct Picture with all tiles and necessary information
  tile0 is in left top corner, last tile in right bottom corner
  0 1 2 3
  4 5 6 7
  .......
  Can be accessed by pic->tile[number]
*/
extern "C"
void tiling (struct Picture *pic, struct Bitmap *img, int cb_dim);


/*Dc-Shift and Multi Color Transform in one.
  Has to be executed AFTER Tiling!
  
  DC-Shift: centers all the values around zero with
  shiftsize 128

  Multicomponent Color Transformation: transforms RGB into another colorspace
  There are 2 possible modes:
  LOSSLESS for Reversible color transformation (Used with 5/3 reversible wavelet transform=lossless)
  Results in integer numbers
  LOSSY for Irreversible color transformation (Used with 9/7 irreversible wavelet transform=lossy)
  results in float numbers */
extern "C"
void dcshift_mct (struct Picture *pic, int mode, cudaStream_t stream);

#endif
