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

#ifndef BOXES_H
#define BOXES_H
/* Main boxes */
#define JP2_SIGNATURE_BOX 0x6A502020
#define JP2_FILETYPE_BOX 0x66747970
#define JP2_HEADER_BOX 0x6A703268
#define CODE_STREAM_BOX 0x6A703263
#define INTELLECTUAL_PROPERTY_BOX 0x64703269
#define XML_BOX 0x786d6c20
#define UUID_BOX 0x75756964
#define UUID_INFO_BOX 0x75696e66

/* JP2 Header boxes */
#define IMAGE_HEADER_BOX 0x69686472
#define COLOR_BOX 0x636F6C72   
#define BITS_PER_COMPONENT_BOX 0x62706363
#define PALETTE_BOX 0x70636c72
#define COMPONENT_MAPPING_BOX 0x636d6170
#define CHANNEL_DEFINITION_BOX 0x63646566
#define RESOLUTION_BOX 0x72657320
#define CAPTURE_RESOLUTION_BOX 0x72657363  
#define DEFAULT_DISPLAY_RESOLUTION_BOX 0x72657364

/* UUID Info Boxes */

#define UUID_LIST_BOX 0x75637374
#define URL_BOX 0x75726c20

/* Image Header Box Fields */
#define IMB_VERS 0x0100
#define IMB_C 7
#define IMB_UnkC 1
#define IMB_IPR 0
/* Colour Specification Box Fields */
#define CSB_METH 1
#define CSB_PREC 0
#define CSB_APPROX 0
#define CSB_ENUM_SRGB 16 

#define CSB_ENUM_GREY 17

/*Brand */
#define FT_BR 0x6a703220

/*Some length information about fixed length boxes*/
/*Color specification */
#define CSB_LENGTH 15

/* Length of File Type Box */
#define FTB_LENGTH 20
/* Length of Image Header Box */
#define IHB_LENGTH 22

/* base length of Bits Per Component box */
#define BPC_LENGTH 8

#endif
