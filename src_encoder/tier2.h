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

/* tier2.h
 *
 * Contains the Data struct to order the data as a packet.
 */

#ifndef TIER2_H
#define TIER2_H

#include "bitmap.h"
#include "BufferedWriter.h"
#include "bitmap.h"

struct Packet{
	int **inclusion;
	int **zeroBitPlane;
	int *CBx;
	int *CBy;
	int reslevel;
	int *nCodeBlocks;
	struct Codeblock **cb;
};


/* Gets the indexed Packet out of the Picture structure and returns it */
extern "C" struct Packet *getPacket(struct Picture *pic, int res, int comp, int tile);
/* Writes the indexed Tile Header to the given buffer */
extern "C" int encodeTileHeader(struct Buffer *buffer, struct Picture *pic, int tile);
/* Writes the header for the given packet to the buffer*/
extern "C" int encodePacketHeader(struct Buffer *buffer, struct Packet *packet, int res, int packetSeq);
/* Contributes the Code Blocks in the given packet to the buffer */
extern "C" int contributeCodeBlocks(struct Buffer *buffer, struct Packet *packet, int res, struct Tier1_Pic *t1pic);

/*encodes entire Code Stream out of Tier 1 compressed Picture struct */
extern "C" int encodeCodeStream(struct Buffer *buffer, struct Picture *pic, struct Tier1_Pic *t1pic, int dwt, int mode, int quant_enable, int bps);







#endif
