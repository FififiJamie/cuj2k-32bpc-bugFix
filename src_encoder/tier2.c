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

/* Tier2.c
 *
 * Contains all the functions needed to encode all nessecary header information to a buffer.
 *
 */

#define MAX_LEVEL 4
#include "tier2.h"
#include "Tag_Trees.h"
#include <stdlib.h>
#include <assert.h>
#include "waveletTransform.h"
#include "Markers.h"
#include "BufferedWriter.h"
#include <math.h>
#include "tier1.h"
#include "rate-control.h"


/* frees struct packet */
void freePacket(struct Packet *packet){
	free(packet->cb);
	packet->cb = NULL;
	free(packet->CBx);
	packet->CBx = NULL;
	free(packet->CBy);
	packet->CBy = NULL;
	free(packet->zeroBitPlane);
	packet->zeroBitPlane = NULL;
	if(packet->reslevel == 0){
		free(packet->inclusion[0]);
		packet->inclusion[0] = NULL;
	}
	else{
		int i;
		for(i = 0;i<3;i++){
			free(packet->inclusion[i]);
			packet->inclusion[i] = NULL;
		}
	}
	free(packet->nCodeBlocks);
	packet->nCodeBlocks = NULL;
}


int power(int base, int exp){
	int temp,i;
	temp = base;
	if(exp == 0){
		return 1;
	}
	if(exp == 1){
		return base;
	}
	for(i = 1;i < exp;i++){
		temp *= base;
	}
	return temp;
}
/* type = number in array of subband, look at waveletTransform.h */
subband *getSubband(struct Picture *pic, int res, int comp, int tile, int type){
	subband *temp = ((subband **)(pic->tiles[tile].imgData))[comp];
	int i;
	for(i = 0;i < MAX_LEVEL - res;i++){
		temp = temp->subbands[0];
	}
	if(res == 0)
		return temp;
	else
		return temp->subbands[type];
}

/*returns an int *, size large, filled with zero */
int *zeroArray(int size){
	int *output = (int*) malloc(sizeof(int)*size);
	int i;
	for(i = 0;i<size;i++){
		output[i] = 0;
	}
	return output;
}

/* updates the inclusion array to contribute some blocks to layer 2.*/
int *getInclusion(struct Codeblock *cb, int size){
	int *temp = zeroArray(size);
	int i;
	for(i = 0;i<size;i++){
		if(cb[i].nCodingPasses == 0){
			temp[i]++;
		}
	}
	return temp;
}



/* Gets the indexed Packet out of the Picture structure and returns it */
struct Packet *getPacket(struct Picture *pic, int res, int comp, int tile){
	struct Packet *temp = (struct Packet*) malloc(sizeof(struct Packet));
	subband *sband;

	// if resolution level = 0 then only one subband is stored
	if(res == 0){
		sband = getSubband(pic, res, comp, tile, 0);
		temp->inclusion = (int**) malloc(sizeof(int *));
		temp->zeroBitPlane = (int**) malloc(sizeof(int *));
		temp->nCodeBlocks = (int*) malloc(sizeof(int));
		temp->CBx = (int*) malloc(sizeof(int *));
		temp->CBy = (int*) malloc(sizeof(int *));
		temp->CBx[0] = sband->nCBx;
		temp->CBy[0] = sband->nCBy;
		temp->reslevel = res;
		temp->zeroBitPlane[0] = sband->K_msbs;
		temp->nCodeBlocks[0] = sband->nCodeblocks;



		temp->cb = (struct Codeblock**) malloc(sizeof(struct Codeblock *));
		temp->cb[0] = sband->codeblocks;
		// since single layer compression the inclusion array is all zero
		temp->inclusion[0] = getInclusion(temp->cb[0], temp->nCodeBlocks[0]);


	}
	else{
		int i;
		// everything is needed 3 times. one time for each subband
		temp->inclusion = (int**) malloc(sizeof(int *)*3);
		temp->zeroBitPlane = (int**) malloc(sizeof(int *)*3);  //3
		temp->nCodeBlocks = (int*) malloc(sizeof(int)*3);
		temp->cb = (struct Codeblock**) malloc(sizeof(struct Codeblock *)*3);
		temp->CBx = (int*) malloc(sizeof(int *)*3);
		temp->CBy = (int*) malloc(sizeof(int *)*3);
		temp->reslevel = res;
		for(i = 0; i < 3; i++){
			sband = getSubband(pic, res, comp, tile, i+1);
			temp->zeroBitPlane[i] = sband->K_msbs;
			temp->nCodeBlocks[i] = sband->nCodeblocks;
			temp->cb[i] = sband->codeblocks;
			temp->CBx[i] = sband->nCBx;
			temp->CBy[i] = sband->nCBy;
			// since single layer compression the inclusion array is all zero
			temp->inclusion[i] = getInclusion(temp->cb[i], temp->nCodeBlocks[i]);
		}
	}
	return temp;
}

/* encodes a whole tile including the header */
int encodeTile(struct Buffer *buffer, struct Picture *pic, struct Tier1_Pic *t1pic, int tile, int dwt){
	/* Contains:
	 * Tile Header
	 * Packet Stream */
	 //printf("DEBUG: encodeTile\n");
	int res, ch, length;
	int lengthField = encodeTileHeader(buffer , pic, tile);
	int packetSeq = 0;
	struct Packet *packet;
	// Start of Data marker
	BufferShort(buffer, SOD);

    /* 2 of the 5 nested loops which define the progression order.
	   the 3rd loop is over the tiles in the main program, since single
	   and no precincts the other 2 are not needed.*/
	for(res = 0;res < dwt + 1;res++){
		for(ch = 0;ch < 3;ch++){
			packet = getPacket(pic, res, ch, tile);
			encodePacketHeader(buffer, packet, res, packetSeq);
			contributeCodeBlocks(buffer, packet, res, t1pic);
			freePacket(packet);
			free(packet);
			packet = NULL;
			packetSeq++;
		}
	}

	// update length field, length + 6 because the whole tile counts excluding nothing
    length = buffer->ByteCounter - lengthField + 6;
	buffer->Data[lengthField] = (unsigned char)(length>>24);
	buffer->Data[lengthField + 1] = (unsigned char)(length>>16);
	buffer->Data[lengthField + 2] = (unsigned char)(length>>8);
	buffer->Data[lengthField + 3] = (unsigned char)(length);

	return 0;
}






/* Writes the indexed Tile Header to the given buffer returns data->ByteCounter from the location where the Tile length field is located*/
int encodeTileHeader(struct Buffer *buffer, struct Picture *pic, int tile){
	/* Contains:
	 * Sot Marker
	 * Length Field Marker
	 * Tile Number
	 * Length Field Tile
	 * Tile Part Number
	 * Total amount of Tile Parts */
	int lengthField;
	BufferShort(buffer, SOT);
	BufferShort(buffer, 10);
	BufferShort(buffer, tile);
	lengthField = buffer->ByteCounter;

	// is updated by another function when the tile stream is written(writes tilestream length)
	BufferInt(buffer, 0);
	// tile part number and total number of tile parts. since there is only one tile part set to 0 and 1
	BufferByte(buffer, 0x00);
	BufferByte(buffer, 0x01);

	return lengthField;


}

/* encodes variable length codes for the number of coding passes. view taubmann table 8.4 */
int codenCodingpasses(int n, struct Buffer *buffer){
	assert(n > 0 && n < 165);
	if(n == 1){
		BufferZero(buffer);
		return 0;
	}
	if(n == 2){
		BufferOne(buffer);
		BufferZero(buffer);
		return 0;
	}
	if(n > 2 && n < 6){
		int temp, i;
		BufferOne(buffer);
		BufferOne(buffer);
		temp = n - 3;
		for(i = 1;i>-1;i--){
			if(temp >= power(2,i)){
				BufferOne(buffer);
				temp -= power(2, i);
			}
			else
				BufferZero(buffer);
		}
		return 0;
	}
	if(n > 5 && n < 37){
		int i, temp;
		for(i = 0;i<4;i++){
			BufferOne(buffer);
		}
		temp = n - 6;
		for(i = 4;i>-1;i--){
			if(temp >= power(2, i)){
				BufferOne(buffer);
				temp -= power(2, i);
			}
			else
				BufferZero(buffer);
		}
		return 0;
	}
	if(n > 36){
		int i, temp;
		//if(n>108)
		//printf("[WARNING] n = %d\n", n);
		for(i = 0;i< 9;i++){ //8
			BufferOne(buffer);
		}
		temp = n - 37;
		for(i = 6;i>-1;i--){
			if(temp >= power(2, i)){
				BufferOne(buffer);
				temp -= power(2, i);
			}
			else
				BufferZero(buffer);
		}
		return 0;
	}
	return 0;
}

int calcBinaryLength(int bytes){
	int i = 0;
	while(bytes > 0){
		bytes /= 2;
		i++;
	}
	return i;
}
/* encodes the number of submitted bytes for each codeblock. its signalled through a
 * state variable which is initialized with 3 and its value is changed if it doesn't
 * fit the formular shown below. */
int encodeCodeBytes(int nCodingpasses, int bytes, struct Buffer *buffer){
	int i, length, temp;
	length = calcBinaryLength(bytes);
	temp = length - (int)floor(log((double)nCodingpasses)/log((double)2));
	temp -= 3;
	//printf("codeBytes\n");
	if(temp > 0){
		//printf("[Warning] ADD LBlock\n");
		for(i = 0;i < temp;i++){
			BufferOne(buffer);
		}
	}
	BufferZero(buffer);
	if(temp < 0){
		while(temp < 0){
			BufferZero(buffer);
			temp++;
		}
	}
	for(i = length - 1; i>=0;i--){
		if(bytes >= power(2,i)){
			bytes -= power(2,i);
			BufferOne(buffer);
		}
		else{
			BufferZero(buffer);
		}
	}
	return 0;
}


/* encodes the main header. reads all required parameters out of *pic */
int encodeMainHeader(struct Buffer *buffer, struct Picture *pic,int dwt, int mode, int quant_enable, int bps){
	/* Contains:
	 * SOC Marker
	 * SIZ Marker
	 * QCD Marker
	 * COD Marker */
	int lengthField, length, i;
	unsigned short int *standard_qcd;
    unsigned char *standard_qcd_lossless;



	// Start of Codestream Marker
	BufferShort(buffer, SOC);
	/* SIZ Marker
	 * Marker Consists of:
	 * SIZ|Length Field|Ca Field|Pic Dimensions|Anchor Point|Tile Size|Tile Anchor Point|
	 * Bits Per Component */
	BufferShort(buffer, SIZ);
	lengthField = buffer->ByteCounter;
	// length field is updated later
	BufferShort(buffer, 0x0000);

	// in JPEG2000 pt. I the CA field is 0x0000
	BufferShort(buffer, 0x0000);

	// since we have no canvas, parameters in the SIZ field maps the image at (0,0) on the canvas
	BufferInt(buffer, pic->xSize);
	BufferInt(buffer, pic->ySize);
	BufferInt(buffer, 0x00000000);
	BufferInt(buffer, 0x00000000);
	// variable Tilesize( set in tiling() )
	BufferInt(buffer, pic->tilesize);
	BufferInt(buffer, pic->tilesize);
	// Tile Anchor point (0,0)
	BufferInt(buffer, 0x00000000);
	BufferInt(buffer, 0x00000000);
	// only rgb mode yet
	BufferShort(buffer, 0x0003);
	// sample range -126 .. 127 <=> 24bit?
	BufferByte(buffer, (unsigned char)(bps-1));
	// subsampleling factor for each component(forced to 1)
	BufferByte(buffer, 0x01);
	BufferByte(buffer, 0x01);
	// same for each component
	BufferByte(buffer, (unsigned char)(bps-1));
	// subsampleling factor for each component(forced to 1)
	BufferByte(buffer, 0x01);
	BufferByte(buffer, 0x01);

	BufferByte(buffer, (unsigned char)(bps-1));
	// subsampleling factor for each component(forced to 1)
	BufferByte(buffer, 0x01);
	BufferByte(buffer, 0x01);
	// update length field
	length = buffer->ByteCounter - lengthField;
	buffer->Data[lengthField] = (unsigned char)(length >> 8);
	buffer->Data[lengthField + 1] = (unsigned char)(length);

	/* QCD Marker
	 * Marker Consists of:
	 * QCD|Length Field|Quantisation Parameters */
	BufferShort(buffer, QCD);
	//printf("write QCD\n");
	lengthField = buffer->ByteCounter;
	// length field is updated later
	BufferShort(buffer, 0x0000);
	BufferByte(buffer, pic->tiles[0].QS);

    if (mode == LOSSLESS){//test
			//printf("god!!!!!!!!!\n");
	    standard_qcd_lossless = get_standard_qcd_lossless(4, bps);
        for(i = 0;i < 13;i++){ //13 before
	        BufferByte(buffer, standard_qcd_lossless[i]);
	    }
        free(standard_qcd_lossless);
        standard_qcd_lossless = NULL;
    }
    else{
        standard_qcd = get_standard_qcd(4, quant_enable);
        for(i = 0;i < 13;i++){
	        BufferShort(buffer, standard_qcd[i]);
	    }
        free(standard_qcd);
        standard_qcd = NULL;
    }

	length = (buffer->ByteCounter - lengthField);
	buffer->Data[lengthField] = (unsigned char)(length >>8);
	buffer->Data[lengthField + 1] = (unsigned char)length;

	/* COD Marker
	 * Marker Consists of:
	 * COD|Length Field|Coding Style Flags|Progression|Mct Flag|DWT Lvl|Cb Size|Modeswitch Flags|
	 * WT Field*/
	BufferShort(buffer, COD);
	lengthField = buffer->ByteCounter;
	BufferShort(buffer, 0x0000);

	// CS field
	/* CS(0) = precinct size defined?, CS(1) = SOP used?, CS(2) = EPH used?, CS(3 .. 4) = Anchor point,
	 * CS(5 .. 7) = 0 */
	BufferZero(buffer);
	BufferZero(buffer);
	BufferZero(buffer);
	BufferZero(buffer);
	BufferZero(buffer);
	BufferOne(buffer);  //EPH shall be used
	BufferOne(buffer); //BufferZero(buffer);
	BufferZero(buffer);
	// Op field = 0 since only LRCP progression is implemented yet
	BufferByte(buffer, 0x00);
	// single layer compression
	BufferShort(buffer, 0x0001);
	// MCT flag
	BufferByte(buffer, 0x01);
	// DWT level
	BufferByte(buffer, dwt);
	// Codeblock size
	BufferByte(buffer, pic->cb_xdim_exp-2);
	BufferByte(buffer, pic->cb_ydim_exp-2);
	// Modeswitch byte. All modeswitches off since now
	BufferByte(buffer, 0x00);
	if(mode == LOSSY){
	// WT field, irrreversible
		BufferByte(buffer, 0x00);
	}
	else{
		// WT field, reversible
		BufferByte(buffer, 0x01);
	}

	// update length field
	length = buffer->ByteCounter - lengthField;
	buffer->Data[lengthField] = (unsigned char)(length >> 8);
	buffer->Data[lengthField + 1] = (unsigned char)(length);


	//insert optional markers here ^^
	return 0;
}

/* Writes the header for the given packet to the buffer*/
int encodePacketHeader(struct Buffer *buffer, struct Packet *packet, int res, int packetSeq){
	/* Contains:
	 * Inclusion Tag Tree
	 * Zero Bit Plane Tag Tree
	 * Number of Coding Passes
	 * Length Information */
	 //printf("DEBUG: PAcket header!!!\n");

	 //SOP for debuging
	BufferShort(buffer, SOP);
	BufferShort(buffer, 0x4);
	BufferShort(buffer, packetSeq);


	int i, length;
	struct TagTree *inclusion, *zeroBitplane;
	length = buffer->ByteCounter;
	BufferOne(buffer);  // not empty

	if(packet->reslevel == 0){
		//printf("packet if\n");
		inclusion = makeTree(packet->inclusion[0], packet->CBx[0], packet->CBy[0]);
		zeroBitplane = makeTree(packet->zeroBitPlane[0], packet->CBx[0], packet->CBy[0]);
		for(i = 0;i<packet->nCodeBlocks[0];i++){
			codeNode(inclusion, i % packet->CBx[0], i / packet ->CBx[0], buffer);
			if(packet->cb[0][i].nCodingPasses != 0){
				codeNode(zeroBitplane, i % packet->CBx[0], i / packet->CBx[0], buffer);
				codenCodingpasses(packet->cb[0][i].nCodingPasses, buffer);
				encodeCodeBytes(packet->cb[0][i].nCodingPasses, packet->cb[0][i].L, buffer);
			}
		}

		freeTree(zeroBitplane);
		free(zeroBitplane);
		zeroBitplane = NULL;
		freeTree(inclusion);
		free(inclusion);
		inclusion = NULL;
	}
	else{
		//printf("packet else\n");
		int sband;
		for(sband = 0;sband<3;sband++){
			inclusion = makeTree(packet->inclusion[sband], packet->CBx[sband], packet->CBy[sband]);
			zeroBitplane = makeTree(packet->zeroBitPlane[sband], packet->CBx[sband], packet->CBy[sband]);
			for(i = 0;i<packet->nCodeBlocks[sband];i++){
				codeNode(inclusion, i % packet->CBx[sband], i / packet ->CBx[sband], buffer);
				if(packet->cb[sband][i].nCodingPasses != 0){
					//printf("packet else inner if call, i= %d\n", i);
					codeNode(zeroBitplane, i % packet->CBx[sband], i / packet ->CBx[sband], buffer);
					codenCodingpasses(packet->cb[sband][i].nCodingPasses, buffer);
					encodeCodeBytes(packet->cb[sband][i].nCodingPasses, packet->cb[sband][i].L, buffer);
				}

			}
			freeTree(zeroBitplane);
			free(zeroBitplane);
			zeroBitplane = NULL;
			freeTree(inclusion);
			free(inclusion);
			inclusion = NULL;
		}
	}



	StuffTempZero(buffer);

	BufferShort(buffer, EPH);
	return 0;


}

#define min(a,b) (((a)<(b))?(a):(b))

//appends the data of one codeblock to the buffer
void appendCB(struct Buffer *buffer, struct Codeblock *cb) {
	//discard 1st byte
	BufferAppendArray(buffer, (cb->B_h)+1, cb->L);
}


/* Contributes the Code Blocks in the given packet to the buffer */
int contributeCodeBlocks(struct Buffer *buffer, struct Packet *packet, int res, struct Tier1_Pic *t1pic){
	int i, sb;
	if(packet->reslevel == 0){
		for(i = 0;i < packet->nCodeBlocks[0];i++){
			if(packet->cb[0][i].nCodingPasses != 0){

				appendCB(buffer, &(packet->cb[0][i]));
				//add 1 to B's address because 1st byte must be discarded
				//BufferAppendArray(buffer, packet->cb[0][i].buffer_h + 1, packet->cb[0][i].L);
				/*for(j = 0;j < packet->cb[0][i].L;j++){
					//BufferByte(buffer, packet->cb[0][i].B[j+1]);
				}*/
			}
		}
	}
	else{
		for(sb = 0;sb < 3;sb++){
			for(i = 0;i < packet->nCodeBlocks[sb];i++){
				if(packet->cb[sb][i].nCodingPasses != 0){
					appendCB(buffer, &(packet->cb[sb][i]));
					//add 1 to B's address because 1st byte must be discarded
					//BufferAppendArray(buffer, packet->cb[sb][i].buffer_h + 1, packet->cb[sb][i].L);
					/*for(j = 0;j < packet->cb[sb][i].L;j++){
						BufferByte(buffer, packet->cb[sb][i].B[j+1]);
					}*/
				}
			}
		}
	}
	return 0;
}




/*encodes entire Code Stream out of Tier 1 compressed Picture struct */
int encodeCodeStream(struct Buffer *buffer, struct Picture *pic, struct Tier1_Pic *t1pic, int dwt, int mode, int quant_enable, int bps){
	int i;
	encodeMainHeader(buffer, pic, dwt, mode, quant_enable, bps);
	for(i = 0;i < pic->tile_number;i++){
		//printf("DEBUG: how many tile?\n");
		encodeTile(buffer, pic, t1pic, i, dwt);
	}
	BufferShort(buffer, EOC);
	return 0;
}
