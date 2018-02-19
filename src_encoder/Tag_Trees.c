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

/* Tag Trees.h
 *
 * Includes functions to realize the Tag Tree Coding.
 * The Tag Tree is calculated through invoking CalcNextLevel until the root node is reached.
 * 
 */
#include "Tag_Trees.h"
#include <stdlib.h>






/* to allow array near access */
int getCoords(int x, int y, int xDim, int size){
	
	if(x >= xDim || y*xDim + x >= size){
		return -1;
	}
	return y*xDim + x;
}


/* calculates the minimum along the 4 childs of a node in the array given */
int getMin(int x, int y, int* input, int xDim, int size){
	int current;
	int min = input[getCoords(x*2,y*2,xDim,size)];
	if(getCoords(x*2 + 1,y*2,xDim,size) > -1)
		 current = input[getCoords(x*2 + 1,y*2,xDim,size)];
	else
		current = min;
	if(current < min)
		min = current;
	if(getCoords(x*2,y*2 + 1,xDim,size) > -1)
		current = input[getCoords(x*2,y*2 + 1,xDim,size)];
	if(current < min)
		min = current;
	if(getCoords(x*2 + 1,y*2 + 1,xDim,size) > -1)
		current = input[getCoords(x*2 + 1,y*2 + 1,xDim,size)];
	if(current < min)
		min = current;
	return min;
}


/* internal use, calculates next level for the given array*/
int* CalcNextLevel(int *input, int x, int y){
	int resultxDim = (x /2) + (x % 2);
	int resultyDim = (y / 2) + (y % 2);
	int resultsize = resultxDim*resultyDim;
	int* output = (int*) malloc(sizeof(int)*resultsize);
	int i;
	int j;
	for(i = 0;i < resultxDim;i++){
		for(j = 0;j < resultyDim;j++){
			if(getCoords(i,j,resultxDim, resultsize) > -1)
			   output[getCoords(i,j,resultxDim, resultsize)] = getMin(i,j,input,x,x*y);
		}
	}
	return output;
}


/* calc tree maxlevel by incrementing a counter and doubeling the value from one until it is greater
 * than the greater dimension of the original array */
int calcMaxlevel(int x){
	int i, temp;
	if(x == 1)
		return 0;
	i = 0;
    temp = 1;
	while(x > temp){
		temp *= 2;
		i++;
	}
	return i;
}





/*calculates a tag tree from the given array and returns it as a pre defined struct */
struct TagTree* makeTree(int *input, int x, int y){
	struct TagTree *output = (struct TagTree*) malloc(sizeof(struct TagTree));
	int maxlvl, i, lvlx, lvly, lvl;
	if(x > y)
		maxlvl = calcMaxlevel(x);
	else
		maxlvl = calcMaxlevel(y);
	output->maxlvl = maxlvl;
	/* in tree[0] the original array is stored */
	output->tree = (int**) malloc(sizeof(int *)*(maxlvl + 1));
	output->tree[0] = input;
	output->xLvl = (int*) malloc(sizeof(int)*(maxlvl + 1));
	output->yLvl = (int*) malloc(sizeof(int)*(maxlvl + 1));
	output->visited = (int**) malloc(sizeof(int *)*(maxlvl + 1));
	output->xLvl[0] = x;
	output->yLvl[0] = y;
	output->visited[0] = (int*) malloc(sizeof(int)*x*y);

	for(i = 0;i < x*y;i++){
			output->visited[0][i] = 0;
	}

	/* stores the dimensions of the current level */
	lvlx = x;
	lvly = y;

	for(lvl = 1;lvl < maxlvl + 1;lvl++){
		output->tree[lvl] = CalcNextLevel(output->tree[lvl - 1], lvlx, lvly);
		output->visited[lvl] = (int*) malloc(sizeof(int)*lvlx*lvly);
		for(i = 0;i < lvlx*lvly;i++){
			output->visited[lvl][i] = 0;
		}
		lvlx = (lvlx / 2) + (lvlx % 2);
		lvly = (lvly / 2) + (lvly % 2);
		output->xLvl[lvl] = lvlx;
		output->yLvl[lvl] = lvly;
	}
	return output;
}

/* returns the parent value of indexed node out of given tree */

int getParentvalue(struct TagTree *data, int x, int y, int level){
	return data->tree[level + 1][getCoords(x / 2, y / 2, data->xLvl[level+1], data->xLvl[level+1]*data->yLvl[level+1])];
}


/* codes one node of the tag tree.*/
int codeNode(struct TagTree *data, int x, int y, struct Buffer *buffer){
	int i;
	int lvlx, lvly, lvlxDim, lvlsize;
	int level;
	int parentvalue;
	// root node has to be coded
	if(data->visited[data->maxlvl][0] == 0){
		data->visited[data->maxlvl][0] = 1;
		for(i = 0;i < data->tree[data->maxlvl][0];i++){
			BufferZero(buffer);
		}
		BufferOne(buffer);
	}
	// all levels except zero level
	for(level = data->maxlvl - 1; level > 0;level--){
		lvlx = x/(2*level);
		lvly = y/(2*level);
		lvlxDim = data->xLvl[level];
		lvlsize = data->xLvl[level]*data->yLvl[level];

		if(data->visited[level][getCoords(lvlx, lvly, lvlxDim, lvlsize)] == 0){
			parentvalue = getParentvalue(data, lvlx, lvly, level);
			data->visited[level][getCoords(lvlx, lvly, lvlxDim, lvlsize)] = 1;

			for(i = 0;i <data->tree[level][getCoords(lvlx, lvly, lvlxDim, lvlsize)] - parentvalue;i++){
					BufferZero(buffer);
			}
			BufferOne(buffer);
		}
	}
	// zero level, contains root if maxlevel = 0
	if(data->maxlvl > 0){
		data->visited[0][getCoords(x, y, data->xLvl[0], data->xLvl[0]*data->yLvl[0])] = 1;
		parentvalue = data->tree[1][getCoords(x / 2, y / 2, data->xLvl[1], data->xLvl[1]*data->yLvl[1])];
		for(i = 0;i< data->tree[0][getCoords(x, y, data->xLvl[0], data->xLvl[0]*data->yLvl[0])] - parentvalue;i++){
					BufferZero(buffer);
		}
		BufferOne(buffer);
	}
	return 0;
}


/* uses codeNode to code every node of the tag tree to given buffer */
void codeTree(struct TagTree *data, struct Buffer *buffer){
	int i;
	int j;
	for(j = 0;j<data->yLvl[0];j++){
		for(i = 0;i < data->xLvl[0];i++){
			codeNode(data, i, j, buffer);
		}
	}
}

void freeTree(struct TagTree *data){
	int i;
	for(i = 1;i<data->maxlvl;i++){
		if(data->tree[i] != NULL){
			free(data->tree[i]);
			data->tree[i] = NULL;
		}
		if(data->visited[i] != NULL){
			free(data->visited[i]);
			data->visited[i] = NULL;
		}
	}
	// free(data->visited[0]);
	free(data->tree);
	data->tree = NULL;
	free(data->visited);
	data->visited = NULL;
	free(data->xLvl);
	data->xLvl = NULL;
	free(data->yLvl);
	data->yLvl = NULL;
}
/*debugging*/
void putTree(struct TagTree *data){
	int lvl, i;
	for(lvl = data->maxlvl;lvl >= 0;lvl--){
		printf("level: %d\n", lvl);
		for(i = 0;i < data->xLvl[lvl]*data->yLvl[lvl];i++){
			printf("value[%d]: %d\n", i, data->tree[lvl][i]);
		}
	}
}
