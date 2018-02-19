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

/* Tag_Trees.h
 * 
 * Contains the structure for the tag trees.
 * T encode the whole tree into the Data-strucutre only codeTree is needed.
 *
 */

#ifndef TAG_TREES
#define TAG_TREES
#include "BufferedWriter.h"




struct TagTree{
	/* max lvl of tree */
	int maxlvl;
	/* x and y Dimension of each level */
	int* xLvl;
	int* yLvl;
	/* the actual Tag Tree. tree[i][j] gets the element at level i at position j, while
	   position = row * xDim / (lvl + 1) + column. Level-Dimension = (xDim|yDim)/ lvl + 1.
	   the highest level contains the root.
    */
	int **tree;
	/* same as tree pointer, but states only if the node at the position is already visited or not */
	int **visited;
};
/* makes a tagtree from the one dimensional input array which is interpretet as 2-dimensional with the
   help of the other parameters */
struct TagTree* makeTree(int *input, int x, int y);
/* codes the tag tree into the buffer */
void codeTree(struct TagTree *data, struct Buffer *buffer);
void freeTree(struct TagTree *date);
void putTree(struct TagTree *data);
int codeNode(struct TagTree *data, int x, int y, struct Buffer *buffer);

#endif
