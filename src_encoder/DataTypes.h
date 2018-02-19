#ifndef DATATYPES_H
#define DATATYPES_H
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
/*#define NULL 0*/

	typedef struct Parameter{
		const char *parameter;
		struct Parameter *next;
	} Parameter;

	typedef struct Response
	{
        	int errorCode;
        	unsigned char *output;
        	uint64_t outputSize;

	} Response;

	typedef struct SentMetadata
	{
        	uint64_t index;
        	uint64_t blockSize;
        	uint64_t originalSize;
        	int request;
        	int verbose;
        	char *codecTag;
        	struct Parameter *parameters;
        	struct Parameter *scriptTags;
	} SentMetadata;

#endif /*DATATYPES_H*/
