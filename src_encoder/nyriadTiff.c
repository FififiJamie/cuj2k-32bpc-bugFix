#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "nyriadTiff.h"

int readTiffData(simpleTIFF *img, int dataOffSets, int channels, unsigned char *src){
  //only deal with single strip now
  char *data;
  data = (char*)malloc(img->area_alloc);
  //printf("img->area_alloc %d\n", img->area_alloc);
  if(channels==1){
    memcpy((void*)data, src+dataOffSets, img->area_alloc);
    img->imgData = data;
  }
  else{
    printf("Sorry, only deal with greyscale now\n");
    return 1;
  }
  //printf("ads\n");
  //free (data);

  return 0;

}
