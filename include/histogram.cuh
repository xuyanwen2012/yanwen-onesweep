#pragma once

__global__ void k_GlobalHistogram(unsigned int* sort,
                                  unsigned int* globalHistogram,
                                  int size);
