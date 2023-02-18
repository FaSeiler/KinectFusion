#pragma once

#define CUDA_NN_DIM 3 // data dimension

#include <cstdio>
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <float.h>

namespace Cuda {
	void CUDA_NN_Search(const float* query, int query_pts, const float* data, int data_pts,
		int* idxs, float* dist_sq);
}

