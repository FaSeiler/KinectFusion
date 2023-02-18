#include "SurfacePoseEstimationCuda.h"

namespace Cuda
{
    static float* gpu_query;
    static float* gpu_data;
    static int* gpu_idxs;
    static float* gpu_dist_sq;

    void CheckCUDAError(const char* msg) {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    __global__ void Search(const float* query, int query_pts, const float* data, int data_pts,
        int* idxs, float* dist_sq)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= query_pts)
            return;

        int best_idx = -1;
        float best_dist = FLT_MAX;

        for (int i = 0; i < data_pts; i++) {
            float dist_sq = 0;

            for (int j = 0; j < CUDA_NN_DIM; j++) {
                float d = query[idx * CUDA_NN_DIM + j] - data[i * CUDA_NN_DIM + j];
                dist_sq += d * d;
            }

            if (dist_sq < best_dist) {
                best_dist = dist_sq;
                best_idx = i;
            }
        }

        idxs[idx] = best_idx;
        dist_sq[idx] = best_dist;
    }

    void CUDA_NN_Search(const float* query, int query_pts, const float* data, int data_pts,
        int* idxs, float* dist_sq)
    {
        int threads = 256;
        int blocks = query_pts / threads + ((query_pts % threads) ? 1 : 0);

        cudaMalloc((void**)&gpu_data, sizeof(float) * data_pts * CUDA_NN_DIM);
        cudaMalloc((void**)&gpu_query, sizeof(float) * query_pts * CUDA_NN_DIM);
        cudaMalloc((void**)&gpu_idxs, sizeof(int) * query_pts);
        cudaMalloc((void**)&gpu_dist_sq, sizeof(float) * query_pts);

        CheckCUDAError("initilisation");

        cudaMemcpy(gpu_query, query, sizeof(float) * query_pts * CUDA_NN_DIM, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_data, data, sizeof(float) * data_pts * CUDA_NN_DIM, cudaMemcpyHostToDevice);

        CheckCUDAError("memory copying");

        Search << <blocks, threads >> > (gpu_query, query_pts, gpu_data, data_pts, gpu_idxs, gpu_dist_sq);
        cudaThreadSynchronize();

        cudaMemcpy(idxs, gpu_idxs, sizeof(int) * query_pts, cudaMemcpyDeviceToHost);
        cudaMemcpy(dist_sq, gpu_dist_sq, sizeof(float) * query_pts, cudaMemcpyDeviceToHost);

        cudaFree(gpu_query);
        cudaFree(gpu_data);
        cudaFree(gpu_idxs);
        cudaFree(gpu_dist_sq);
    }
}