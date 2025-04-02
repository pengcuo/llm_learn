#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>



__global__ void max_in_strided_groups(float *data, float *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = threadIdx.x % 32;  // lane_id in warp (0-31)
    float val = data[tid];

    uint32_t int_val = __float_as_uint(val);
    unsigned group_mask = 0x11111111 << (lane_id % 4);
    auto max_val_uint = __reduce_max_sync(group_mask, int_val);
    auto max_v = __uint_as_float(max_val_uint);
    output[tid] = max_v;
    //output[tid] = 1;
}

int main() {
    const int N = 32;  // 1 warp
    float *d_data, *d_output;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));  // 每个线程存储自己的最大值

    // 初始化数据（示例：0-31）
    float h_data[N];

    for(int i = 0; i < 8; ++i) {
        for(int j = 0; j < 4; ++j) {
            int idx = 4 * i + j;
            h_data[idx] = rand() % 20;
            printf("%.1f\t", h_data[idx]);
        }
        printf("\n");
    }
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel（1 block, 32 threads）
    max_in_strided_groups<<<1, 32>>>(d_data, d_output);

    // 取回结果
    float h_output[N];
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Max values for each thread:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("Thread %2d (group %d): max = %.1f\n", i, i % 4, h_output[i]);
    // }

    for(int i = 0; i < 8; ++i) {
        for(int j = 0; j < 4; ++j) {
            int idx = 4 * i + j;
            printf("%.1f ", h_output[idx]);
        }
        printf("\n");
    }



    cudaFree(d_data);
    cudaFree(d_output);
    return 0;
}
