#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <random>
#include <chrono>


namespace cg = cooperative_groups;

__device__ float warp_stride_sum(float val) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int lane_id = warp.thread_rank();  // 当前线程在 warp 中的 ID (0-31)
    int group_id = lane_id % 4;        // 组 ID (0-3)

    // 使用 labeled_partition 创建跨步分组
    auto group = cg::labeled_partition(warp, group_id);

    // 组内归约求和
    float sum = cg::reduce(group, val, cg::plus<float>());
    return sum;
}

__global__ void sum_in_strided_groups(float *data, float *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = data[tid];

    float sum_v = warp_stride_sum(val);
    output[tid] = sum_v;
}

int main() {
    const int N = 32;  // 1 warp
    float *d_data, *d_output;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));  // 每个线程存储自己的最大值

    // 初始化数据（示例：0-31）
    float h_data[N];

    //unsigned seed = 199;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(-10.0, 10.0);

    float sum[4];
    for(int i = 0; i < 8; ++i) {

        for(int j = 0; j < 4; ++j) {
            int idx = 4 * i + j;
            h_data[idx] = (float)distribution(generator);

            sum[j] += h_data[idx];

            printf("%.1f\t", h_data[idx]);
        }
        printf("\n");
    }

    for(int j = 0; j < 4; ++j) {
        printf("%1f\t", sum[j]);
    }
    printf("\n");


    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel（1 block, 32 threads）
    sum_in_strided_groups<<<1, 32>>>(d_data, d_output);

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
