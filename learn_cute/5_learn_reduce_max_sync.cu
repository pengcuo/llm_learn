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

/*
__device__ float warp_stride_max(float val) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int lane_id = warp.thread_rank();  // 当前线程在 warp 中的 ID (0-31)
    int group_id = lane_id % 4;        // 组 ID (0-3)

    // 使用 labeled_partition 创建跨步分组（每组 8 线程）
    auto group = cg::labeled_partition(warp, group_id);

    // 组内求最大值（支持任意数据类型，如 float/int/double）
    float max_val = cg::reduce(group, val, cg::maximum<float>());

    return max_val;
}
*/

__device__ float warp_stride_max(float val) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto group = cg::labeled_partition(warp, warp.thread_rank() % 4);
    return cg::reduce(group, val, [](float a, float b) { return max(a, b); });
}


__global__ void max_in_strided_groups(float *data, float *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = data[tid];

    /*
    int lane_id = threadIdx.x % 32;  // lane_id in warp (0-31)
    int int_val = __float_as_int(val);
    unsigned group_mask = 0x11111111 << (lane_id % 4);
    auto max_val_int = __reduce_max_sync(group_mask, int_val);
    auto max_v = __int_as_float(max_val_int);
    */
    float max_v = warp_stride_max(val);
    output[tid] = max_v;
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

    for(int i = 0; i < 8; ++i) {
        for(int j = 0; j < 4; ++j) {
            int idx = 4 * i + j;
            //h_data[idx] = -rand() % 20;
            h_data[idx] = (float)distribution(generator);

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
