#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// CUDA 核函数：计算前 4 个 warp 的最大值，并广播到所有线程
__global__ void max_4warps(float* data) {
    int tid = threadIdx.x;
    float val = data[tid];
    float global_max = val; // 初始化（后续会被覆盖）


    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cta);

    if (tid < 128) { // 仅前 4 个 warp 参与计算
        // cg::coalesced_group group = cg::coalesced_threads();
        // cg::thread_block_tile<32> warp = cg::tiled_partition<32>(group);

        // 1. Warp 内归约
        // float warp_max = cg::reduce(warp, val, cg::greater<float>());


        cg::thread_block_tile<128> fourWarpsGroup = cg::tiled_partition<128>(cta);

        // // 使用cg::reduce求最大值
        
        global_max = cg::reduce(fourWarpsGroup, val, cg::greater<float>());



        // // 2. 跨 Warp 归约（由每个 warp 的第一个线程完成）
        // if (warp.thread_rank() == 0) {
        //     global_max = cg::reduce(group, warp_max, cg::greater<float>());
        // }

        // // 3. 广播到当前 warp 的所有线程
        // global_max = warp.shfl(global_max, 0);
    }

    // 4. 将结果写回（仅前 4 个 warp 有效，后 4 个 warp 保持原值）
    data[tid] = global_max;
}

int main() {
    const int N = 256;
    float h_data[N], h_result[N];
    float *d_data;

    // 初始化数据（随机值）
    for (int i = 0; i < N; i++) {
        h_data[i] = static_cast<float>(rand() % 1000) / 10.0f; // 0~99.9
    }

    // 打印前 10 个值和真实最大值
    float true_max = 0;
    for (int i = 0; i < 128; i++) {
        if (h_data[i] > true_max) true_max = h_data[i];
        if (i < 10) std::cout << "h_data[" << i << "] = " << h_data[i] << std::endl;
    }
    std::cout << "CPU True Max = " << true_max << std::endl;

    // 分配设备内存并拷贝数据
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数（1 个 block，256 线程）
    max_4warps<<<1, N>>>(d_data);

    // 回传结果
    cudaMemcpy(h_result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    bool correct = true;
    for (int i = 0; i < 128; i++) {
        if (h_result[i] != true_max) {
            if (i < 128) { // 前 4 个 warp 应被覆盖为最大值
                std::cerr << "Error at h_result[" << i << "] = " << h_result[i] 
                          << " (expected " << true_max << ")" << std::endl;
                correct = false;
                break;
            }
        }
    }

    if (correct) {
        std::cout << "Success! All threads in warp0~3 got the max value: " << true_max << std::endl;
    } else {
        std::cerr << "Validation failed!" << std::endl;
    }

    // 释放资源
    cudaFree(d_data);
    return 0;
}
