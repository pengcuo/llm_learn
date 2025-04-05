#include <cstdio>
#include <cooperative_groups.h>
#include <cmath>
#include <cooperative_groups/reduce.h>

#include <cutlass/cutlass.h>


namespace cg = cooperative_groups;

__device__ float group_4x8_4_max(float val) {

        __shared__ float shared_max[4][4];

        int lane = threadIdx.x % 32;
        int col = lane % 4;
        int row = lane / 4;
        int warp_id = threadIdx.x >> 5; // threadIdx.x / 32
        cg::coalesced_group active = cg::coalesced_threads();
        auto column_group = cg::labeled_partition(active, col);
        
        float max_val = cg::reduce(column_group, val, cg::greater<float>());
        if(row == 0) {
            shared_max[warp_id][col] = max_val;
        }
        __syncthreads();

        float warp_max = shared_max[0][col];
        for(int i = 0; i < 4; ++i) {
            if(shared_max[i][col] > warp_max) {
                warp_max = shared_max[i][col];
            }
        }

        max_val = warp_max;       
        return max_val;
}

__global__ void warp_8x4_max_reduce(float* data) {
    int warp_group_idx = cutlass::canonical_warp_group_idx();

    if(warp_group_idx == 0) {
        float val = data[blockIdx.x * blockDim.x + threadIdx.x];
        float max_val = group_4x8_4_max(val);
        data[blockIdx.x * blockDim.x + threadIdx.x] = max_val;
    }
    else {
        if(threadIdx.x == 128) {
            printf("warp_group_idx %d threadIdx.x %d\n", warp_group_idx, threadIdx.x);
        }
    }

}

// 验证函数
bool verify_results(float* h_data, int num_elements, int warps_per_block) {
    const int threads_per_warp = 32;
    const int block_size = warps_per_block * threads_per_warp;
    
    // 计算每个列组的最大值 (共4个列组)
    float expected_max[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    
    // 收集所有warp的列0-3的第一个线程的值
    for (int w = 0; w < warps_per_block; ++w) {
        for (int col = 0; col < 4; ++col) {
            int lane_id = col; // 每列的第一个线程lane_id等于列号
            int tid = w * threads_per_warp + lane_id;
            expected_max[col] = fmaxf(expected_max[col], h_data[tid]);
        }
    }
    
    // 验证每个线程的结果
    for (int w = 0; w < warps_per_block; ++w) {
        for (int t = 0; t < threads_per_warp; ++t) {
            int tid = w * threads_per_warp + t;
            int col = t % 4;  // 列号
            if (fabs(h_data[tid] - expected_max[col]) > 1e-5f) {
                printf("Mismatch at warp %d, thread %d (lane %d): expected %.2f, got %.2f\n",
                      w, t, t % 32, expected_max[col], h_data[tid]);
                return false;
            }
            else {
                printf("Match at warp %d, thread %d (lane %d): expected %.2f, %.2f\n",
                      w, t, t % 32, expected_max[col], h_data[tid]);
            }
        }
    }
    
    return true;
}

int main() {
    const int warps_per_block = 8;  // 测试4个warp
    const int block_size = warps_per_block * 32;
    const int num_blocks = 1;
    const int num_elements = block_size * num_blocks;
    
    // 准备测试数据
    float* h_data = new float[num_elements];
    float* h_data_result = new float[num_elements];
    for (int i = 0; i < num_elements; i++) {
        // 创建有规律的数据便于验证
        int warp_id = i / 32;
        int lane_id = i % 32;
        int col = lane_id % 4;
        h_data[i] = 10.0f * warp_id + col; // 不同warp有不同基值，不同列有不同偏移
    }
    
    // 设置一些特殊值作为最大值
    h_data[0] = 100.0f;   // warp0, 列0
    h_data[5] = 250.0f;   // warp0, 列1 (lane5 = 1*8 + 1, 但实际是行优先布局)
    h_data[34] = 377.0f;  // warp1, 列2 (lane2)
    h_data[103] = 423.0f; // warp3, 列3 (lane7 = 3*32 + 7)


    for(int r = 0; r < 32; ++r) {
        for(int c = 0; c < 4; ++c) {
            printf("%.1f\t\t", h_data[r * 4 + c]);
        }
        printf("\n");
    }
    
    // 分配设备内存
    float* d_data;
    cudaMalloc(&d_data, num_elements * sizeof(float));
    cudaMemcpy(d_data, h_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动内核
    warp_8x4_max_reduce<<<num_blocks, block_size>>>(d_data);
    
    // 拷贝回结果
    cudaMemcpy(h_data_result, d_data, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    // bool success = verify_results(h_data, num_elements, warps_per_block);

    printf("Result : \n");

    for(int r = 0; r < 32; ++r) {
        for(int c = 0; c < 4; ++c) {
            printf("%.1f\t\t", h_data_result[r * 4 + c]);
        }
        printf("\n");
    }


    // if (success) {
    //     printf("Test passed!\n");
        
    //     // 打印部分结果
    //     printf("Column max values: %.1f, %.1f, %.1f, %.1f\n",
    //           h_data[0], h_data[1], h_data[2], h_data[3]);
        
    //     printf("\nWarp 0 first 16 threads (8x4 layout):\n");
    //     for (int r = 0; r < 8; ++r) {
    //         for (int c = 0; c < 4; ++c) {
    //             int lane = r * 4 + c;
    //             printf("%5.1f ", h_data[lane]);
    //         }
    //         printf("\n");
    //     }
    // }
    
    // 清理
    delete[] h_data;
    cudaFree(d_data);

    return 0;

    
    // return success ? 0 : 1;
}
