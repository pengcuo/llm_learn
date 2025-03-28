#include <iostream>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

using namespace cute;

// #include "named_barrier.h"
// #include "utils.h"
// #include "softmax.h"
// #include "static_switch.h"
// #include "flash_mla.h"


template<bool Transposed=false, typename Layout0>
auto xx_convert_layout_acc_rowcol(Layout0 acc_layout) {
    std::cout << "xx_convert_layout_acc_rowcol" << std::endl;
    if constexpr (decltype(rank<0>(acc_layout))::value == 3) {  // SM90
        static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
        static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = acc_layout;
        if constexpr (!Transposed) {
            std::cout << __LINE__ << std::endl;
            return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
        } else {
            std::cout << __LINE__ << std::endl;
             return make_layout(make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)), make_layout(get<0, 1>(l), get<1>(l)));
        }

    } else {  // SM80
        std::cout << __LINE__ << std::endl;
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
        if constexpr (!Transposed) {
            return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
        } else {
            return make_layout(make_layout(get<0, 0>(l), get<2>(l)), make_layout(get<0, 1>(l), get<1>(l)));
        }
    }
};

constexpr int kBlockN = 64;
constexpr int kHeadDimV = 512;

template<typename PrecType, int DIM, int DIM2 = DIM>
constexpr auto getSmemLayoutK() {
    constexpr int headSizeBytes = sizeof(PrecType) * DIM;
    constexpr int headSizeBytes2 = sizeof(PrecType) * DIM2;

    if constexpr (headSizeBytes % 128 == 0 && headSizeBytes2 % 128 == 0) {
        return GMMA::Layout_K_SW128_Atom<PrecType>{};
    } else if constexpr (headSizeBytes % 64 == 0 && headSizeBytes2 % 64 == 0) {
        return GMMA::Layout_K_SW64_Atom<PrecType>{};
    } else {
        return GMMA::Layout_K_SW32_Atom<PrecType>{};
    }
}



int main() {

    static constexpr int kHeadDim = 576;
    static constexpr int kHeadDimV = 512;
    static constexpr int kBlockN = 64;
    static constexpr int kBlockM = 16;
    static constexpr int kNWarpsS = 4;

    using Element = cutlass::half_t;
    using ElementAccum = float;


    auto xx = cute::make_tensor<float>(cute::make_shape(cute::Int<10>{}, cute::Int<20>{}));

    int idx = 10;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 20; ++j) {
            // xx(i, j) = i * 20 + j;  // 按行存储
            xx(i, j) = idx++;  // 按行存储
        }
    }

    std::cout << "tensor(2, 3) = " << xx(2, 3) << std::endl;  // 2*20 + 3 = 43

    return 0;

}
