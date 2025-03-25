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
    // static constexpr int kBlockM = 16;
    static constexpr int kBlockM = 64;
    static constexpr int kNWarpsS = 4;

    using Element = cutlass::half_t;
    using ElementAccum = float;

    static constexpr int AtomLayoutNO = 2;
    using TiledMmaO = decltype(make_tiled_mma(
            cute::GMMA::rs_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kHeadDimV / AtomLayoutNO>, Int<kBlockN>>,
                    GMMA::Major::K, GMMA::Major::MN>(),
            Layout<Shape<Int<kNWarpsS / 4>, Int<AtomLayoutNO>, _1>>{}));

    using SmemLayoutV = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    Tensor sVt = make_tensor<Element>(Shape<Int<kHeadDimV>, Int<kBlockM>>{});

    int tidx = 1;
    TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrVt = thr_mma_o.partition_fragment_A(sVt);                // (MMA, MMA_K,MMA_N)
    Tensor tOrO = partition_fragment_C(tiled_mma_o, Shape<Int<kHeadDimV>, Int<kBlockM>>{});  // ((MMA=4, X), MMA_M, MMA_N=1)


    print(tOrO);
    std::cout << std::endl;
    std::cout << size<0>(tOrO) << std::endl;
    std::cout << size<1>(tOrO) << std::endl;
    std::cout << size<2>(tOrO) << std::endl;





    return 0;
}
