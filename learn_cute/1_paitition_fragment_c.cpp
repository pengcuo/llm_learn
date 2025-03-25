#include <iostream>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

using namespace cute;

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
    // static constexpr int kBlockM = 64;
    static constexpr int kNWarpsS = 4;

    // using SmemLayoutV = decltype(tile_to_shape(
    //         getSmemLayoutK<cutlass::bfloat16_t, kHeadDim, kHeadDimV>(),
    //         Shape<Int<kBlockN>, Int<kHeadDimV>>{}));

    // // using SmemLayoutV = decltype(make_layout(Shape<Int<kBlockN>, Int<kHeadDimV>>{}, GenRowMajor{}));
    // SmemLayoutV smemLayoutV;
    // print(smemLayoutV);
    
    // using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    // std::cout << std::endl;

    // SmemLayoutVtransposed smemLayoutVtransposed;
    // print(smemLayoutVtransposed);

    // std::cout << std::endl;

    // static constexpr int kNThreadsS = 128;
    // using SmemLayoutP = Layout<Shape<Shape<_2, _2>, Int<kNThreadsS>, _1, Int<kBlockN / 8>>>;
    // SmemLayoutP smemLayoutP;
    // std::cout << "smemLayoutP:" << std::endl;
    // print(smemLayoutP);


    using TiledMma = decltype(make_tiled_mma(
        cute::GMMA::ss_op_selector<cutlass::bfloat16_t, cutlass::bfloat16_t, float, Shape<Int<kBlockN>, Int<kBlockM>, Int<kHeadDim>>,
                GMMA::Major::K, GMMA::Major::K>(),
        Layout<Shape<Int<kNWarpsS / 4>, _1, _1>>{}));

    // TiledMma tiled_mma;
    // Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<kBlockN>, Int<kBlockM>>{});
    // print(tSrS);



    TiledMma tiled_mma;
    Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<kBlockN>, Int<kBlockM>>{});

    auto thr_mma = tiled_mma.get_thread_slice(0);
    Tensor cS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});
    Tensor tScS = thr_mma.partition_C(cS);
    print(tScS);

    std::cout << std::endl;
    std::cout << "size tSrS : " << size(tSrS) << std::endl;

    for (int i = 0; i < size(tSrS); ++i) {
        // if constexpr (!Is_causal) {  // Just masking based on col
        //     if (int(get<0>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) tSrS(i) = -INFINITY;
        // } else {
        //     // Ensure seqlen_k - 1 - (n_block * kBlockN + col) >= (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
        //     // col <= seqlen_k - 1 - n_block * kBlockN - (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
        //     int row = int(get<1>(tScS(i)));
        //     int col_limit_right = seqlen_k - 1 - n_block * kBlockN - (params.seqlen_q - 1 - (m_block * kBlockM + row)) / params.ngroups;
        //     if (int(get<0>(tScS(i))) > col_limit_right) tSrS(i) = -INFINITY;
        // }
        int row = int(get<1>(tScS(i)));
        int col = int(get<0>(tScS(i)));
        std::cout << "row: " << row << " col: " << col << std::endl;
    }



    std::cout << std::endl;

    return 0;
}
