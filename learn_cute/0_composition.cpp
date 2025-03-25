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

    using SmemLayoutV = decltype(tile_to_shape(
            getSmemLayoutK<cutlass::bfloat16_t, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDimV>>{}));

    // using SmemLayoutV = decltype(make_layout(Shape<Int<kBlockN>, Int<kHeadDimV>>{}, GenRowMajor{}));
    SmemLayoutV smemLayoutV;
    print(smemLayoutV);
    
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));
    // using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{})));

    std::cout << std::endl;

    SmemLayoutVtransposed smemLayoutVtransposed;
    print(smemLayoutVtransposed);

    std::cout << std::endl;

    static constexpr int kNThreadsS = 128;
    using SmemLayoutP = Layout<Shape<Shape<_2, _2>, Int<kNThreadsS>, _1, Int<kBlockN / 8>>>;
    SmemLayoutP smemLayoutP;
    std::cout << "smemLayoutP:" << std::endl;
    print(smemLayoutP);


    std::cout << std::endl;

    return 0;
}
