
echo $1
#g++ -I csrc/cutlass/include/ -I /usr/local/cuda/include -I csrc/ -std=c++17 $1
nvcc -arch=compute_90a -code=sm_90a -I csrc/cutlass/include/ -I /usr/local/cuda/include -I csrc/ -std=c++17 $1
./a.out
