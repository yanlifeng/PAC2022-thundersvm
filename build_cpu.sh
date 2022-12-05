rm -rf build

#CC=icpx
CC=dpcpp

echo "##${MKLROOT}##"

#mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=$CC -DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_PAPI=OFF -DCMAKE_CXX_FLAGS="-qopenmp" -DCMAKE_MODULE_PATH=. -DOpenMP_CXX_FLAGS="-qopenmp" -DOpenMP_CXX_LIB_NAMES="libiomp5" -DOpenMP_libiomp5_LIBRARY=/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so .. && make -j
#mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=$CC -DOpenMP_CXX_FLAGS="-qopenmp" -DOpenMP_CXX_LIB_NAMES="libiomp5" -DOpenMP_libiomp5_LIBRARY=/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so -DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_PAPI=OFF -DCMAKE_CXX_FLAGS="-qopenmp -DEIGEN_DONT_VECTORIZE -D__SYCL_DISABLE_NAMESPACE_INLINE__" -DCMAKE_C_FLAGS="-DEIGEN_DONT_VECTORIZE -D__SYCL_DISABLE_NAMESPACE_INLINE__" -DCMAKE_MODULE_PATH=. .. && make -j
#mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=$CC -DOpenMP_CXX_FLAGS="-qopenmp" -DOpenMP_CXX_LIB_NAMES="libiomp5" -DOpenMP_libiomp5_LIBRARY=/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so -DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_PAPI=OFF -DCMAKE_CXX_FLAGS="-qopenmp" -DCMAKE_MODULE_PATH=. .. && make -j
mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=$CC -DOpenMP_CXX_FLAGS="-qopenmp" -DOpenMP_CXX_LIB_NAMES="libiomp5" -DOpenMP_libiomp5_LIBRARY=/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so -DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_PAPI=OFF -DCMAKE_MODULE_PATH=. .. && make -j
