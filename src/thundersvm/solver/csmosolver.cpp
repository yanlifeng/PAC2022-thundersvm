#include <thundersvm/solver/csmosolver.h>
#include <thundersvm/kernel/smo_kernel.h>
#include <climits>
//#include <thundersvm/time_static.h>

#include "mkl.h"
#include "mkl_omp_offload.h"
#include <omp.h>
#include <thread>
#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <atomic>
#include <cassert>
#include <CL/sycl.hpp>
#include "mkl.h"
#include "oneapi/mkl/blas.hpp"
#include "oneapi/mkl/spblas.hpp"
#include <ipp.h>
#include <ippcore.h>
#include <ippvm.h>
#include "tbb/parallel_for.h"

#include <assert.h>

#include <chrono>
#include <sys/time.h>
double GetTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000;
}

typedef std::chrono::high_resolution_clock Clock;
//#define TDEF(x_) std::chrono::high_resolution_clock::time_point x_##_t0, x_##_t1;
//#define TSTART(x_) x_##_t0 = Clock::now();
//#define TEND(x_) x_##_t1 = Clock::now();
//#define TPRINT(x_, str) printf("%-20s \t%.6f\t sec\n", str, std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()/1e6);
//#define TINT(x_) std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()

#define TDEF(x_)
#define TSTART(x_)
#define TEND(x_)
#define TPRINT(x_, str)
#define TINT(x_) 0

/*
 *
 *
 *
 *
 */


const int BCSR_Block_Size = 2048;


struct SparseData_BCSR{
    std::vector<kernel_type> val_data;
    std::vector<int>  row_ptr;
    std::vector<int>  col_begin_ptr;
    std::vector<int>  col_end_ptr;
    std::vector<int>  col_ptr;
    int block_size;
    int total_num;
};

/*
struct SparseData{
    std::vector<kernel_type> val_data;
    std::vector<int> row_ptr;
    std::vector<int> col_ptr;
    int* table;
};


struct DenseData{
    kernel_type *val;
    int m;
    int n;
    int* Ttable;
    int* Ftable;
};


struct Node{
    int num;
    int x;
};
*/
struct SparseData{
    std::vector<kernel_type> val_data;
    std::vector<int> row_ptr;
    std::vector<int> col_ptr;
    int* table;
    int row;
    int col;
    bool is_use;
};


struct DenseData{
    kernel_type *val;
    int row;
    int col;
    int* Ttable;
    int* Ftable;
    bool is_use;
};


struct Node{
    int num;
    int x;
};



struct MyType{
	int x;
	int y;
	int xlen;
	int ylen;
	int nnz;
	bool is_dense;

	//for sparse

	oneapi::mkl::sparse::matrix_handle_t handle;
	kernel_type *csr_val;
	int *csr_row_ptr;
	int *csr_col_ind;
	//for dense
	kernel_type *dense;
};	


struct MyType2{
	int x;
	int y;
	int xlen;
	int ylen;
	int nnz;
	bool is_dense;

	//for sparse on host
	kernel_type *csr_val_host;
	int *csr_col_ind_host;
	int *csr_row_ptr_host;

	//for sparse on gpu0
	oneapi::mkl::sparse::matrix_handle_t handle0;

	kernel_type *csr_val0;
	int *csr_col_ind0;
	int *csr_row_ptr0;

	//for sparse on gpu1
	oneapi::mkl::sparse::matrix_handle_t handle1;

	kernel_type *csr_val1;
	int *csr_col_ind1;
	int *csr_row_ptr1;

	//for dense on host
	kernel_type *dense_host;

	//for dense on gpu0
	kernel_type *dense0;

	//for dense on gpu1
	kernel_type *dense1;
};



void dns_csr_mul_block_one_gpu(int mm, int nn, int kk, kernel_type *dense_mat, kernel_type *csr_val,
		int *csr_row_ptr, int *csr_col_ind, int nnz,
		kernel_type *result)
{
	long long m = mm;
	long long n = nn;
	long long k = kk;
	static bool inited = false;
	static sycl::queue q(sycl::device{sycl::gpu_selector()});

	static constexpr double sparse_density = 0.4;
	static constexpr long long M = 3458;
	static constexpr long long N = 231;
	static constexpr long long K = 5431;
	//const int M = m;
	//const int N = n;
	//const int K = 20958;

	static const int bm = (m + M - 1) / M;
	const int bn = (n + N - 1) / N;
	static const int bk = (k + K - 1) / K;

	static MyType *A;
	static kernel_type *B; //dense_mat on gpu
	static kernel_type *ABblock;
	static kernel_type *NM;

	if(!inited)
	{
		inited = true;

		std::cout << "intied start" << std::endl;
		const kernel_type *csr_val_ptr = csr_val;
		const int *csr_col_ind_ptr = csr_col_ind;
		const int *csr_row_ptr_ptr = csr_row_ptr;

		B = sycl::malloc_device<kernel_type>(k * n, q);
		ABblock = sycl::malloc_shared<kernel_type>(M * N * bk, q);

		A = new MyType[bm * bk];

		int *nnzcount = new int[bk];
		for(int i = 0; i < bm; ++i)
		{
			int mm = std::min(M, m - i * M);
			memset(nnzcount, 0, sizeof(int) * bk);

			int jj = std::min(i * M + M, m);
			for(int j = i * M; j < jj; ++j)
			{
				int sidx = csr_row_ptr_ptr[j];
				int eidx = csr_row_ptr_ptr[j + 1];
				while(sidx < eidx)
				{
					int idx = csr_col_ind_ptr[sidx];
					++nnzcount[idx / K];
					++sidx;
				}
			}

			MyType *Atmp = A + i * bk;
			for(int j = 0; j < bk; ++j)
			{
				int kk = std::min(K, k - j * K);
				double density = ((double)(nnzcount[j])) / (mm * kk);

				Atmp[j].nnz = nnzcount[j];

				if(density <= sparse_density)
				{
					Atmp[j].is_dense = false;
					if(Atmp[j].nnz != 0)
					{
						Atmp[j].csr_val = sycl::malloc_shared<kernel_type>(Atmp[j].nnz, q);
						Atmp[j].csr_col_ind = sycl::malloc_shared<int>(Atmp[j].nnz, q);
						Atmp[j].csr_row_ptr = sycl::malloc_shared<int>(mm + 1, q);
					}
				}
				else
				{
					Atmp[j].is_dense = true;
					Atmp[j].dense = sycl::malloc_shared<kernel_type>(mm * kk, q);
					memset(Atmp[j].dense, 0, sizeof(kernel_type) * mm * kk);
				}
			}

			memset(nnzcount, 0, sizeof(int) * bk);
			for(int j = i * M; j < jj; ++j)
			{
				int j0 = j - i * M;
				for(int x = 0; x < bk; ++x)
				{
					if(!Atmp[x].is_dense && Atmp[x].nnz != 0)
						Atmp[x].csr_row_ptr[j0] = nnzcount[x];
				}

				int sidx = csr_row_ptr_ptr[j];
				int eidx = csr_row_ptr_ptr[j + 1];
				while(sidx < eidx)
				{
					int idx = csr_col_ind_ptr[sidx];
					int blockidx = idx / K;
					int innerblockidx = idx % K;

					int kk = std::min(K, k - blockidx * K);

					if(Atmp[blockidx].is_dense)
					{
						Atmp[blockidx].dense[j0 * kk + innerblockidx] = csr_val_ptr[sidx];
					}
					else
					{
						Atmp[blockidx].csr_val[nnzcount[blockidx]] = csr_val_ptr[sidx];
						Atmp[blockidx].csr_col_ind[nnzcount[blockidx]] = innerblockidx;
						++nnzcount[blockidx];
					}

					++sidx;
				}
			}

			{

				//TODO 行主列主转换
				for(int j = 0; j < bk; ++j)
				{
					if(Atmp[j].nnz != 0)
					{
						int kk = std::min(K, k - j * K);

						if(Atmp[j].is_dense)
						{
							//mm * kk 列主
							oneapi::mkl::blas::row_major::imatcopy_batch(q, oneapi::mkl::transpose::T, mm, kk, 1, Atmp[j].dense, kk, mm, mm * kk, 1).wait();
						}
						else //TODO
						{
							Atmp[j].csr_row_ptr[mm] = nnzcount[j];
							oneapi::mkl::sparse::init_matrix_handle(&(Atmp[j].handle));
							oneapi::mkl::sparse::set_csr_data(Atmp[j].handle, mm, kk, oneapi::mkl::index_base::zero, Atmp[j].csr_row_ptr, Atmp[j].csr_col_ind, Atmp[j].csr_val);
						}
					}
				}
			}

		}
		std::cout << "intied finish" << std::endl;
	}

	static int itercount = 0;
	std::cout << itercount << std::endl;
	++itercount;

	const kernel_type *dense_mat_ptr = dense_mat;
	kernel_type *result_ptr = result;

	//q.memcpy(B, dense_mat_ptr, sizeof(kernel_type) * k * n).wait();
	auto e0 = q.memcpy(B, dense_mat_ptr, sizeof(kernel_type) * k * n);

	for(int i = 0; i < bm; ++i)
	{
		int mm = std::min(M, m - i * M);

		for(int j = 0; j < bn; ++j)
		{
			int nn = std::min(N, n - j * N);

			vector<sycl::event> events(bk);

			for(int x = 0; x < bk; ++x)
			{
				int kk = std::min(K, k - x * K);

				MyType *Ablock = A + i * bk + x;
				kernel_type *tmpresult = ABblock + x * M * N;

				if(Ablock->nnz != 0)
				{
					if(Ablock->is_dense)
					{
						//TODO ABblock 初始化为0
						//oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, mm, nn, kk, 1, Ablock->dense, mm, B + x * K + j * N * k, k, 0, tmpresult, mm).wait();
						events[x] = oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, mm, nn, kk, 1, Ablock->dense, mm, B + x * K + j * N * k, k, 0, ABblock + x * M * N, mm, {e0});
					}
					else
					{
						//oneapi::mkl::sparse::gemm(q, oneapi::mkl::layout::C, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, 1, Ablock->handle, B + x * K + j * N * k, nn, k, 0, tmpresult, mm).wait();
						events[x] = oneapi::mkl::sparse::gemm(q, oneapi::mkl::layout::C, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, 1, Ablock->handle, B + x * K + j * N * k, nn, k, 0, ABblock + x * M * N, mm, {e0});
					}
				}
				else
				{
					{
						//q.memset(tmpresult, 0, sizeof(kernel_type) * mm * nn).wait();
						events[x] = q.memset(ABblock + x * M * N, 0, sizeof(kernel_type) * mm * nn);
					}
				}

			}

			events[0].wait();

			for(int x = 1; x < bk; ++x)
			{
				events[x].wait();
				int kk = std::min(K, k - x * K);
				//oneapi::mkl::blas::column_major::omatadd_batch(q, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, mm, nn, 1, NM, mm, 0, 1, ABblock + x * M * N, mm, 0, NM, mm, 0, 1);
				oneapi::mkl::blas::column_major::omatadd_batch(q, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, mm, nn, 1, ABblock, mm, 0, 1, ABblock + x * M * N, mm, 0, ABblock, mm, mm * nn, 1);
			}

			q.wait();

			for(int x = 0; x < nn; ++x)
				memcpy(result_ptr + (j * N + x) * m + i * M, ABblock + x * mm, sizeof(kernel_type) * mm);
		}
	}

}





void dns_csr_mul_block_two_gpu(int mm, int nn, int kk, kernel_type *dense_mat, kernel_type *csr_val,
		int *csr_row_ptr, int *csr_col_ind, int nnz,
		kernel_type *result)
{
	long long m = mm;
	long long n = nn;
	long long k = kk;
	static bool inited = false;
	//static sycl::queue q0(sycl::device{sycl::gpu_selector()});
	//static sycl::queue q1(sycl::device{sycl::gpu_selector()});
	static sycl::queue q0;
	static sycl::queue q1;

	static constexpr double sparse_density = 0.1;
	static constexpr long long M = 3457;
	static constexpr long long N = 128;
	static constexpr long long K = 2341;
	//const int M = m;
	//const int N = n;
	//const int K = 20958;

	static const int bm = (m + M - 1) / M;
	const int bn = (n + N - 1) / N;
	static const int bk = (k + K - 1) / K;

	static MyType2 *A;
	static int Asize;
	static kernel_type *B0; //dense_mat on gpu0
	static kernel_type *B1; //dense_mat on gpu1
	static kernel_type *ABblock0;
	static kernel_type *ABblock1;

	if(!inited)
	{
		inited = true;

		std::vector<sycl::queue> qs;
		auto platforms = sycl::platform::get_platforms();
		for(auto &platform : platforms)
		{
			if(platform.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
				continue;

			auto devices = platform.get_devices();
			for(auto &device : devices)
			{
				if(device.is_gpu())
					qs.push_back(sycl::queue(device));
			}
		}

		if(qs.size() != 2)
		{
			std::cout << "Not found 2 devices" << std::endl;
			exit(1);
		}

		q0 = qs[0];
		q1 = qs[1];

		std::cout << "intied start" << std::endl;
		const kernel_type *csr_val_ptr = csr_val;
		const int *csr_col_ind_ptr = csr_col_ind;
		const int *csr_row_ptr_ptr = csr_row_ptr;

		B0 = sycl::malloc_device<kernel_type>(k * n, q0);
		B1 = sycl::malloc_device<kernel_type>(k * n, q1);

		ABblock0 = sycl::malloc_shared<kernel_type>(M * N * bk, q0);
		ABblock1 = sycl::malloc_shared<kernel_type>(M * N * bk, q1);

		Asize = bm * bk;
		A = new MyType2[Asize];

		//std::cout << __LINE__ << std::endl;

		int *nnzcount = new int[bk];
		for(int i = 0; i < bm; ++i)
		{
			//std::cout << __LINE__ << std::endl;

			int mm = std::min(M, m - i * M);
			memset(nnzcount, 0, sizeof(int) * bk);

			int jj = std::min(i * M + M, m);
			for(int j = i * M; j < jj; ++j)
			{
				int sidx = csr_row_ptr_ptr[j];
				int eidx = csr_row_ptr_ptr[j + 1];
				while(sidx < eidx)
				{
					int idx = csr_col_ind_ptr[sidx];
					++nnzcount[idx / K];
					++sidx;
				}
			}

			//std::cout << __LINE__ << std::endl;

			MyType2 *Atmp = A + i * bk;

			for(int j = 0; j < bk; ++j)
			{
				int kk = std::min(K, k - j * K);
				double density = ((double)(nnzcount[j])) / (mm * kk);

				Atmp[j].nnz = nnzcount[j];

				if(density <= sparse_density)
				{
					Atmp[j].is_dense = false;
					if(Atmp[j].nnz != 0)
					{
						//Atmp[j].csr_val = sycl::malloc_shared<kernel_type>(Atmp[j].nnz, q);
						//Atmp[j].csr_col_ind = sycl::malloc_shared<int>(Atmp[j].nnz, q);
						//Atmp[j].csr_row_ptr = sycl::malloc_shared<int>(mm + 1, q);

						Atmp[j].csr_val_host = new kernel_type[Atmp[j].nnz];
						Atmp[j].csr_col_ind_host = new int[Atmp[j].nnz];
						Atmp[j].csr_row_ptr_host = new int[mm + 1];
					}
				}
				else
				{
					Atmp[j].is_dense = true;
					//Atmp[j].dense = sycl::malloc_shared<kernel_type>(mm * kk, q);
					//memset(Atmp[j].dense, 0, sizeof(kernel_type) * mm * kk);
					Atmp[j].dense_host = new kernel_type[mm * kk];
					memset(Atmp[j].dense_host, 0, sizeof(kernel_type) * mm * kk);
				}
			}

			//std::cout << __LINE__ << std::endl;

			memset(nnzcount, 0, sizeof(int) * bk);
			for(int j = i * M; j < jj; ++j)
			{
				int j0 = j - i * M;
				for(int x = 0; x < bk; ++x)
				{
					if(!Atmp[x].is_dense && Atmp[x].nnz != 0)
						//Atmp[x].csr_row_ptr[j0] = nnzcount[x];
						Atmp[x].csr_row_ptr_host[j0] = nnzcount[x];
				}

				int sidx = csr_row_ptr_ptr[j];
				int eidx = csr_row_ptr_ptr[j + 1];
				while(sidx < eidx)
				{
					int idx = csr_col_ind_ptr[sidx];
					int blockidx = idx / K;
					int innerblockidx = idx % K;

					int kk = std::min(K, k - blockidx * K);

					if(Atmp[blockidx].is_dense)
					{
						//Atmp[blockidx].dense[j0 * kk + innerblockidx] = csr_val_ptr[sidx];
						Atmp[blockidx].dense_host[j0 * kk + innerblockidx] = csr_val_ptr[sidx];
					}
					else
					{
						//Atmp[blockidx].csr_val[nnzcount[blockidx]] = csr_val_ptr[sidx];
						//Atmp[blockidx].csr_col_ind[nnzcount[blockidx]] = innerblockidx;
						Atmp[blockidx].csr_val_host[nnzcount[blockidx]] = csr_val_ptr[sidx];
						Atmp[blockidx].csr_col_ind_host[nnzcount[blockidx]] = innerblockidx;
						++nnzcount[blockidx];
					}

					++sidx;
				}
			}

			//std::cout << __LINE__ << std::endl;

			//TODO 行主列主转换
			for(int j = 0; j < bk; ++j)
			{
				if(Atmp[j].nnz != 0)
				{
					int kk = std::min(K, k - j * K);

					if(Atmp[j].is_dense)
					{
						//std::cout << __LINE__ << std::endl;
						Atmp[j].dense0 = sycl::malloc_device<kernel_type>(mm * kk, q0);
						Atmp[j].dense1 = sycl::malloc_device<kernel_type>(mm * kk, q1);

						auto e0 = q0.memcpy(Atmp[j].dense0, Atmp[j].dense_host, sizeof(kernel_type) * mm * kk);
						auto e1 = q1.memcpy(Atmp[j].dense1, Atmp[j].dense_host, sizeof(kernel_type) * mm * kk);

						oneapi::mkl::blas::row_major::imatcopy_batch(q0, oneapi::mkl::transpose::T, mm, kk, 1, Atmp[j].dense0, kk, mm, mm * kk, 1, {e0});
						oneapi::mkl::blas::row_major::imatcopy_batch(q1, oneapi::mkl::transpose::T, mm, kk, 1, Atmp[j].dense1, kk, mm, mm * kk, 1, {e1});

						//mm * kk 列主
						//oneapi::mkl::blas::row_major::imatcopy_batch(q, oneapi::mkl::transpose::T, mm, kk, 1, Atmp[j].dense, kk, mm, mm * kk, 1).wait();

						//std::cout << __LINE__ << std::endl;

					}
					else
					{
						//std::cout << __LINE__ << std::endl;

						Atmp[j].csr_row_ptr_host[mm] = nnzcount[j];
						//-------------------------

						Atmp[j].csr_val0 = sycl::malloc_device<kernel_type>(nnzcount[j], q0);

						//std::cout << __LINE__ << std::endl;
						Atmp[j].csr_val1 = sycl::malloc_device<kernel_type>(nnzcount[j], q1);
						//std::cout << __LINE__ << std::endl;

						Atmp[j].csr_col_ind0 = sycl::malloc_device<int>(nnzcount[j], q0);
						//std::cout << __LINE__ << std::endl;
						Atmp[j].csr_col_ind1 = sycl::malloc_device<int>(nnzcount[j], q1);
						//std::cout << __LINE__ << std::endl;

						Atmp[j].csr_row_ptr0 = sycl::malloc_device<int>(mm + 1, q0);
						//std::cout << __LINE__ << std::endl;
						Atmp[j].csr_row_ptr1 = sycl::malloc_device<int>(mm + 1, q1);
						//std::cout << __LINE__ << std::endl;
						//-------------------------

						q0.memcpy(Atmp[j].csr_val0, Atmp[j].csr_val_host, sizeof(kernel_type) * nnzcount[j]).wait();
						//std::cout << __LINE__ << std::endl;
						q1.memcpy(Atmp[j].csr_val1, Atmp[j].csr_val_host, sizeof(kernel_type) * nnzcount[j]).wait();
						//std::cout << __LINE__ << std::endl;

						q0.memcpy(Atmp[j].csr_col_ind0, Atmp[j].csr_col_ind_host, sizeof(int) * nnzcount[j]).wait();
						//std::cout << __LINE__ << std::endl;
						q1.memcpy(Atmp[j].csr_col_ind1, Atmp[j].csr_col_ind_host, sizeof(int) * nnzcount[j]).wait();
						//std::cout << __LINE__ << std::endl;

						q0.memcpy(Atmp[j].csr_row_ptr0, Atmp[j].csr_row_ptr_host, sizeof(int) * (mm + 1)).wait();
						//std::cout << __LINE__ << std::endl;
						q1.memcpy(Atmp[j].csr_row_ptr1, Atmp[j].csr_row_ptr_host, sizeof(int) * (mm + 1)).wait();
						//std::cout << __LINE__ << std::endl;

						q0.wait();
						q1.wait();


						oneapi::mkl::sparse::init_matrix_handle(&(Atmp[j].handle0));
						oneapi::mkl::sparse::init_matrix_handle(&(Atmp[j].handle1));

						oneapi::mkl::sparse::set_csr_data(Atmp[j].handle0, mm, kk, oneapi::mkl::index_base::zero, Atmp[j].csr_row_ptr0, Atmp[j].csr_col_ind0, Atmp[j].csr_val0);
						oneapi::mkl::sparse::set_csr_data(Atmp[j].handle1, mm, kk, oneapi::mkl::index_base::zero, Atmp[j].csr_row_ptr1, Atmp[j].csr_col_ind1, Atmp[j].csr_val1);

						//std::cout << __LINE__ << std::endl;

						//Atmp[j].csr_row_ptr[mm] = nnzcount[j];
						//oneapi::mkl::sparse::init_matrix_handle(&(Atmp[j].handle));
						//oneapi::mkl::sparse::set_csr_data(Atmp[j].handle, mm, kk, oneapi::mkl::index_base::zero, Atmp[j].csr_row_ptr, Atmp[j].csr_col_ind, Atmp[j].csr_val);
					}
				}
			}

			//std::cout << __LINE__ << std::endl;

		}
		std::cout << "intied finish" << std::endl;
	}

	static int itercount = 0;
	std::cout << itercount << std::endl;
	++itercount;

	const kernel_type *dense_mat_ptr = dense_mat;
	kernel_type *result_ptr = result;

	//q.memcpy(B, dense_mat_ptr, sizeof(kernel_type) * k * n).wait();
	auto e0 = q0.memcpy(B0, dense_mat_ptr, sizeof(kernel_type) * k * n);
	auto e1 = q1.memcpy(B1, dense_mat_ptr, sizeof(kernel_type) * k * n);

	std::atomic_int block_idx(0);

	std::thread t0([&]()
			{
			const int Csize = bm * bn;

			for(;;)
			{
			int idx = block_idx++;
			if(idx >= Csize)
			break;

			int i = idx / bn;

			int j = idx % bn;

			int mm = std::min(M, m - i * M);
			int nn = std::min(N, n - j * N);

			vector<sycl::event> events(bk);

			for(int x = 0; x < bk; ++x)
			{
				int kk = std::min(K, k - x * K);

				MyType2 *Ablock = A + i * bk + x;
				kernel_type *tmpresult = ABblock0 + x * M * N;

				if(Ablock->nnz != 0)
				{
					if(Ablock->is_dense)
					{
						events[x] = oneapi::mkl::blas::column_major::gemm(q0, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, mm, nn, kk, 1, Ablock->dense0, mm, B0 + x * K + j * N * k, k, 0, ABblock0 + x * M * N, mm, {e0});
					}
					else
					{
						events[x] = oneapi::mkl::sparse::gemm(q0, oneapi::mkl::layout::C, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, 1, Ablock->handle0, B0 + x * K + j * N * k, nn, k, 0, ABblock0 + x * M * N, mm, {e0});
					}
				}
				else
				{
					events[x] = q0.memset(ABblock0 + x * M * N, 0, sizeof(kernel_type) * mm * nn);
				}
			}

			//-------------
			events[0].wait();

			for(int x = 1; x < bk; ++x)
			{
				events[x].wait();
				int kk = std::min(K, k - x * K);
				oneapi::mkl::blas::column_major::omatadd_batch(q0, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, mm, nn, 1, ABblock0, mm, 0, 1, ABblock0 + x * M * N, mm, 0, ABblock0, mm, mm * nn, 1).wait();
			}

			q0.wait();

			//-------------
			for(int x = 0; x < nn; ++x)
				memcpy(result_ptr + (j * N + x) * m + i * M, ABblock0 + x * mm, sizeof(kernel_type) * mm);

			q0.wait();

			}
			});

	std::thread t1([&]()
			{
			const int Csize = bm * bn;

			for(;;)
			{
			int idx = block_idx++;
			if(idx >= Csize)
			break;

			int i = idx / bn;

			int j = idx % bn;

			int mm = std::min(M, m - i * M);
			int nn = std::min(N, n - j * N);

			vector<sycl::event> events(bk);

			for(int x = 0; x < bk; ++x)
			{
				int kk = std::min(K, k - x * K);

				MyType2 *Ablock = A + i * bk + x;
				kernel_type *tmpresult = ABblock1 + x * M * N;

				if(Ablock->nnz != 0)
				{
					if(Ablock->is_dense)
					{
						events[x] = oneapi::mkl::blas::column_major::gemm(q1, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, mm, nn, kk, 1, Ablock->dense1, mm, B1 + x * K + j * N * k, k, 0, ABblock1 + x * M * N, mm, {e1});
					}
					else
					{
						events[x] = oneapi::mkl::sparse::gemm(q1, oneapi::mkl::layout::C, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, 1, Ablock->handle1, B1 + x * K + j * N * k, nn, k, 0, ABblock1 + x * M * N, mm, {e1});
					}
				}
				else
				{
					events[x] = q1.memset(ABblock1 + x * M * N, 0, sizeof(kernel_type) * mm * nn);
				}
			}

			//-------------
			events[0].wait();

			for(int x = 1; x < bk; ++x)
			{
				events[x].wait();
				int kk = std::min(K, k - x * K);
				oneapi::mkl::blas::column_major::omatadd_batch(q1, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, mm, nn, 1, ABblock1, mm, 0, 1, ABblock1 + x * M * N, mm, 0, ABblock1, mm, mm * nn, 1).wait();
			}

			q1.wait();

			//-------------
			for(int x = 0; x < nn; ++x)
				memcpy(result_ptr + (j * N + x) * m + i * M, ABblock1 + x * mm, sizeof(kernel_type) * mm);

			q1.wait();

			}
			});

	t0.join();
	t1.join();

	//for(int i = 0; i < bm; ++i)
	//{
	//    int mm = std::min(M, m - i * M);

	//    for(int j = 0; j < bn; ++j)
	//    {
	//        int nn = std::min(N, n - j * N);

	//        vector<sycl::event> events(bk);

	//        for(int x = 0; x < bk; ++x)
	//        {
	//            int kk = std::min(K, k - x * K);

	//            MyType2 *Ablock = A + i * bk + x;
	//            kernel_type *tmpresult = ABblock + x * M * N;

	//            if(Ablock->nnz != 0)
	//            {
	//                if(Ablock->is_dense)
	//                {
	//                    //TODO ABblock 初始化为0
	//                    //oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, mm, nn, kk, 1, Ablock->dense, mm, B + x * K + j * N * k, k, 0, tmpresult, mm).wait();
	//                    events[x] = oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, mm, nn, kk, 1, Ablock->dense, mm, B + x * K + j * N * k, k, 0, ABblock + x * M * N, mm, {e0});
	//                }
	//                else
	//                {
	//                    //oneapi::mkl::sparse::gemm(q, oneapi::mkl::layout::C, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, 1, Ablock->handle, B + x * K + j * N * k, nn, k, 0, tmpresult, mm).wait();
	//                    events[x] = oneapi::mkl::sparse::gemm(q, oneapi::mkl::layout::C, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, 1, Ablock->handle, B + x * K + j * N * k, nn, k, 0, ABblock + x * M * N, mm, {e0});
	//                }
	//            }
	//            else
	//            {
	//                {
	//                //q.memset(tmpresult, 0, sizeof(kernel_type) * mm * nn).wait();
	//                events[x] = q.memset(ABblock + x * M * N, 0, sizeof(kernel_type) * mm * nn);
	//                }
	//            }

	//        }

	//        events[0].wait();

	//        for(int x = 1; x < bk; ++x)
	//        {
	//            events[x].wait();
	//            int kk = std::min(K, k - x * K);
	//            //oneapi::mkl::blas::column_major::omatadd_batch(q, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, mm, nn, 1, NM, mm, 0, 1, ABblock + x * M * N, mm, 0, NM, mm, 0, 1);
	//            oneapi::mkl::blas::column_major::omatadd_batch(q, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, mm, nn, 1, ABblock, mm, 0, 1, ABblock + x * M * N, mm, 0, ABblock, mm, mm * nn, 1);
	//        }

	//        q.wait();

	//        for(int x = 0; x < nn; ++x)
	//            memcpy(result_ptr + (j * N + x) * m + i * M, ABblock + x * mm, sizeof(kernel_type) * mm);
	//    }
	//}

}


void CSRtoDenseandCSR(const KernelMatrix &k_mat,DenseData &denseData_cpu,SparseData &sparseData_cpu,DenseData &denseData_gpu,SparseData &sparseData_gpu){
    const long long m = k_mat.n_instances_;

    const long long n = k_mat.n_features_;


    const kernel_type * csr_val = k_mat.val_.host_data();
    const int * csr_row_ptr = k_mat.row_ptr_.host_data();
    const int * csr_col_ind = k_mat.col_ind_.host_data();



    Node* col_num = new Node[n+10];

    for (int i=0;i<n;i++) {
        col_num[i].num=0;
        col_num[i].x=i;
    }


    for (int i=0;i<m;i++){
        int csr_row_begin = csr_row_ptr[i];
        int csr_row_end = csr_row_ptr[i+1];
        for (int j=csr_row_begin;j<csr_row_end;j++){
            col_num[csr_col_ind[j]].num++;
        }
    }

    std::sort(col_num,col_num+n,[=](Node a,Node b){
        if (a.num<b.num) return true;
        if (a.num==b.num && a.x < b.x) return true;
        return false;
    });


    /*
     * 考虑一下CSR部分以及DENSE部分的数据量大小问题
     *
     *
     */

    long long denseDataNum = 0;
    long long csrDataNum = k_mat.nnz_;





    long long t_num=0;
    long long a_num=0;

    bool* densefg = new bool[n];
    for (int i=0;i<n;i++) densefg[i]=true;

    bool* gpufg = new bool[n];
    for (int i=0;i<n;i++) gpufg[i]=false;

    int densecolnum=n;


    // 密度从小到大



    long long total_csr=0;

    long long total_dense=m*n;


    for (int i=0;i<n;i++){
    
        total_csr+=col_num[i].num;
        total_dense-=m;


        t_num+=col_num[i].num;
        a_num+=m;
        double ans=1.0l*t_num/a_num;
        //printf("MIDU : %lf\n",ans);
        if (ans > 0.02 || 1.0l*col_num[i].num/m > 0.3 ||total_csr * 8 > total_dense){
            break;
        }
		densecolnum--;
        densefg[col_num[i].x]= false;


    }


    //printf("dense col num : %d\n",densecolnum);



    /*
     * 决定数据的CPU和GPU的归属
     */

    long long gpu_memory_size =  8192LL <<20;

    for (int i=n-1;i>=0;i--){
        if (densefg[col_num[i].x]==true){
            gpu_memory_size -= m*sizeof(kernel_type);
        } else {
            gpu_memory_size -= col_num[i].num*sizeof(kernel_type)*2LL;
        }


        if (gpu_memory_size < 0) break;
        //printf("Index : %d |  %lld\n",i,gpu_memory_size);

        gpufg[col_num[i].x]=true;
    }





    int dense_col_num_cpu=0;
    int dense_col_num_gpu=0;

    for (int i=0;i<n;i++){

        if (gpufg[i] == true){
            dense_col_num_gpu += densefg[i] == true ? 1: 0;
        } else {
            dense_col_num_cpu += densefg[i] == true ? 1: 0;
        }

    }
    //printf("Dense Col CPU num : %d\n",dense_col_num_cpu);
    //printf("Dense Col GPU num : %d\n",dense_col_num_gpu);

//    DenseData denseData_cpu;
//    DenseData denseData_gpu;
//
//    SparseData sparseData_cpu;
//    SparseData sparseData_gpu;



    if (dense_col_num_cpu > 0){
        denseData_cpu.val = new kernel_type[m*dense_col_num_cpu];
        denseData_cpu.row=m;
        denseData_cpu.col=dense_col_num_cpu;
        denseData_cpu.Ttable=new int[n];
        denseData_cpu.is_use=true;
    } else {
        denseData_cpu.row=m;
        denseData_cpu.col=dense_col_num_cpu;
        denseData_cpu.is_use= false;
    }


    if (dense_col_num_gpu > 0){
        denseData_gpu.val = new kernel_type[m*dense_col_num_gpu];
        denseData_gpu.row=m;
        denseData_gpu.col=dense_col_num_gpu;
        denseData_gpu.Ttable=new int[n];
        denseData_gpu.is_use=true;
    }else {
        denseData_gpu.row=m;
        denseData_gpu.col=dense_col_num_gpu;
        denseData_gpu.is_use=false;
    }


    for (int i=0,num_cpu=0,num_gpu=0;i<n;i++){
        if (densefg[i]==false){
//            denseData_cpu.Ttable[i]=-1;
//            denseData_gpu.Ttable[i]=-1;
        }else{
            if (gpufg[i]==true){
                denseData_gpu.Ttable[i]=num_gpu;
                num_gpu++;
            } else {
                denseData_cpu.Ttable[i]=num_cpu;
                num_cpu++;
            }
        }
    }

    //printf("Malloc Merroy OK\n");


    sparseData_cpu.val_data.clear();
    sparseData_cpu.col_ptr.clear();
    sparseData_cpu.row_ptr.clear();
    sparseData_cpu.row_ptr.push_back(0);
    sparseData_cpu.row=m;
    sparseData_cpu.col=n;


    sparseData_gpu.val_data.clear();
    sparseData_gpu.col_ptr.clear();
    sparseData_gpu.row_ptr.clear();
    sparseData_gpu.row_ptr.push_back(0);
    sparseData_gpu.row=m;
    sparseData_gpu.col=n;

#pragma omp parallel for num_threads(8)
    for (int i=0;i<m;i++){
        int csr_row_begin = csr_row_ptr[i];
        int csr_row_end = csr_row_ptr[i+1];
        for (int j=csr_row_begin;j<csr_row_end;j++) {
            if (densefg[csr_col_ind[j]] == true) {

                if (gpufg[csr_col_ind[j]] == true) {
                    denseData_gpu.val[i * dense_col_num_gpu + denseData_gpu.Ttable[csr_col_ind[j]]] = csr_val[j];
                } else {
                    denseData_cpu.val[i * dense_col_num_cpu + denseData_cpu.Ttable[csr_col_ind[j]]] = csr_val[j];
                }
            }
        }
    }

    if (densecolnum < n)
    for (int i=0;i<m;i++){
        int csr_row_begin = csr_row_ptr[i];
        int csr_row_end = csr_row_ptr[i+1];
        for (int j=csr_row_begin;j<csr_row_end;j++){
            if (densefg[csr_col_ind[j]]==false){
                if (gpufg[csr_col_ind[j]] == true){
                    sparseData_gpu.val_data.push_back(csr_val[j]);
                    sparseData_gpu.col_ptr.push_back(csr_col_ind[j]);
                }else {
                    sparseData_cpu.val_data.push_back(csr_val[j]);
                    sparseData_cpu.col_ptr.push_back(csr_col_ind[j]);
                }
            }
        }

        sparseData_cpu.row_ptr.push_back(sparseData_cpu.val_data.size());
        sparseData_gpu.row_ptr.push_back(sparseData_gpu.val_data.size());

    }

    //printf("Sparse CPU Size : %d\n",sparseData_cpu.val_data.size());
    //printf("Sparse GPU Size : %d\n",sparseData_gpu.val_data.size());



    if (sparseData_cpu.val_data.size()>0) sparseData_cpu.is_use=true;
    else sparseData_cpu.is_use=false;

    if (sparseData_gpu.val_data.size()>0) sparseData_gpu.is_use=true;
    else sparseData_gpu.is_use=false;


    delete[] col_num;
    delete[] densefg;
    delete[] gpufg;
}

void CSRtoDenseandCSR2(const KernelMatrix &k_mat,DenseData &denseData_cpu,SparseData &sparseData_cpu,DenseData &denseData_gpu,SparseData &sparseData_gpu){
    const long long m = k_mat.n_instances_;

    const long long n = k_mat.n_features_;


    const kernel_type * csr_val = k_mat.val_.host_data();
    const int * csr_row_ptr = k_mat.row_ptr_.host_data();
    const int * csr_col_ind = k_mat.col_ind_.host_data();



    Node* col_num = new Node[n+10];

    for (int i=0;i<n;i++) {
        col_num[i].num=0;
        col_num[i].x=i;
    }


    for (int i=0;i<m;i++){
        int csr_row_begin = csr_row_ptr[i];
        int csr_row_end = csr_row_ptr[i+1];
        for (int j=csr_row_begin;j<csr_row_end;j++){
            col_num[csr_col_ind[j]].num++;
        }
    }

    std::sort(col_num,col_num+n,[=](Node a,Node b){
        if (a.num<b.num) return true;
        if (a.num==b.num && a.x < b.x) return true;
        return false;
    });


    /*
     * 考虑一下CSR部分以及DENSE部分的数据量大小问题
     *
     *
     */

    long long denseDataNum = 0;
    long long csrDataNum = k_mat.nnz_;





    long long t_num=0;
    long long a_num=0;

    bool* densefg = new bool[n];
    for (int i=0;i<n;i++) densefg[i]=true;

    bool* gpufg = new bool[n];
    for (int i=0;i<n;i++) gpufg[i]=false;

    int densecolnum=n;


    // 密度从小到大



    long long total_csr=0;

    long long total_dense=m*n;


    for (int i=0;i<n;i++){
        //densecolnum--;
        //densefg[col_num[i].x]= false;


        total_csr+=col_num[i].num;
        total_dense-=m;


        t_num+=col_num[i].num;
        a_num+=m;
        double ans=1.0l*t_num/a_num;
        //printf("MIDU : %lf\n",ans);
        if (ans > 0.02 || 1.0l*col_num[i].num/m > 0.3 ||total_csr * 8 > total_dense){
            break;
        }
        densecolnum--;
        densefg[col_num[i].x]= false;


    }


    //printf("dense col num : %d\n",densecolnum);



    /*
     * 决定数据的CPU和GPU的归属
     */

    long long gpu_memory_size =  8192LL <<20;

    for (int i=n-1;i>=0;i--){
        if (densefg[col_num[i].x]==true){
            gpu_memory_size -= m*sizeof(kernel_type);
        } else {
            gpu_memory_size -= col_num[i].num*sizeof(kernel_type)*2LL;
        }


        if (gpu_memory_size < 0) break;
        //printf("Index : %d |  %lld\n",i,gpu_memory_size);

        gpufg[col_num[i].x]=true;
    }





    int dense_col_num_cpu=0;
    int dense_col_num_gpu=0;

    for (int i=0;i<n;i++){

        if (gpufg[i] == true){
            dense_col_num_gpu += densefg[i] == true ? 1: 0;
        } else {
            dense_col_num_cpu += densefg[i] == true ? 1: 0;
        }

    }
    //printf("Dense Col CPU num : %d\n",dense_col_num_cpu);
    //printf("Dense Col GPU num : %d\n",dense_col_num_gpu);

//    DenseData denseData_cpu;
//    DenseData denseData_gpu;
//
//    SparseData sparseData_cpu;
//    SparseData sparseData_gpu;



    if (dense_col_num_cpu > 0){
        denseData_cpu.val = new kernel_type[m*dense_col_num_cpu];
        denseData_cpu.row=m;
        denseData_cpu.col=dense_col_num_cpu;
        denseData_cpu.Ttable=new int[n];
        denseData_cpu.is_use=true;
    } else {
        denseData_cpu.row=m;
        denseData_cpu.col=dense_col_num_cpu;
        denseData_cpu.is_use= false;
    }


    if (dense_col_num_gpu > 0){
        denseData_gpu.val = new kernel_type[m*dense_col_num_gpu];
        denseData_gpu.row=m;
        denseData_gpu.col=dense_col_num_gpu;
        denseData_gpu.Ttable=new int[n];
        denseData_gpu.is_use=true;
    }else {
        denseData_gpu.row=m;
        denseData_gpu.col=dense_col_num_gpu;
        denseData_gpu.is_use=false;
    }


    for (int i=0,num_cpu=0,num_gpu=0;i<n;i++){
        if (densefg[i]==false){
//            denseData_cpu.Ttable[i]=-1;
//            denseData_gpu.Ttable[i]=-1;
        }else{
            if (gpufg[i]==true){
                denseData_gpu.Ttable[i]=num_gpu;
                num_gpu++;
            } else {
                denseData_cpu.Ttable[i]=num_cpu;
                num_cpu++;
            }
        }
    }

    //printf("Malloc Merroy OK\n");


    sparseData_cpu.val_data.clear();
    sparseData_cpu.col_ptr.clear();
    sparseData_cpu.row_ptr.clear();
    sparseData_cpu.row_ptr.push_back(0);
    sparseData_cpu.row=m;
    sparseData_cpu.col=n;


    sparseData_gpu.val_data.clear();
    sparseData_gpu.col_ptr.clear();
    sparseData_gpu.row_ptr.clear();
    sparseData_gpu.row_ptr.push_back(0);
    sparseData_gpu.row=m;
    sparseData_gpu.col=n;


    for (int i=0;i<m;i++){
        int csr_row_begin = csr_row_ptr[i];
        int csr_row_end = csr_row_ptr[i+1];
        for (int j=csr_row_begin;j<csr_row_end;j++){
            if (densefg[csr_col_ind[j]]==true){

                if (gpufg[csr_col_ind[j]]==true){
                    denseData_gpu.val[i*dense_col_num_gpu+denseData_gpu.Ttable[csr_col_ind[j]]]=csr_val[j];
                }else{
                    denseData_cpu.val[i*dense_col_num_cpu+denseData_cpu.Ttable[csr_col_ind[j]]]=csr_val[j];
                }
            }else{
                if (gpufg[csr_col_ind[j]] == true){
                    sparseData_gpu.val_data.push_back(csr_val[j]);
                    sparseData_gpu.col_ptr.push_back(csr_col_ind[j]);
                }else {
                    sparseData_cpu.val_data.push_back(csr_val[j]);
                    sparseData_cpu.col_ptr.push_back(csr_col_ind[j]);
                }
            }
        }

        sparseData_cpu.row_ptr.push_back(sparseData_cpu.val_data.size());
        sparseData_gpu.row_ptr.push_back(sparseData_gpu.val_data.size());

    }

    //printf("Sparse CPU Size : %d\n",sparseData_cpu.val_data.size());
    //printf("Sparse GPU Size : %d\n",sparseData_gpu.val_data.size());



    if (sparseData_cpu.val_data.size()>0) sparseData_cpu.is_use=true;
    else sparseData_cpu.is_use=false;

    if (sparseData_gpu.val_data.size()>0) sparseData_gpu.is_use=true;
    else sparseData_gpu.is_use=false;


    delete[] col_num;
    delete[] densefg;
    delete[] gpufg;
}


void CSRtoDenseandCSR(const KernelMatrix &k_mat,DenseData &denseData,SparseData &sparseData){
    const int m = k_mat.n_instances_;

    const int n = k_mat.n_features_;


    const kernel_type * csr_val = k_mat.val_.host_data();
    const int * csr_row_ptr = k_mat.row_ptr_.host_data();
    const int * csr_col_ind = k_mat.col_ind_.host_data();



    Node* col_num = new Node[n+10];

    for (int i=0;i<n;i++) {
        col_num[i].num=0;
        col_num[i].x=i;
    }


    for (int i=0;i<m;i++){
        int csr_row_begin = csr_row_ptr[i];
        int csr_row_end = csr_row_ptr[i+1];
        for (int j=csr_row_begin;j<csr_row_end;j++){
            col_num[csr_col_ind[j]].num++;
        }
    }

    std::sort(col_num,col_num+n,[=](Node a,Node b){
        if (a.num>b.num) return true;
        if (a.num==b.num && a.x < b.x) return true;
        return false;
    });


    long long t_num=0;
    long long a_num=0;

    bool* densefg = new bool[n];
    for (int i=0;i<n;i++) densefg[i]=false;

    int densecolnum=0;

    for (int i=0;i<n;i++){
        densecolnum++;
        densefg[col_num[i].x]=true;
        t_num+=col_num[i].num;
        a_num+=m;
        double ans=1.0l*t_num/a_num;
        //printf("MIDU   :  %10lf\n",ans);
        //printf("%8d : %8d   ---  %8d\n",i,col_num[i].num,col_num[i].x);
        if (ans < 0.9){
            break;
        }
    }

    delete col_num;
    denseData.val=new kernel_type[m*densecolnum];
    for (int i=0;i<m*densecolnum;i++) denseData.val[i]=0;
    denseData.Ttable=new int[m];
    //    denseData.Ftable=new int[densecolnum];
    denseData.row=m;
    denseData.col=densecolnum;



    for (int i=0,num=0;i<n;i++){
        if (densefg[i]==false){
            denseData.Ttable[i]=-1;
        }else{
            denseData.Ttable[i]=num;
            //            denseData.Ftable[num]=i;
            num++;
        }
    }


    sparseData.val_data.clear();
    sparseData.col_ptr.clear();
    sparseData.row_ptr.clear();
    sparseData.row_ptr.push_back(0);

    for (int i=0;i<m;i++){
        int csr_row_begin = csr_row_ptr[i];
        int csr_row_end = csr_row_ptr[i+1];
        for (int j=csr_row_begin;j<csr_row_end;j++){
            if (denseData.Ttable[csr_col_ind[j]]>=0){
                denseData.val[i*densecolnum+denseData.Ttable[csr_col_ind[j]]]=csr_val[j];
            }
            if (denseData.Ttable[csr_col_ind[j]]==-1){
                sparseData.val_data.push_back(csr_val[j]);
                sparseData.col_ptr.push_back(csr_col_ind[j]);
            }
        }
        sparseData.row_ptr.push_back(sparseData.val_data.size());
    }
}







void changeCSRtoBCSR(const KernelMatrix &k_mat,SparseData_BCSR &sparseData,SparseData &sparsewithoutdenseData,bool is_use=false){


    const int m = k_mat.n_instances_;

    const int n = k_mat.n_features_;


    const kernel_type * csr_val = k_mat.val_.host_data();
    const int * csr_row_ptr = k_mat.row_ptr_.host_data();
    const int * csr_col_ind = k_mat.col_ind_.host_data();



    Node* col_num = new Node[n+10];

    for (int i=0;i<n;i++) {
        col_num[i].num=0;
        col_num[i].x=i;
    }


    for (int i=0;i<m;i++){
        int csr_row_begin = csr_row_ptr[i];
        int csr_row_end = csr_row_ptr[i+1];
        if (csr_row_end<=csr_row_begin) {
            sparseData.row_ptr.push_back(sparseData.total_num);
            continue;
        }
        for (int j=csr_row_begin+1;j<csr_row_end;j++){
            col_num[csr_col_ind[j]].num++;
        }
    }

    std::sort(col_num,col_num+n,[=](Node a,Node b){
        if (a.num>b.num) return true;
        if (a.num==b.num && a.x < b.x) return true;
        return false;
    });
    long long t_num=0;
    long long a_num=0;
    for (int i=0;i<n;i++){
        //printf("%8d : %8d   ---  %8d\n",i,col_num[i].num,col_num[i].x);
        t_num+=col_num[i].num;
        a_num+=m;
        double ans=1.0l*t_num/a_num;
        //printf("MIDU   :  %10lf\n",ans);


    }

    assert(0);





    //    const int m = k_mat.n_instances_;
    //
    //    const int n = k_mat.n_features_;
    //
    //    const kernel_type * csr_val = k_mat.val_.host_data();
    //    const int * csr_row_ptr = k_mat.row_ptr_.host_data();
    //    const int * csr_col_ind = k_mat.col_ind_.host_data();
    //
    //
    ////    const int m = 4;
    ////
    ////    const kernel_type csr_val[20] = {10,20,30,40,50,60,70,80};
    ////    const int csr_row_ptr[20] =  {0,2,4,7,8};
    ////    const int csr_col_ind[20] ={0,1,1,3,2,3,4,5};
    //
    //
    //    sparseData.val_data.clear();
    //    sparseData.row_ptr.clear();
    //    sparseData.col_begin_ptr.clear();
    //    sparseData.col_end_ptr.clear();
    //
    //    int input_val_num=0;
    //
    //
    //
    //
    //    sparseData.row_ptr.push_back(0);
    //
    //    sparseData.total_num=0;
    //
    //
    //
    //    int GapLength[20] = {2,4,8,16,24,32,48,64,64,64,64,64,64,64,64,64};
    //
    //
    //
    //    for (int i=0;i<m;i++){
    //
    //        int csr_row_begin = csr_row_ptr[i];
    //        int csr_row_end = csr_row_ptr[i+1];
    //        if (csr_row_end<=csr_row_begin) {
    //            sparseData.row_ptr.push_back(sparseData.total_num);
    //            continue;
    //        }
    //
    //        sparseData.col_ptr.push_back(input_val_num++);
    //        sparseData.val_data.push_back(csr_val[csr_row_begin]);
    //        sparseData.col_begin_ptr.push_back(csr_col_ind[csr_row_begin]);
    //
    //        sparseData.total_num++;
    //        int last_begin = csr_col_ind[csr_row_begin];
    //        int last_now   = csr_col_ind[csr_row_begin]+1;
    //        int fg=0;
    //        for (int j=csr_row_begin+1;j<csr_row_end;j++){
    //
    //            if (csr_col_ind[j]-last_now<GapLength[min(fg,15)]){
    //                for (int k=last_now;k<csr_col_ind[j];k++) {
    //                    sparseData.val_data.push_back(0);
    //                    input_val_num++;
    //                }
    //                input_val_num++;
    //                sparseData.val_data.push_back(csr_val[j]);
    //                last_now=csr_col_ind[j]+1;
    //                fg++;
    //            } else {
    //                sparseData.col_end_ptr.push_back(last_now);
    //                sparseData.col_begin_ptr.push_back(csr_col_ind[j]);
    //                sparseData.col_ptr.push_back(input_val_num++);
    //                sparseData.val_data.push_back(csr_val[j]);
    //                last_begin=csr_col_ind[j];
    //                last_now=csr_col_ind[j]+1;
    //                sparseData.total_num++;
    //                fg=0;
    //            }
    //
    //        }
    //        sparseData.col_end_ptr.push_back(last_now);
    //        sparseData.row_ptr.push_back(sparseData.total_num);
    //    }
    //
    //
    //    printf("Sparse val data size : %d\n",sparseData.val_data.size());
    //
    //    int MaxSizeSum=0;
    //
    //    for (int i=0;i<m;i++){
    ////        printf("Row %d ::::::::: \n",i);
    ////        printf("Row Begin : %d |||  End : %d \n",sparseData.row_ptr[i],sparseData.row_ptr[i+1]);
    //        for (int j=sparseData.row_ptr[i];j<sparseData.row_ptr[i+1];j++){
    ////            printf("Col Begin : %d ||||  Col End : %d\n",sparseData.col_begin_ptr[j],sparseData.col_end_ptr[j]);
    //            MaxSizeSum=max(MaxSizeSum,sparseData.col_end_ptr[j]-sparseData.col_begin_ptr[j]+1);
    //        }
    //    }
    //
    //    int *sizeSum = new int[MaxSizeSum+10];
    //
    //    for (int i=1;i<=MaxSizeSum;i++) sizeSum[i]=0;
    //    for (int i=0;i<m;i++){
    ////        printf("Row %d ::::::::: \n",i);
    ////        printf("Row Begin : %d |||  End : %d \n",sparseData.row_ptr[i],sparseData.row_ptr[i+1]);
    //        for (int j=sparseData.row_ptr[i];j<sparseData.row_ptr[i+1];j++){
    ////            printf("Col Begin : %d ||||  Col End : %d\n",sparseData.col_begin_ptr[j],sparseData.col_end_ptr[j]);
    //            sizeSum[sparseData.col_end_ptr[j]-sparseData.col_begin_ptr[j]+1]++;
    //        }
    //    }
    //
    //    for (int i=1;i<=MaxSizeSum;i++){
    //       if (sizeSum[i]>0) printf("Size %d : %d\n",i,sizeSum[i]);
    //    }
    ////    printf("Size > 300 : %d\n",sizeSum[301]);
    //    /*
    //     *
    //     * 删除部分
    //     *
    //     *
    //     */
    //    if (!is_use) return;
    //
    //
    //    sparsewithoutdenseData.val_data.clear();
    //    sparsewithoutdenseData.col_ptr.clear();
    //    sparsewithoutdenseData.row_ptr.clear();
    //    sparsewithoutdenseData.row_ptr.push_back(0);
    //
    //
    //
    //
    //    SparseData_BCSR denseData;
    //    denseData.total_num=0;
    //    denseData.val_data.clear();
    //    denseData.row_ptr.clear();
    //    denseData.col_ptr.clear();
    //    denseData.col_begin_ptr.clear();
    //    denseData.col_end_ptr.clear();
    //
    //
    //    denseData.row_ptr.push_back(denseData.total_num);
    //
    //
    //    for (int i=0;i<m;i++){
    //        int row_begin=sparseData.row_ptr[i];
    //        int row_end=sparseData.row_ptr[i+1];
    //
    //
    //        for (int j=row_begin;j<row_end;j++){
    //            if (sparseData.col_end_ptr[j]-sparseData.col_begin_ptr[j]>BCSR_Block_Size){
    //                denseData.col_ptr.push_back(denseData.val_data.size());
    //                int start=sparseData.col_ptr[j];
    //                for (int k=0;k<sparseData.col_end_ptr[j]-sparseData.col_begin_ptr[j];k++){
    //                    denseData.val_data.push_back(sparseData.val_data[start++]);
    //
    //                }
    //                denseData.col_begin_ptr.push_back(sparseData.col_begin_ptr[j]);
    //                denseData.col_end_ptr.push_back(sparseData.col_end_ptr[j]);
    //                denseData.total_num++;
    //            }else{
    //                int start=sparseData.col_ptr[j];
    //                for (int k=sparseData.col_begin_ptr[j];k<sparseData.col_end_ptr[j];k++){
    ////                   denseData.val_data.push_back(sparseData.val_data[start++]);
    //                    if (sparseData.val_data[start]!=0){
    //                        sparsewithoutdenseData.val_data.push_back(sparseData.val_data[start]);
    //                        sparsewithoutdenseData.col_ptr.push_back(k);
    //                    }
    //                    start++;
    //
    //
    //                }
    //            }
    //        }
    //        denseData.row_ptr.push_back(denseData.total_num);
    //        sparsewithoutdenseData.row_ptr.push_back(sparsewithoutdenseData.val_data.size());
    //
    //
    //
    //    }
    //
    //    sparseData.total_num=denseData.total_num;
    //    sparseData.val_data=denseData.val_data;
    //    sparseData.row_ptr=denseData.row_ptr;
    //    sparseData.col_ptr=denseData.col_ptr;
    //    sparseData.col_begin_ptr=denseData.col_begin_ptr;
    //    sparseData.col_end_ptr=denseData.col_end_ptr;
    //
    //    printf("Dense Data : %d\n",denseData.val_data.size());



}

void dns_csr_mul_CSR_intel_gpu(const KernelMatrix &k_mat,int m, int n, int k, const SyncArray<kernel_type> &dense_mat,SparseData& sparseData, int nnz,
                               kernel_type* &result){
    const kernel_type* dense_data = dense_mat.host_data();
    int* col_data = sparseData.col_ptr.data();
    int* row_index_data = sparseData.row_ptr.data();
    kernel_type * sparse_data = sparseData.val_data.data();
    kernel_type * ref_data = result;
#pragma omp parallel for num_threads(64)
    for (int i=0;i<m;i++){
        int row_begin=row_index_data[i];
        int row_end=row_index_data[i+1];
        for (int j=0;j<n;j++){
            kernel_type sum=0;
            for (int kk=row_begin;kk<row_end;kk++){
                sum+=sparse_data[kk]*dense_data[col_data[kk]+j*k];
            }

            ref_data[i+j*m]=sum;
        }
    }
}

void dns_csr_mul_CSR_MKL_intel_gpu(const KernelMatrix &k_mat,int m, int n, int k, const SyncArray<kernel_type> &dense_mat,SparseData& sparseData, int nnz,
                                   kernel_type* &result){
    const MKL_INT ldx = k;
    const MKL_INT ldy = m;
    struct matrix_descr descrA;
    sparse_matrix_t csrA;
    kernel_type *values = sparseData.val_data.data();
    // Create matrix descriptor
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    static MKL_INT *columns;
    static MKL_INT *row_index;
    static int first_tag = 1;
    MKL_INT i;
    if(first_tag){
        columns = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * sparseData.val_data.size(), 64);
        row_index = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (m + 1), 64);
#pragma omp parallel for num_threads(64)
        for (i = 0; i < sparseData.val_data.size(); i++) {
            columns[i] = sparseData.col_ptr[i];
        }
#pragma omp parallel for num_threads(64)
        for (i = 0; i < m + 1; i++) {
            row_index[i] = sparseData.row_ptr[i];
        }
    }
    sparse_layout_t layout = SPARSE_LAYOUT_COLUMN_MAJOR;
    sparse_status_t ie_status;
    // Create handle with matrix stored in CSR format
    ie_status = mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO,
                                        m, // number of rows
                                        k, // number of cols
                                        row_index, row_index + 1, columns, values);
    ie_status = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descrA,
                                layout, dense_mat.host_data(), n, ldx, 0.0, result, ldy);
    first_tag = 0;
}




void dns_csr_mul_BCSR_intel_gpu(const KernelMatrix &k_mat,int m, int n, int k, const SyncArray<kernel_type> &dense_mat,SparseData_BCSR& sparseDataBcsr, int nnz,
                                SyncArray<kernel_type> &result){
    const kernel_type* dense_data = dense_mat.host_data();
    int* col_begin_data = sparseDataBcsr.col_begin_ptr.data();
    int* col_end_data = sparseDataBcsr.col_end_ptr.data();
    int* col_data = sparseDataBcsr.col_ptr.data();
    int* row_index_data = sparseDataBcsr.row_ptr.data();
    kernel_type * sparse_data = sparseDataBcsr.val_data.data();
    kernel_type * ref_data = result.host_data();
#pragma omp parallel for num_threads(64)
    for (int i=0;i<m;i++){
        int row_begin=row_index_data[i];
        int row_end=row_index_data[i+1];
        for (int j=0;j<n;j++){
            kernel_type sum=0;

            for (int kk=row_begin;kk<row_end;kk++){
                int start=col_data[kk];
                for (int jj=col_begin_data[kk];jj<col_end_data[kk];jj++)
                    sum+=sparse_data[start+jj-col_begin_data[kk]]*dense_data[jj+j*k];
            }

            ref_data[i+j*m]=sum;
        }
    }
}




void dns_csr_mul_BCSR_with_RBF_intel_gpu(const KernelMatrix &k_mat,int m, int n, int k, const SyncArray<kernel_type> &dense_mat,SparseData_BCSR& sparseDataBcsr, int nnz,const SyncArray<int> &self_dot0_idx,
                                         SyncArray<kernel_type> &result){
    const kernel_type* dense_data = dense_mat.host_data();
    int* col_begin_data = sparseDataBcsr.col_begin_ptr.data();
    int* col_end_data = sparseDataBcsr.col_end_ptr.data();
    int* col_data = sparseDataBcsr.col_ptr.data();
    int* row_index_data = sparseDataBcsr.row_ptr.data();
    kernel_type * sparse_data = sparseDataBcsr.val_data.data();


    const int *self_dot0_idx_data = self_dot0_idx.host_data();
    const kernel_type *self_dot1_data = k_mat.self_dot_.host_data();
    kernel_type gamma=k_mat.param.gamma;

    kernel_type* result_data = new kernel_type[m*n];
    kernel_type * ref_data = result.host_data();
#pragma omp parallel for num_threads(64)
    for (int i=0;i<m;i++){
        int row_begin=row_index_data[i];
        int row_end=row_index_data[i+1];
        for (int j=0;j<n;j++){
            kernel_type sum=0;

            for (int kk=row_begin;kk<row_end;kk++){
                int start=col_data[kk];
                for (int jj=col_begin_data[kk];jj<col_end_data[kk];jj++)
                    sum+=sparse_data[start+jj-col_begin_data[kk]]*dense_data[jj+j*k];
            }

            ref_data[i+j*m]=expf(-(self_dot1_data[self_dot0_idx_data[j]] + self_dot1_data[i] - sum*2)*gamma);
        }
    }
}


void dns_csr_mul_with_RBF_intel_gpu(const KernelMatrix &k_mat,int m, int n, int k, const SyncArray<kernel_type> &dense_mat, const SyncArray<kernel_type> &csr_val,
                                    const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz,const SyncArray<int> &self_dot0_idx,
                                    SyncArray<kernel_type> &result) {
    const kernel_type* dense_data = dense_mat.host_data();
    const int* col_index_data = csr_col_ind.host_data();
    const int* row_index_data = csr_row_ptr.host_data();
    const kernel_type * sparse_data = csr_val.host_data();


    const int *self_dot0_idx_data = self_dot0_idx.host_data();
    const kernel_type *self_dot1_data = k_mat.self_dot_.host_data();
    kernel_type gamma=k_mat.param.gamma;


    //    for (int i = 0; i < n; i++) {
    //        for (int j = 0; j < m; ++j) {
    //            dot_product_data[i * m + j] = expf(
    //                    -(self_dot1_data[self_dot0_idx_data[i]] + self_dot1_data[j] - dot_product_data[i * m + j] * 2) *
    //                    gamma);
    //        }
    //    }





    kernel_type* result_data = new kernel_type[m*n];
    kernel_type * ref_data = result.host_data();
#pragma omp parallel for num_threads(64)
    for (int i=0;i<m;i++){
        int row_begin=row_index_data[i];
        int row_end=row_index_data[i+1];
        for (int j=0;j<n;j++){
            kernel_type sum=0;

            for (int kk=row_begin;kk<row_end;kk++){
                sum+=sparse_data[kk]*dense_data[col_index_data[kk]+j*k];
            }

            ref_data[i+j*m]=expf(-(self_dot1_data[self_dot0_idx_data[j]] + self_dot1_data[i] - sum*2)*gamma);
        }
    }
}

void dns_csr_mul_with_poly_intel_gpu(const KernelMatrix &k_mat,int m, int n, int k, const SyncArray<kernel_type> &dense_mat, const SyncArray<kernel_type> &csr_val,
                                     const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz,
                                     SyncArray<kernel_type> &result) {
    const kernel_type* dense_data = dense_mat.host_data();
    const int* col_index_data = csr_col_ind.host_data();
    const int* row_index_data = csr_row_ptr.host_data();
    const kernel_type * sparse_data = csr_val.host_data();




    //    for (int i = 0; i < n; i++) {
    //        for (int j = 0; j < m; ++j) {
    //            dot_product_data[i * m + j] = expf(
    //                    -(self_dot1_data[self_dot0_idx_data[i]] + self_dot1_data[j] - dot_product_data[i * m + j] * 2) *
    //                    gamma);
    //        }
    //    }


    const kernel_type gamma = k_mat.param.gamma;
    const kernel_type coef0 = k_mat.param.coef0;
    const int degree = k_mat.param.degree;



    kernel_type* result_data = new kernel_type[m*n];
    kernel_type * ref_data = result.host_data();
#pragma omp parallel for num_threads(64)
    for (int i=0;i<m;i++){
        int row_begin=row_index_data[i];
        int row_end=row_index_data[i+1];
        for (int j=0;j<n;j++){
            kernel_type sum=0;
            for (int kk=row_begin;kk<row_end;kk++){
                sum+=sparse_data[kk]*dense_data[col_index_data[kk]+j*k];
            }

            ref_data[i+j*m]=powf(gamma*sum + coef0, degree);
        }
    }
}

void dns_csr_mul_with_sigmoid_intel_gpu(const KernelMatrix &k_mat,int m, int n, int k, const SyncArray<kernel_type> &dense_mat, const SyncArray<kernel_type> &csr_val,
                                        const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz,
                                        SyncArray<kernel_type> &result) {
    const kernel_type* dense_data = dense_mat.host_data();
    const int* col_index_data = csr_col_ind.host_data();
    const int* row_index_data = csr_row_ptr.host_data();
    const kernel_type * sparse_data = csr_val.host_data();




    //    for (int i = 0; i < n; i++) {
    //        for (int j = 0; j < m; ++j) {
    //            dot_product_data[i * m + j] = expf(
    //                    -(self_dot1_data[self_dot0_idx_data[i]] + self_dot1_data[j] - dot_product_data[i * m + j] * 2) *
    //                    gamma);
    //        }
    //    }


    const kernel_type gamma = k_mat.param.gamma;
    const kernel_type coef0 = k_mat.param.coef0;



    kernel_type* result_data = new kernel_type[m*n];
    kernel_type * ref_data = result.host_data();
#pragma omp parallel for num_threads(64)
    for (int i=0;i<m;i++){
        int row_begin=row_index_data[i];
        int row_end=row_index_data[i+1];
        for (int j=0;j<n;j++){
            kernel_type sum=0;
            for (int kk=row_begin;kk<row_end;kk++){
                sum+=sparse_data[kk]*dense_data[col_index_data[kk]+j*k];
            }

            ref_data[i+j*m]=tanhf(gamma * sum + coef0);
        }
    }
}


void dns_csr_mul_intel_gpu(const KernelMatrix &k_mat,int m, int n, int k, const SyncArray<kernel_type> &dense_mat, const SyncArray<kernel_type> &csr_val,
                           const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz,
                           SyncArray<kernel_type> &result){
    //CHECK_EQ(dense_mat.size(), n * k_mat.n_features_) << "dense matrix features doesn't match";
    const kernel_type* dense_data = dense_mat.host_data();
    const int* col_index_data = csr_col_ind.host_data();
    const int* row_index_data = csr_row_ptr.host_data();
    const kernel_type * sparse_data = csr_val.host_data();

    kernel_type* result_data = new kernel_type[m*n];
    kernel_type * ref_data = result.host_data();
#pragma omp parallel for num_threads(64)
    for (int i=0;i<m;i++){
        int row_begin=row_index_data[i];
        int row_end=row_index_data[i+1];
        for (int j=0;j<n;j++){
            kernel_type sum=0;
            for (int kk=row_begin;kk<row_end;kk++){
                sum+=sparse_data[kk]*dense_data[col_index_data[kk]+j*k];
            }
            ref_data[i+j*m]=sum;
        }
    }
}

void get_working_set_ins_intel_gpu(const KernelMatrix &k_mat,const SyncArray<kernel_type> &val, const SyncArray<int> &col_ind, const SyncArray<int> &row_ptr,
                                   const SyncArray<int> &data_row_idx, SyncArray<kernel_type> &data_rows, int m, int n){
    const int *data_row_idx_data = data_row_idx.host_data();
    kernel_type *data_rows_data = data_rows.host_data();
    const int *row_ptr_data = row_ptr.host_data();
    const int *col_ind_data = col_ind.host_data();
    const kernel_type *val_data = val.host_data();
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < m; i++) {
        int row = data_row_idx_data[i];
        for (int j = row_ptr_data[row]; j < row_ptr_data[row + 1]; ++j) {
            int col = col_ind_data[j];
            data_rows_data[i * n + col] = val_data[j]; //row major
        }
    }
}
/*
 * KernelMatrix::dns_csr_mul(const SyncArray<kernel_type> &dense_mat, int n_rows, SyncArray<kernel_type> &result) const {
 CHECK_EQ(dense_mat.size(), n_rows * n_features_) << "dense matrix features doesn't match";
 svm_kernel::dns_csr_mul(n_instances_, n_rows, n_features_, dense_mat, val_, row_ptr_, col_ind_, nnz_, result);
 }
 */

void MergeResultIPP(kernel_type* dot_product,kernel_type* dot_product_sparse,int size){
//      ippsAdd_32f_A24(dot_product, dot_product_sparse, dot_product, size);
//}
  const int th_num = 16;
  int size_per_th = (size + th_num - 1) / th_num;
#pragma omp parallel for num_threads(th_num)
  for(int i = 0; i < th_num; i++)
  {
        int my_rank = omp_get_thread_num();
        int start = my_rank * size_per_th;
        int end = ((my_rank + 1) * size_per_th) > size ? size : (my_rank + 1) * size_per_th;
        int len = end > start ? end - start : 0;
        ippsAdd_32f_A24( dot_product + start, dot_product_sparse + start, dot_product + start, len );

  }
}

void MergeResult(kernel_type* dot_product,kernel_type* dot_product_sparse,int size){
#pragma omp parallel for num_threads(16)
    for(int i=0;i<size;i++)
        dot_product[i]+=dot_product_sparse[i];
}

void MergeResult(kernel_type* A,kernel_type* B,kernel_type *C,int size){
#pragma omp parallel for num_threads(16)
    for(int i=0;i<size;i++)
        A[i]=B[i]+C[i];
}


void MergeTransResult(kernel_type* dot_product,kernel_type* dot_product_sparse,int size){
#pragma omp parallel for num_threads(16)
    for(int i=0;i<size;i++)
        dot_product[i]+=dot_product_sparse[i];
}



void get_dot_product_dns_csr_intel_gpu(const KernelMatrix &k_mat,SparseData_BCSR& sparseDataBcsr,SparseData& sparsewithoutdenseData,const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product , SyncArray<kernel_type> &data_rows){
    //    SyncArray<kernel_type> data_rows(idx.size() * k_mat.n_features_);
    //    data_rows.mem_set(0);
    memset(data_rows.host_data(),0x00,sizeof(kernel_type)*idx.size()*k_mat.n_features_);
    get_working_set_ins_intel_gpu(k_mat,k_mat.val_, k_mat.col_ind_, k_mat.row_ptr_, idx, data_rows, idx.size(), k_mat.n_features_);
    TDEF(dense)
    TSTART(dense)
    //std::thread t1([&]() {
    dns_csr_mul_BCSR_intel_gpu(k_mat,k_mat.n_instances_,idx.size(),k_mat.n_features_,data_rows,sparseDataBcsr,k_mat.nnz_,dot_product);
    //});
    TEND(dense)
    TPRINT(dense,"$#%dense dense mul time  : " )
    TDEF(sparse)
    TSTART(sparse)
    static kernel_type *dot_product_sparse;
    static bool fg=false;
    if (!fg) {
        fg=true;
        dot_product_sparse=new kernel_type[k_mat.n_instances_*idx.size()];

    }
    //std::thread t2([&]() {
    dns_csr_mul_CSR_MKL_intel_gpu(k_mat,k_mat.n_instances_,idx.size(),k_mat.n_features_,data_rows,sparsewithoutdenseData,k_mat.nnz_,dot_product_sparse);
    //});
    //t1.join();
    //t2.join();
    TEND(sparse)
    TPRINT(sparse,"$#%sparse dense mul time : ")

    TDEF(merge)
    TSTART(merge)
    MergeResult(dot_product.host_data(),dot_product_sparse,k_mat.n_instances_*idx.size());
    TEND(merge)
    TPRINT(merge,"$#%merge  time : ")
    //delete dot_product_sparse;



}

void RBF_kernel_intel_gpu(const SyncArray<int> &self_dot0_idx, const SyncArray<kernel_type> &self_dot1,
                          SyncArray<kernel_type> &dot_product, int m,
                          int n, kernel_type gamma){
    kernel_type *dot_product_data = dot_product.host_data();
    const int *self_dot0_idx_data = self_dot0_idx.host_data();
    const kernel_type *self_dot1_data = self_dot1.host_data();
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; ++j) {
            dot_product_data[i * n + j] = expf(
                    -(self_dot1_data[self_dot0_idx_data[i]] + self_dot1_data[j] - dot_product_data[i * n + j] * 2) *
                    gamma);
        }
    }
}

void poly_kernel_intel_gpu(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int degree, int mn){
    kernel_type *dot_product_data = dot_product.host_data();
#pragma omp parallel for schedule(guided)
    for (int idx = 0; idx < mn; idx++) {
        dot_product_data[idx] = powf(gamma * dot_product_data[idx] + coef0, degree);
    }
}

void sigmoid_kernel_intel_gpu(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int mn){
    kernel_type *dot_product_data = dot_product.host_data();
#pragma omp parallel for schedule(guided)
    for (int idx = 0; idx < mn; idx++) {
        dot_product_data[idx] = tanhf(gamma * dot_product_data[idx] + coef0);
    }
}




void get_rows_intel_gpu(const KernelMatrix &k_mat,SparseData_BCSR& sparseDataBcsr,SparseData& sparsewithoutdenseData,const SyncArray<int> &idx,
                        SyncArray<kernel_type> &kernel_rows,SyncArray<kernel_type> &data_rows){


    CHECK_GE(kernel_rows.size(), idx.size() * k_mat.n_instances_) << "kernel_rows memory is too small";
#ifdef USE_CUDA
    get_dot_product_dns_csr(idx, kernel_rows);
#else

    if(k_mat.n_features_ < 1000000)
        get_dot_product_dns_csr_intel_gpu(k_mat,sparseDataBcsr,sparsewithoutdenseData,idx, kernel_rows,data_rows);
    //    else
    //        get_dot_product_csr_csr(idx, kernel_rows);
    //    get_dot_product_dns_dns(idx, kernel_rows);
#endif
    switch (k_mat.param.kernel_type) {
        case SvmParam::RBF:
        case SvmParam::PRECOMPUTED://precomputed uses rbf as default
            //In this;
            RBF_kernel_intel_gpu(idx, k_mat.self_dot_, kernel_rows, idx.size(), k_mat.n_instances_, k_mat.param.gamma);

            //printf("RBF OR PRECOMPUTED\n");
            break;
        case SvmParam::LINEAR:
            //do nothing
            //printf("LINEAR\n");
            break;
        case SvmParam::POLY:
            poly_kernel_intel_gpu(kernel_rows, k_mat.param.gamma, k_mat.param.coef0, k_mat.param.degree, kernel_rows.size());
            //printf("POLY\n");
            break;
        case SvmParam::SIGMOID:
            sigmoid_kernel_intel_gpu(kernel_rows, k_mat.param.gamma, k_mat.param.coef0, kernel_rows.size());
            //printf("SIGMOID\n");

            break;
    }


}




/*
 *
 * 基于列密度的划分策略
 *
 */


void RBF_kernel_ColDensity(const SyncArray<int> &self_dot0_idx, const SyncArray<kernel_type> &self_dot1,
                           SyncArray<kernel_type> &dot_product, int m,
                           int n, kernel_type gamma){
    kernel_type *dot_product_data = dot_product.host_data();
    const int *self_dot0_idx_data = self_dot0_idx.host_data();
    const kernel_type *self_dot1_data = self_dot1.host_data();
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; ++j) {
            dot_product_data[i * n + j] = expf(
                    -(self_dot1_data[self_dot0_idx_data[i]] + self_dot1_data[j] - dot_product_data[i * n + j] * 2) *
                    gamma);
        }
    }
}

void poly_kernel_ColDensity(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int degree, int mn){
    kernel_type *dot_product_data = dot_product.host_data();
#pragma omp parallel for schedule(guided)
    for (int idx = 0; idx < mn; idx++) {
        dot_product_data[idx] = powf(gamma * dot_product_data[idx] + coef0, degree);
    }
}

void sigmoid_kernel_ColDensity(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int mn){
    kernel_type *dot_product_data = dot_product.host_data();
#pragma omp parallel for schedule(guided)
    for (int idx = 0; idx < mn; idx++) {
        dot_product_data[idx] = tanhf(gamma * dot_product_data[idx] + coef0);
    }
}

void dns_csr_mul_CSR_MKL_ColDensity_cpu(const KernelMatrix &k_mat,int m, int n, int k, kernel_type *&dense_mat,SparseData& sparseData, int nnz,
                                    kernel_type* result, float bbeta){
    const MKL_INT ldx = k;
    const MKL_INT ldy = m;
    struct matrix_descr descrA;
    sparse_matrix_t csrA;
    kernel_type *values = sparseData.val_data.data();
    // Create matrix descriptor
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    static MKL_INT *columns;
    static MKL_INT *row_index;
    static int first_tag = 1;
    MKL_INT i;
    if(first_tag){
        columns = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * sparseData.val_data.size(), 64);
        row_index = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (m + 1), 64);
#pragma omp parallel for num_threads(64)
        for (i = 0; i < sparseData.val_data.size(); i++) {
            columns[i] = sparseData.col_ptr[i];
        }
#pragma omp parallel for num_threads(64)
        for (i = 0; i < m + 1; i++) {
            row_index[i] = sparseData.row_ptr[i];
        }
    }
    sparse_layout_t layout = SPARSE_LAYOUT_COLUMN_MAJOR;
    sparse_status_t ie_status;
    // Create handle with matrix stored in CSR format
    ie_status = mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO,
                                        m, // number of rows
                                        k, // number of cols
                                        row_index, row_index + 1, columns, values);
    ie_status = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descrA,
                                layout, dense_mat, n, ldx, bbeta, result, ldy);
    first_tag = 0;
}


void dns_csr_mul_CSR_MKL_ColDensity_cpu(const KernelMatrix &k_mat,int m, int n, int k, kernel_type *&dense_mat,SparseData& sparseData, int nnz,
                                    kernel_type* &result){
    const MKL_INT ldx = k;
    const MKL_INT ldy = m;
    struct matrix_descr descrA;
    sparse_matrix_t csrA;
    kernel_type *values = sparseData.val_data.data();
    // Create matrix descriptor
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    static MKL_INT *columns;
    static MKL_INT *row_index;
    static int first_tag = 1;
    MKL_INT i;
    if(first_tag){
        columns = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * sparseData.val_data.size(), 64);
        row_index = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (m + 1), 64);
#pragma omp parallel for num_threads(64)
        for (i = 0; i < sparseData.val_data.size(); i++) {
            columns[i] = sparseData.col_ptr[i];
        }
#pragma omp parallel for num_threads(64)
        for (i = 0; i < m + 1; i++) {
            row_index[i] = sparseData.row_ptr[i];
        }
    }
    sparse_layout_t layout = SPARSE_LAYOUT_COLUMN_MAJOR;
    sparse_status_t ie_status;
    // Create handle with matrix stored in CSR format
    ie_status = mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO,
                                        m, // number of rows
                                        k, // number of cols
                                        row_index, row_index + 1, columns, values);
    ie_status = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descrA,
                                layout, dense_mat, n, ldx, 0.0, result, ldy);
    first_tag = 0;
}

void dns_csr_mul_Dense_MKL_ColDensity_gpu(int mm, int nn, int kk, kernel_type *B_host,  DenseData& denseData, kernel_type *result, float bbeta){
	//printf("----- %d %d %d\n", mm, nn, kk);
    //double tt0 = GetTime();
	long long n = nn;
	long long m = mm;
	long long k = kk;
	kernel_type *C_host = (kernel_type*)result;
	kernel_type *A_host = denseData.val;


	static bool inited = false;
	static sycl::queue q0;
	static kernel_type *A0;
	static kernel_type *B0;
	static kernel_type *C0;


	if(!inited)
	{
        //double t0 = GetTime();
		TDEF(init_cost1)
		TSTART(init_cost1)
		inited = true;

        //double t1 = GetTime();
		std::vector<sycl::queue> qs;
		auto platforms = sycl::platform::get_platforms();
		for (auto & p : platforms)
		{
			if(p.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
				continue;
			auto devices = p.get_devices();
			for(auto &d : devices)
			{
				if(d.is_gpu())
					qs.push_back(sycl::queue(d));
			}
		}
		TEND(init_cost1)
		TPRINT(init_cost1, "$#%init dense cost1 ")


		TDEF(init_cost2)
		TSTART(init_cost2)
		assert(qs.size() == 2);
		q0 = qs[0];
        //printf("qs cost %lf\n", GetTime() - t1);

		A0 = sycl::malloc_device<kernel_type>(m * k, q0);

		B0 = sycl::malloc_device<kernel_type>(n * k, q0);

		C0 = sycl::malloc_device<kernel_type>(n * m, q0);

		auto e0 = q0.memcpy(A0, A_host, sizeof(kernel_type) * m * k);

		oneapi::mkl::blas::row_major::imatcopy_batch(q0, oneapi::mkl::transpose::T, m, k, 1, A0, k, m, m * k, 1, {e0});

        q0.memset(B0, 0, sizeof(kernel_type) * n * k);
        q0.memset(C0, 0, sizeof(kernel_type) * n * m);
		q0.wait();

		//std::cout << "Init finish!!" << std::endl;
		TEND(init_cost2)
		TPRINT(init_cost2, "$#%init dense cost2 ")
        //printf("init dense cost %lf\n", GetTime() - t0);
	}

	auto e0 = q0.memcpy(B0, B_host, sizeof(kernel_type) * n * k);
	oneapi::mkl::blas::row_major::gemm(q0, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, n, m, k, 1, B0, k, A0, m, bbeta, C0, m, {e0}).wait();
	q0.memcpy(C_host, C0, sizeof(kernel_type) * n * m).wait();
    //static int cnttt = 0;
    //if(cnttt == 0){
    //    printf("dense 1 round cost %lf\n", GetTime() - tt0);
    //}
    //cnttt++;

}

//
//void dns_csr_mul_Dense_MKL_ColDensity_gpu(int mm, int nn, int kk, kernel_type *B_host,  DenseData& denseData, kernel_type *result){
//	//printf("----- %d %d %d\n", mm, nn, kk);
//	long long n = nn;
//	long long m = mm;
//	long long k = kk;
//	kernel_type *C_host = (kernel_type*)result;
//	kernel_type *A_host = denseData.val;
//
//
//	static bool inited = false;
//	static sycl::queue q0;
//	static kernel_type *A0;
//	static kernel_type *B0;
//	static kernel_type *C0;
//
//
//	if(!inited)
//	{
//		TDEF(init_cost1)
//		TSTART(init_cost1)
//		inited = true;
//
//		std::vector<sycl::queue> qs;
//		auto platforms = sycl::platform::get_platforms();
//		for (auto & p : platforms)
//		{
//			if(p.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
//				continue;
//			auto devices = p.get_devices();
//			for(auto &d : devices)
//			{
//				if(d.is_gpu())
//					qs.push_back(sycl::queue(d));
//			}
//		}
//		TEND(init_cost1)
//		TPRINT(init_cost1, "$#%init dense cost1 ")
//
//
//		TDEF(init_cost2)
//		TSTART(init_cost2)
//		assert(qs.size() == 2);
//		q0 = qs[0];
//
//		A0 = sycl::malloc_device<kernel_type>(m * k, q0);
//
//		B0 = sycl::malloc_device<kernel_type>(n * k, q0);
//
//		C0 = sycl::malloc_device<kernel_type>(n * m, q0);
//
//		auto e0 = q0.memcpy(A0, A_host, sizeof(kernel_type) * m * k);
//
//		oneapi::mkl::blas::row_major::imatcopy_batch(q0, oneapi::mkl::transpose::T, m, k, 1, A0, k, m, m * k, 1, {e0});
//
//		q0.wait();
//
//		std::cout << "Init finish!!" << std::endl;
//		TEND(init_cost2)
//		TPRINT(init_cost2, "$#%init dense cost2 ")
//
//
//	}
//
//	auto e0 = q0.memcpy(B0, B_host, sizeof(kernel_type) * n * k);
//	oneapi::mkl::blas::row_major::gemm(q0, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, n, m, k, 1, B0, k, A0, m, 0, C0, m, {e0}).wait();
//	q0.memcpy(C_host, C0, sizeof(kernel_type) * n * m).wait();
//
//}
//



void dns_csr_mul_Dense_MKL_ColDensity_two_gpu(int mm, int nn, int kk, kernel_type *B_host,  DenseData& denseData, kernel_type *result, float bbeta){
    //printf("----- %d %d %d\n", mm, nn, kk);
    long long n = nn;
    long long m = mm;
    long long k = kk;
    kernel_type *C_host = (kernel_type*)result;
    kernel_type *A_host = denseData.val;



    static bool inited = false;
    static sycl::queue q0;
    static kernel_type *A0;
    static kernel_type *B0;
    static kernel_type *C0;

    static sycl::queue q1;
    static kernel_type *A1;
    static kernel_type *B1;
    static kernel_type *C1;

    static kernel_type *C0_tmp;
    static kernel_type *C1_tmp;

    if(!inited)
    {
        inited = true;

        std::vector<sycl::queue> qs;
        auto platforms = sycl::platform::get_platforms();
        for (auto & p : platforms)
        {
            if(p.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
                continue;
            auto devices = p.get_devices();
            for(auto &d : devices)
            {
                if(d.is_gpu())
                    qs.push_back(sycl::queue(d));
            }
        }

        assert(qs.size() == 2);
        q0 = qs[0];
        q1 = qs[1];

        A0 = sycl::malloc_device<kernel_type>((m / 2) * k, q0);
        A1 = sycl::malloc_device<kernel_type>(((m + 1) / 2) * k, q1);

        B0 = sycl::malloc_device<kernel_type>(n * k, q0);
        B1 = sycl::malloc_device<kernel_type>(n * k, q1);

        C0 = sycl::malloc_device<kernel_type>(n * (m / 2), q0);
        C1 = sycl::malloc_device<kernel_type>(n * ((m + 1) / 2), q1);

        auto e0 = q0.memcpy(A0, A_host, sizeof(kernel_type) * (m / 2) * k);
        auto e1 = q1.memcpy(A1, A_host + (m / 2) * k, sizeof(kernel_type) * ((m + 1) / 2) * k);

        oneapi::mkl::blas::row_major::imatcopy_batch(q0, oneapi::mkl::transpose::T, m / 2, k, 1, A0, k, m / 2, (m / 2) * k, 1, {e0});
        oneapi::mkl::blas::row_major::imatcopy_batch(q1, oneapi::mkl::transpose::T, (m + 1) / 2, k, 1, A1, k, (m + 1) / 2, ((m + 1) / 2) * k, 1, {e1});

        q0.wait();
        q1.wait();

        C0_tmp = new kernel_type[n * (m / 2)];
        C1_tmp = new kernel_type[n * ((m + 1) / 2)];

        std::cout << "Init finish!!" << std::endl;
    }

    std::thread t0([&]()
                   {
                       auto e0 = q0.memcpy(B0, B_host, sizeof(kernel_type) * n * k);
                       oneapi::mkl::blas::row_major::gemm(q0, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, n, m / 2, k, 1, B0, k, A0, m / 2, 0, C0, m / 2, {e0}).wait();

                       q0.memcpy(C0_tmp, C0, sizeof(kernel_type) * n * (m / 2)).wait();
                   });

    std::thread t1([&]()
                   {
                       auto e1 = q1.memcpy(B1, B_host, sizeof(kernel_type) * n * k);
                       oneapi::mkl::blas::row_major::gemm(q1, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, n, (m + 1) / 2, k, 1, B1, k, A1, (m + 1) / 2, 0, C1, (m + 1) / 2, {e1}).wait();

                       q1.memcpy(C1_tmp, C1, sizeof(kernel_type) * n * ((m + 1) / 2)).wait();
                   });

    t0.join();
    t1.join();

    //for(int i = 0; i < n * (m / 2); ++i)
    //    std::cout << C0_tmp[i] << ' ';
    //std::cout << std::endl;

    //for(int i = 0; i < n * ((m + 1) / 2); ++i)
    //    std::cout << C1_tmp[i] << ' ';
    //std::cout << std::endl;

    for(long long i = 0; i < n; ++i)
    {
        memcpy(C_host + i * m, C0_tmp + i * (m / 2), sizeof(kernel_type) * (m / 2));
        memcpy(C_host + i * m + (m / 2), C1_tmp + i * ((m + 1) / 2), sizeof(kernel_type) * ((m + 1) / 2));
    }

}


void dns_csr_mul_CSR_MKL_ColDensity_two_gpu(int mm, int nn, int kk, kernel_type *dense_mat,SparseData& sparseData, int nnzz, kernel_type* &result){
    //printf(" 1 :%d %d %d\n", mm, kk, nn);
    long long n = nn;
    long long m = mm;
    long long k = kk;
    long long nnz = sparseData.val_data.size();
    assert(nnz == nnzz);
    assert(nnz == sparseData.row_ptr[m + 1]);
    kernel_type *values = (kernel_type*)sparseData.val_data.data();
    int *colums = (int*)sparseData.col_ptr.data();
    int *row_ptr = (int*)sparseData.row_ptr.data();
    kernel_type *x = (kernel_type*)dense_mat;
    kernel_type *C_host = (kernel_type*)result;
    //static sycl::queue q(sycl::device{sycl::gpu_selector()});
    //static sycl::queue q(sycl::device{sycl::cpu_selector()});
    static kernel_type *A0_val;
    static kernel_type *B0_val;
    static kernel_type *C0_val;
    static kernel_type *D0_val;
    static int *A0_row_ptr;
    static int *A0_col;
    static oneapi::mkl::sparse::matrix_handle_t A0_h;

    static kernel_type *A1_val;
    static kernel_type *B1_val;
    static kernel_type *C1_val;
    static kernel_type *D1_val;
    static int *A1_row_ptr;
    static int *A1_col;
    static oneapi::mkl::sparse::matrix_handle_t A1_h;

    static sycl::queue q0;
    static sycl::queue q1;
    static int first_tag = 1;
    static int m0;
    static int m1;
    static int nnz0;
    static int nnz1;
    if(first_tag){

        std::vector<sycl::queue> qs;
        auto platforms = sycl::platform::get_platforms();
        for (auto & p : platforms)
        {
            if(p.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
                continue;
            auto devices = p.get_devices();
            for(auto &d : devices)
            {
                if(d.is_gpu())
                    qs.push_back(sycl::queue(d));
            }
        }

        assert(qs.size() == 2);
        q0 = qs[0];
        q1 = qs[1];


        int half_nnz = nnz / 2;
        for(int i = 0; i < m + 1; i++){
            if(row_ptr[i] >= half_nnz){
                m0 = i;
                m1 = m - m0;
                nnz0 = row_ptr[i];
                nnz1 = nnz - nnz0;
                break;
            }
        }
        A0_row_ptr = sycl::malloc_shared<int>(m0 + 1, q0);
        A0_col = sycl::malloc_shared<int>(nnz0, q0);
        A0_val = sycl::malloc_shared<kernel_type>(nnz0, q0);
        B0_val = sycl::malloc_shared<kernel_type>(k * n, q0);
        C0_val = sycl::malloc_shared<kernel_type>(m0 * n, q0);
        D0_val = new kernel_type[m0 * n];

        q0.memcpy(A0_row_ptr, row_ptr, sizeof(int) * (m0 + 1)).wait();
        q0.memcpy(A0_col, colums, sizeof(int) * nnz0).wait();
        q0.memcpy(A0_val, values, sizeof(kernel_type) * nnz0).wait();
        oneapi::mkl::sparse::init_matrix_handle(&(A0_h));
        oneapi::mkl::sparse::set_csr_data(A0_h, m0, k, oneapi::mkl::index_base::zero, A0_row_ptr, A0_col, A0_val);
        q0.wait();


        A1_row_ptr = sycl::malloc_shared<int>(m1 + 1, q1);
        A1_col = sycl::malloc_shared<int>(nnz1, q1);
        A1_val = sycl::malloc_shared<kernel_type>(nnz1, q1);
        B1_val = sycl::malloc_shared<kernel_type>(k * n, q1);
        C1_val = sycl::malloc_shared<kernel_type>(m1 * n, q1);
        D1_val = new kernel_type[m1 * n];
        int now_num = 0;
        for(int i = m0; i < m + 1; i++){
            A1_row_ptr[i - m0] = now_num;
            int start = row_ptr[i];
            int num = row_ptr[i + 1] - start;
            for(int j = 0; j < num; j++){
                int col_id_now = colums[j + start];
                A1_col[now_num] = col_id_now;
                A1_val[now_num] = values[j + start];
                now_num++;
            }
        }
        A1_row_ptr[m + 1 - m0] = now_num;
        assert(nnz2 == now_num);
        oneapi::mkl::sparse::init_matrix_handle(&(A1_h));
        oneapi::mkl::sparse::set_csr_data(A1_h, m1, k, oneapi::mkl::index_base::zero, A1_row_ptr, A1_col, A1_val);
    }
    std::thread t0([&](){
        q0.memcpy(B0_val, x, sizeof(kernel_type) * k * n).wait();
        auto e0 = oneapi::mkl::sparse::gemm(q0, oneapi::mkl::layout::C, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 1, A0_h, B0_val, n, k, 0, C0_val, m0);
        e0.wait();
        q0.memcpy(D0_val, C0_val, sizeof(kernel_type) * m0 * n).wait();
    });

    std::thread t1([&](){
        q1.memcpy(B1_val, x, sizeof(kernel_type) * k * n).wait();
        auto e1 = oneapi::mkl::sparse::gemm(q1, oneapi::mkl::layout::C, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 1, A1_h, B1_val, n, k, 0, C1_val, m1);
        e1.wait();
        q1.memcpy(D1_val, C1_val, sizeof(kernel_type) * m1 * n).wait();
    });
    t0.join();
    t1.join();

    for(long long i = 0; i < n; ++i){
        memcpy(C_host + i * m, D0_val + i * m0, sizeof(kernel_type) * m0);
        memcpy(C_host + i * m + m0, D1_val + i * m1, sizeof(kernel_type) * m1);
    }
    first_tag = 0;
}


void dns_csr_mul_CSR_MKL_ColDensity_gpu(int mm, int nn, int kk, kernel_type *dense_mat,SparseData& sparseData, int nnzz, kernel_type* result, float bbeta){
    //double tt0 = GetTime();
    //printf(" 1 :%d %d %d\n", mm, kk, nn);
    long long n = nn;
    long long m = mm;
    long long k = kk;
    long long nnz = sparseData.val_data.size();
    //assert(nnz == nnzz);
    //assert(nnz == sparseData.row_ptr[m + 1]);
    kernel_type *values = (kernel_type*)sparseData.val_data.data();
    int *colums = (int*)sparseData.col_ptr.data();
    int *row_ptr = (int*)sparseData.row_ptr.data();
    kernel_type *x = (kernel_type*)dense_mat;
    kernel_type *y = (kernel_type*)result;
    static sycl::queue q(sycl::device{sycl::gpu_selector()});
    //static sycl::queue q(sycl::device{sycl::cpu_selector()});
    static kernel_type *A_val;
    static kernel_type *B_val;
    static kernel_type *C_val;
    static int *A_row_ptr;
    static int *A_col;
    static oneapi::mkl::sparse::matrix_handle_t A_h;
    static int first_tag = 1;
    if(first_tag){
        //double t0 = GetTime();
        A_row_ptr = sycl::malloc_shared<int>(m + 1, q);
        A_col = sycl::malloc_shared<int>(nnz, q);
        A_val = sycl::malloc_shared<kernel_type>(nnz, q);
        B_val = sycl::malloc_shared<kernel_type>(k * n, q);
        C_val = sycl::malloc_shared<kernel_type>(m * k, q);
        q.memcpy(A_row_ptr, row_ptr, sizeof(int) * (m + 1));
        q.memcpy(A_col, colums, sizeof(int) * nnz);
        q.memcpy(A_val, values, sizeof(kernel_type) * nnz);
        q.memset(B_val, 0, sizeof(kernel_type) * k * n);
        q.memset(C_val, 0, sizeof(kernel_type) * m * n);
        q.wait();
        oneapi::mkl::sparse::init_matrix_handle(&(A_h));
        oneapi::mkl::sparse::set_csr_data(A_h, m, k, oneapi::mkl::index_base::zero, A_row_ptr, A_col, A_val);
        //printf("sp init cost %lf\n", GetTime() - t0);
    }
    q.memcpy(C_val, y, sizeof(kernel_type) * m * n).wait();
    q.memcpy(B_val, x, sizeof(kernel_type) * k * n).wait();
    auto ee = oneapi::mkl::sparse::gemm(q, oneapi::mkl::layout::C, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 1, A_h, B_val, n, k, bbeta, C_val, m);
    ee.wait();
    //printf("done\n");
    q.memcpy(y, C_val, sizeof(kernel_type) * m * n).wait();
    //if(first_tag){
    //    printf("sp 1 round cost %lf\n", GetTime() - tt0);
    //}
    first_tag = 0;
}


void dns_csr_mul_CSR_MKL_ColDensity_gpu(int mm, int nn, int kk, kernel_type *dense_mat,SparseData& sparseData, int nnzz, kernel_type* &result){
    //printf(" 1 :%d %d %d\n", mm, kk, nn);
    long long n = nn;
    long long m = mm;
    long long k = kk;
    long long nnz = sparseData.val_data.size();
    assert(nnz == nnzz);
    assert(nnz == sparseData.row_ptr[m + 1]);
    kernel_type *values = (kernel_type*)sparseData.val_data.data();
    int *colums = (int*)sparseData.col_ptr.data();
    int *row_ptr = (int*)sparseData.row_ptr.data();
    kernel_type *x = (kernel_type*)dense_mat;
    kernel_type *y = (kernel_type*)result;
    static sycl::queue q(sycl::device{sycl::gpu_selector()});
    //static sycl::queue q(sycl::device{sycl::cpu_selector()});
    static kernel_type *A_val;
    static kernel_type *B_val;
    static kernel_type *C_val;
    static int *A_row_ptr;
    static int *A_col;
    static oneapi::mkl::sparse::matrix_handle_t A_h;
    static int first_tag = 1;
    if(first_tag){
        A_row_ptr = sycl::malloc_shared<int>(m + 1, q);
        A_col = sycl::malloc_shared<int>(nnz, q);
        A_val = sycl::malloc_shared<kernel_type>(nnz, q);
        B_val = sycl::malloc_shared<kernel_type>(k * n, q);
        C_val = sycl::malloc_shared<kernel_type>(m * k, q);
        q.memcpy(A_row_ptr, row_ptr, sizeof(int) * (m + 1)).wait();
        q.memcpy(A_col, colums, sizeof(int) * nnz).wait();
        q.memcpy(A_val, values, sizeof(kernel_type) * nnz).wait();
        oneapi::mkl::sparse::init_matrix_handle(&(A_h));
        oneapi::mkl::sparse::set_csr_data(A_h, m, k, oneapi::mkl::index_base::zero, A_row_ptr, A_col, A_val);
    }
    q.memcpy(B_val, x, sizeof(kernel_type) * k * n).wait();
    auto ee = oneapi::mkl::sparse::gemm(q, oneapi::mkl::layout::C, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 1, A_h, B_val, n, k, 0, C_val, m);
    ee.wait();
    //printf("done\n");
    q.memcpy(y, C_val, sizeof(kernel_type) * m * n).wait();
    first_tag = 0;
}


void dns_csr_mul_Dense_MKL_ColDensity_cpu_man(const KernelMatrix &k_mat,int m, int n, int k,kernel_type* &dense_mat,DenseData& denseData,
                                      kernel_type* &result){
    printf("2 : %d %d %d\n", m, k, n);

    const kernel_type* dense_val = denseData.val;
//    kernel_type * result_data = result.host_data();
    //cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, 1, dense_val, k, dense_mat, k, 0, result, m);
#pragma omp parallel for
    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            kernel_type sum=0;
            for (int kk=0;kk<k;kk++){
                sum+=dense_val[i*denseData.col+kk] * dense_mat[kk+j*k];
            }
            result[i+j*m] = sum;
        }
    }



    //#pragma omp parallel for
    //    for (int i=0;i<m;i++){
    //        for (int j=0;j<n;j++){
    //            kernel_type sum=0;
    //            for (int kk=0;kk<k;kk++){
    //                sum+=dense_val[i*denseData.n+kk] * dense_mat[kk+j*k];
    //            }
    //            result_data[i+j*m] = sum;
    //
    //        }
    //    }

    //    kernel_type* result_data = new kernel_type[m*n];
    //    kernel_type * ref_data = result.host_data();
    //#pragma omp parallel for num_threads(64)
    //    for (int i=0;i<m;i++){
    //        int row_begin=row_index_data[i];
    //        int row_end=row_index_data[i+1];
    //        for (int j=0;j<n;j++){
    //            kernel_type sum=0;
    //
    //            for (int kk=row_begin;kk<row_end;kk++){
    //                sum+=sparse_data[kk]*dense_data[col_index_data[kk]+j*k];
    //            }
    //
    //            ref_data[i+j*m]=expf(-(self_dot1_data[self_dot0_idx_data[j]] + self_dot1_data[i] - sum*2)*gamma);
    //        }
    //    }




}


//void dns_csr_mul_Dense_MKL_ColDensity_cpu(const KernelMatrix &k_mat,int m, int n, int k,kernel_type* &dense_mat,DenseData& denseData,
//                                      kernel_type* &result){
//    printf("2 : %d %d %d\n", m, k, n);
//
//    const kernel_type* dense_val = denseData.val;
////    kernel_type * result_data = result.host_data();
//    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, 1, dense_val, k, dense_mat, k, 0.0, result, m);
//
//    //#pragma omp parallel for
//    //    for (int i=0;i<m;i++){
//    //        for (int j=0;j<n;j++){
//    //            kernel_type sum=0;
//    //            for (int kk=0;kk<k;kk++){
//    //                sum+=dense_val[i*denseData.n+kk] * dense_mat[kk+j*k];
//    //            }
//    //            result_data[i+j*m] = sum;
//    //
//    //        }
//    //    }
//
//    //    kernel_type* result_data = new kernel_type[m*n];
//    //    kernel_type * ref_data = result.host_data();
//    //#pragma omp parallel for num_threads(64)
//    //    for (int i=0;i<m;i++){
//    //        int row_begin=row_index_data[i];
//    //        int row_end=row_index_data[i+1];
//    //        for (int j=0;j<n;j++){
//    //            kernel_type sum=0;
//    //
//    //            for (int kk=row_begin;kk<row_end;kk++){
//    //                sum+=sparse_data[kk]*dense_data[col_index_data[kk]+j*k];
//    //            }
//    //
//    //            ref_data[i+j*m]=expf(-(self_dot1_data[self_dot0_idx_data[j]] + self_dot1_data[i] - sum*2)*gamma);
//    //        }
//    //    }
//
//
//
//
//}
//


void dns_csr_mul_Dense_MKL_ColDensity_cpu(const KernelMatrix &k_mat,int m, int n, int k,kernel_type* &dense_mat,DenseData& denseData,
                                      kernel_type* result, float bbeta){
    printf("2 : %d %d %d\n", m, k, n);

    const kernel_type* dense_val = denseData.val;
//    kernel_type * result_data = result.host_data();
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, 1, dense_val, k, dense_mat, k, bbeta, result, m);

    //#pragma omp parallel for
    //    for (int i=0;i<m;i++){
    //        for (int j=0;j<n;j++){
    //            kernel_type sum=0;
    //            for (int kk=0;kk<k;kk++){
    //                sum+=dense_val[i*denseData.n+kk] * dense_mat[kk+j*k];
    //            }
    //            result_data[i+j*m] = sum;
    //
    //        }
    //    }

    //    kernel_type* result_data = new kernel_type[m*n];
    //    kernel_type * ref_data = result.host_data();
    //#pragma omp parallel for num_threads(64)
    //    for (int i=0;i<m;i++){
    //        int row_begin=row_index_data[i];
    //        int row_end=row_index_data[i+1];
    //        for (int j=0;j<n;j++){
    //            kernel_type sum=0;
    //
    //            for (int kk=row_begin;kk<row_end;kk++){
    //                sum+=sparse_data[kk]*dense_data[col_index_data[kk]+j*k];
    //            }
    //
    //            ref_data[i+j*m]=expf(-(self_dot1_data[self_dot0_idx_data[j]] + self_dot1_data[i] - sum*2)*gamma);
    //        }
    //    }




}


void dns_csr_mul_CSR_MKL_ColDensity_gpu_block(int mm, int nn, int kk, kernel_type *dense_mat,SparseData& sparseData, int nnzz, kernel_type* &result){

	dns_csr_mul_block_one_gpu(mm, nn, kk, dense_mat, (kernel_type*)sparseData.val_data.data(), (int*)sparseData.row_ptr.data(), (int*)sparseData.col_ptr.data(), nnzz, result);
	//dns_csr_mul_block_two_gpu(mm, nn, kk, dense_mat, (kernel_type*)sparseData.val_data.data(), (int*)sparseData.row_ptr.data(), (int*)sparseData.col_ptr.data(), nnzz, result);



}



void get_working_set_ins_ColDensity(const KernelMatrix &k_mat,SparseData& sparseData,
                                    const SyncArray<int> &data_row_idx,kernel_type *&data_rows, DenseData& denseData,kernel_type* &dense_rows ,int m){
    const int *data_row_idx_data = data_row_idx.host_data();
    const int *row_ptr_data = sparseData.row_ptr.data();
    const int *col_ind_data = sparseData.col_ptr.data();
    const kernel_type *val_data = sparseData.val_data.data();

    if (sparseData.is_use==true){
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < m; i++) {
            int row = data_row_idx_data[i];
            for (int j = row_ptr_data[row]; j < row_ptr_data[row + 1]; ++j) {
                int col = col_ind_data[j];
                data_rows[i * sparseData.col + col] = val_data[j]; //row major
            }
        }
    }

    if (denseData.is_use == true){
#pragma omp parallel for schedule(guided)
        for (int i=0;i<m;i++){
            int row = data_row_idx_data[i];
            for (int j=0;j<denseData.col;j++)
                dense_rows[i*denseData.col+j] = denseData.val[row*denseData.col+j];
        }
    }

}

void get_working_set_ins_ColDensity(const KernelMatrix &k_mat,const SyncArray<kernel_type> &val, const SyncArray<int> &col_ind, const SyncArray<int> &row_ptr,
                                    const SyncArray<int> &data_row_idx, SyncArray<kernel_type> &data_rows, DenseData& denseData,kernel_type* &dense_rows ,int m, int n){
    const int *data_row_idx_data = data_row_idx.host_data();
    kernel_type *data_rows_data = data_rows.host_data();
    const int *row_ptr_data = row_ptr.host_data();
    const int *col_ind_data = col_ind.host_data();
    const kernel_type *val_data = val.host_data();
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < m; i++) {
        int row = data_row_idx_data[i];
        for (int j = row_ptr_data[row]; j < row_ptr_data[row + 1]; ++j) {
            int col = col_ind_data[j];
            data_rows_data[i * n + col] = val_data[j]; //row major
        }
    }

#pragma omp parallel for schedule(guided)
    for (int i=0;i<m;i++){
        int row = data_row_idx_data[i];
        for (int j=0;j<denseData.col;j++)
            dense_rows[i*denseData.col+j] = denseData.val[row*denseData.col+j];
    }




}





void get_dot_product_dns_csr_ColDensity(const KernelMatrix &k_mat,DenseData &denseData_cpu,SparseData& sparseData_cpu,DenseData &denseData_gpu,SparseData& sparseData_gpu,const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product , SyncArray<kernel_type> &data_rows){
//    memset(data_rows.host_data(),0x00,sizeof(kernel_type)*idx.size()*k_mat.n_features_);

    static bool is_fir = 1;
    static kernel_type *dense_rows_cpu;
    static kernel_type *sparse_row_cpu;
    static kernel_type *dense_rows_gpu;
    static kernel_type *sparse_row_gpu;

    static kernel_type *dot_product_csr_cpu;
    static kernel_type *dot_product_dense_cpu;
    static kernel_type *dot_product_csr_gpu;
    static kernel_type *dot_product_dense_gpu;

    //printf("Start memcpy\n");

    if(is_fir){
        dense_rows_cpu = new kernel_type[idx.size()*denseData_cpu.col];
        sparse_row_cpu = new kernel_type[idx.size()*sparseData_cpu.col];

        dense_rows_gpu = new kernel_type[idx.size()*denseData_gpu.col];
        sparse_row_gpu = new kernel_type[idx.size()*sparseData_gpu.col];

        dot_product_csr_cpu = new kernel_type[idx.size()*sparseData_cpu.row];
        dot_product_dense_cpu = new kernel_type[idx.size()*denseData_cpu.row];

        dot_product_csr_gpu = new kernel_type[idx.size()*sparseData_gpu.row];
        dot_product_dense_gpu = new kernel_type[idx.size()*denseData_gpu.row];

        is_fir = 0;
    }


    TDEF(aaa1)
        TSTART(aaa1)
    memset(sparse_row_gpu,0,sizeof(kernel_type)*idx.size()*sparseData_gpu.col);
    memset(sparse_row_cpu,0,sizeof(kernel_type)*idx.size()*sparseData_cpu.col);
    TEND(aaa1)
        TPRINT(aaa1, "$#%memset cost  : ")


    //printf("End memcpy\n");
    //get_working_set_ins_ColDensity(k_mat,k_mat.val_, k_mat.col_ind_, k_mat.row_ptr_, idx, data_rows, denseData_cpu, dense_rows, idx.size(), k_mat.n_features_);
    TDEF(aaa2)
        TSTART(aaa2)
    get_working_set_ins_ColDensity(k_mat,sparseData_cpu, idx, sparse_row_cpu, denseData_cpu, dense_rows_cpu, idx.size());
    get_working_set_ins_ColDensity(k_mat,sparseData_gpu, idx, sparse_row_gpu, denseData_gpu, dense_rows_gpu, idx.size());
    TEND(aaa2)
        TPRINT(aaa2, "$#%get working cost  : ")


    TDEF(dense)
    TSTART(dense)

    float bbeta = 0;

    if (denseData_cpu.is_use){
        //printf("dense cpu\n");
        //dns_csr_mul_Dense_MKL_ColDensity_cpu(k_mat,k_mat.n_instances_,idx.size(),denseData_cpu.col,dense_rows_cpu,denseData_cpu,dot_product_dense_cpu);
        dns_csr_mul_Dense_MKL_ColDensity_cpu(k_mat,k_mat.n_instances_,idx.size(),denseData_cpu.col,dense_rows_cpu,denseData_cpu,dot_product.host_data(),bbeta);
        bbeta = 1.0;
    }
    if (denseData_gpu.is_use) {
        //printf("dense gpu\n");
        //dns_csr_mul_Dense_MKL_ColDensity_cpu_man(k_mat,k_mat.n_instances_,idx.size(),denseData_gpu.col,dense_rows_gpu,denseData_gpu,dot_product_dense_gpu);
        //dns_csr_mul_Dense_MKL_ColDensity_gpu(k_mat.n_instances_,idx.size(),denseData_gpu.col,dense_rows_gpu,denseData_gpu,dot_product_dense_gpu);
        dns_csr_mul_Dense_MKL_ColDensity_gpu(k_mat.n_instances_,idx.size(),denseData_gpu.col,dense_rows_gpu,denseData_gpu,dot_product.host_data(),bbeta);
        //dns_csr_mul_Dense_MKL_ColDensity_two_gpu(k_mat.n_instances_,idx.size(),denseData_gpu.col,dense_rows_gpu,denseData_gpu,dot_product.host_data(),bbeta);
        bbeta = 1.0;
    }


    TEND(dense)
    TPRINT(dense,"$#%dense dense mul time  : " )

    TDEF(sparse)
    TSTART(sparse)
    if (sparseData_cpu.is_use) {
        //printf("sp cpu\n");
        //dns_csr_mul_CSR_MKL_ColDensity_cpu(k_mat,k_mat.n_instances_,idx.size(),k_mat.n_features_,sparse_row_cpu,sparseData_cpu,sparseData_cpu.val_data.size(),dot_product_csr_cpu);
        dns_csr_mul_CSR_MKL_ColDensity_cpu(k_mat,k_mat.n_instances_,idx.size(),k_mat.n_features_,sparse_row_cpu,sparseData_cpu,sparseData_cpu.val_data.size(),dot_product.host_data(),bbeta);
        bbeta = 1.0;
    }
    if (sparseData_gpu.is_use) {
        //printf("sp gpu\n");
        //dns_csr_mul_CSR_MKL_ColDensity_gpu(k_mat.n_instances_,idx.size(),k_mat.n_features_,sparse_row_gpu,sparseData_gpu,sparseData_gpu.val_data.size(),dot_product_csr_gpu);
        dns_csr_mul_CSR_MKL_ColDensity_gpu(k_mat.n_instances_,idx.size(),k_mat.n_features_,sparse_row_gpu,sparseData_gpu,sparseData_gpu.val_data.size(),dot_product.host_data(),bbeta);
        //dns_csr_mul_CSR_MKL_ColDensity_gpu_block(k_mat.n_instances_,idx.size(),k_mat.n_features_,sparse_row_gpu,sparseData_gpu,sparseData_gpu.val_data.size(),dot_product_csr_gpu);
        //dns_csr_mul_CSR_MKL_ColDensity_two_gpu(k_mat.n_instances_,idx.size(),k_mat.n_features_,sparse_row_gpu,sparseData_gpu,sparseData_gpu.val_data.size(),dot_product_csr_gpu);
        bbeta = 1.0;
    }
    TEND(sparse)
    TPRINT(sparse,"$#%sparse dense mul time : ")
    TDEF(merge)

    TSTART(merge)

    //memset(dot_product.host_data(),0,sizeof(kernel_type)*idx.size()*k_mat.n_instances_);

    //if (denseData_cpu.is_use) MergeResult(dot_product.host_data(),dot_product_dense_cpu,idx.size()*k_mat.n_instances_);
    //if (denseData_gpu.is_use) MergeResult(dot_product.host_data(),dot_product_dense_gpu,idx.size()*k_mat.n_instances_);
    //if (sparseData_cpu.is_use) MergeResult(dot_product.host_data(),dot_product_csr_cpu,idx.size()*k_mat.n_instances_);
    //if (sparseData_gpu.is_use) MergeResult(dot_product.host_data(),dot_product_csr_gpu,idx.size()*k_mat.n_instances_);


    //    MergeResult(dot_product.host_data(),dot_product_sparse,k_mat.n_instances_*idx.size());


    TEND(merge)
    TPRINT(merge,"$#%merge  time : ")
}


void get_dot_product_dns_csr_ColDensity(const KernelMatrix &k_mat,DenseData &denseData,SparseData& sparseData,const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product , SyncArray<kernel_type> &data_rows){
    memset(data_rows.host_data(),0x00,sizeof(kernel_type)*idx.size()*k_mat.n_features_);

    kernel_type *dense_rows = new kernel_type[idx.size()*denseData.col];
    get_working_set_ins_ColDensity(k_mat,k_mat.val_, k_mat.col_ind_, k_mat.row_ptr_, idx, data_rows, denseData, dense_rows, idx.size(), k_mat.n_features_);
    TDEF(dense)
    TSTART(dense)
    //dns_csr_mul_Dense_MKL_ColDensity_cpu(k_mat,k_mat.n_instances_,idx.size(),denseData.col,dense_rows,denseData,dot_product);
    //dns_csr_mul_Dense_MKL_ColDensity_gpu(k_mat.n_instances_,idx.size(),denseData.col,dense_rows,denseData,dot_product.host_data());


    TEND(dense)
    TPRINT(dense,"$#%dense dense mul time  : " )

    TDEF(sparse)
    TSTART(sparse)
    static kernel_type *dot_product_sparse;
    static bool fg=false;
    if (!fg) {
        fg=true;
        dot_product_sparse=new kernel_type[k_mat.n_instances_*idx.size()];

    }
    //TODO n_features
    //dns_csr_mul_CSR_MKL_ColDensity_gpu(k_mat.n_instances_,idx.size(),k_mat.n_features_,data_rows.host_data(),sparseData,sparseData.val_data.size(),dot_product_sparse);
    //dns_csr_mul_CSR_MKL_ColDensity_cpu(k_mat,k_mat.n_instances_,idx.size(),k_mat.n_features_,data_rows,sparseData,sparseData.val_data.size(),dot_product_sparse);
    TEND(sparse)
    TPRINT(sparse,"$#%sparse dense mul time : ")

    delete dense_rows;

    TDEF(merge)

    TSTART(merge)
    MergeResult(dot_product.host_data(),dot_product_sparse,k_mat.n_instances_*idx.size());
    TEND(merge)
    TPRINT(merge,"$#%merge  time : ")
}





void get_rows_ColDensity(const KernelMatrix &k_mat,DenseData &denseData_cpu,SparseData& sparseData_cpu,DenseData &denseData_gpu,SparseData& sparseData_gpu,const SyncArray<int> &idx,
                         SyncArray<kernel_type> &kernel_rows,SyncArray<kernel_type> &data_rows){

    CHECK_GE(kernel_rows.size(), idx.size() * k_mat.n_instances_) << "kernel_rows memory is too small";
#ifdef USE_CUDA
    get_dot_product_dns_csr(idx, kernel_rows);
#else

    if(k_mat.n_features_ < 1000000)
        get_dot_product_dns_csr_ColDensity(k_mat,denseData_cpu,sparseData_cpu,denseData_gpu,sparseData_gpu,idx, kernel_rows,data_rows);
    else
        get_dot_product_csr_csr(idx, kernel_rows);
    //    get_dot_product_dns_dns(idx, kernel_rows);
#endif
    TDEF(RBF)
        TSTART(RBF)
    switch (k_mat.param.kernel_type) {
        case SvmParam::RBF:
        case SvmParam::PRECOMPUTED://precomputed uses rbf as default
            //In this;
            RBF_kernel_intel_gpu(idx, k_mat.self_dot_, kernel_rows, idx.size(), k_mat.n_instances_, k_mat.param.gamma);

            //printf("RBF OR PRECOMPUTED\n");
            break;
        case SvmParam::LINEAR:
            //do nothing
            //printf("LINEAR\n");
            break;
        case SvmParam::POLY:
            poly_kernel_intel_gpu(kernel_rows, k_mat.param.gamma, k_mat.param.coef0, k_mat.param.degree, kernel_rows.size());
            //printf("POLY\n");
            break;
        case SvmParam::SIGMOID:
            sigmoid_kernel_intel_gpu(kernel_rows, k_mat.param.gamma, k_mat.param.coef0, kernel_rows.size());
            //printf("SIGMOID\n");

            break;
    }
    TEND(RBF)
        TPRINT(RBF, "$#% RBF time  : ")
}




void get_rows_ColDensity(const KernelMatrix &k_mat,DenseData &denseData,SparseData& sparseData,const SyncArray<int> &idx,
                         SyncArray<kernel_type> &kernel_rows,SyncArray<kernel_type> &data_rows){

    CHECK_GE(kernel_rows.size(), idx.size() * k_mat.n_instances_) << "kernel_rows memory is too small";
#ifdef USE_CUDA
    get_dot_product_dns_csr(idx, kernel_rows);
#else

    if(k_mat.n_features_ < 1000000)
        get_dot_product_dns_csr_ColDensity(k_mat,denseData,sparseData,idx, kernel_rows,data_rows);
    else
        get_dot_product_csr_csr(idx, kernel_rows);
    //    get_dot_product_dns_dns(idx, kernel_rows);
#endif
    switch (k_mat.param.kernel_type) {
        case SvmParam::RBF:
        case SvmParam::PRECOMPUTED://precomputed uses rbf as default
            //In this;
            RBF_kernel_intel_gpu(idx, k_mat.self_dot_, kernel_rows, idx.size(), k_mat.n_instances_, k_mat.param.gamma);

            //printf("RBF OR PRECOMPUTED\n");
            break;
        case SvmParam::LINEAR:
            //do nothing
            //printf("LINEAR\n");
            break;
        case SvmParam::POLY:
            poly_kernel_intel_gpu(kernel_rows, k_mat.param.gamma, k_mat.param.coef0, k_mat.param.degree, kernel_rows.size());
            //printf("POLY\n");
            break;
        case SvmParam::SIGMOID:
            sigmoid_kernel_intel_gpu(kernel_rows, k_mat.param.gamma, k_mat.param.coef0, kernel_rows.size());
            //printf("SIGMOID\n");

            break;
    }
}














/*
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 */



//
//#include <immintrin.h>
//void mymemcpy(void * __restrict__ dst_, const void * __restrict__ src_, size_t size)
//{
//    char * __restrict__ dst = (char*)dst_;
//    const char* __restrict__ src = (char*)src_;
//
//    size_t size_64 = size & ~63;
//    for(size_t i = 0; i < size_64; i += 64)
//    {
//        __m512i s = _mm512_loadu_si512((void*)(src + i));
//        _mm512_storeu_si512((void*)(dst + i), s);
//    }
//
//    size_t size_left = size - 64;
//    __m512i s = _mm512_loadu_si512((void*)(src + size_left));
//    _mm512_storeu_si512((void*)(dst + size_left), s);
//}
//
//



using namespace std::chrono;

using namespace svm_kernel;

void
CSMOSolver::solve(const KernelMatrix &k_mat, const SyncArray<int> &y, SyncArray<float_type> &alpha, float_type &rho,
                  SyncArray<float_type> &f_val, float_type eps, float_type Cp, float_type Cn, int ws_size,
                  int out_max_iter) const {
    int n_instances = k_mat.n_instances();
    int q = ws_size / 2;





    SyncArray<int> working_set(ws_size);
    SyncArray<int> working_set_first_half(q);
    SyncArray<int> working_set_last_half(q);
#ifdef USE_CUDA
    working_set_first_half.set_device_data(working_set.device_data());
    working_set_last_half.set_device_data(&working_set.device_data()[q]);
#endif
    working_set_first_half.set_host_data(working_set.host_data());
    working_set_last_half.set_host_data(&working_set.host_data()[q]);

    SyncArray<int> f_idx(n_instances);
    SyncArray<int> f_idx2sort(n_instances);
    SyncArray<float_type> f_val2sort(n_instances);
    SyncArray<float_type> alpha_diff(ws_size);
    SyncArray<float_type> diff(2);

    SyncArray<kernel_type> k_mat_rows(ws_size * k_mat.n_instances());
    SyncArray<kernel_type> k_mat_rows_first_half(q * k_mat.n_instances());
    SyncArray<kernel_type> k_mat_rows_last_half(q * k_mat.n_instances());


    SyncArray<kernel_type> data_rows(ws_size*k_mat.n_features_);


#ifdef USE_CUDA
    k_mat_rows_first_half.set_device_data(k_mat_rows.device_data());
    k_mat_rows_last_half.set_device_data(&k_mat_rows.device_data()[q * k_mat.n_instances()]);
#else
    k_mat_rows_first_half.set_host_data(k_mat_rows.host_data());
    k_mat_rows_last_half.set_host_data(&k_mat_rows.host_data()[q * k_mat.n_instances()]);
#endif
    int *f_idx_data = f_idx.host_data();
    for (int i = 0; i < n_instances; ++i) {
        f_idx_data[i] = i;
    }
    init_f(alpha, y, k_mat, f_val);
    LOG(INFO) << "training start";

    int max_iter = max(100000, ws_size > INT_MAX / 100 ? INT_MAX : 100 * ws_size);
    long long local_iter = 0;

    //avoid infinite loop of repeated local diff
    int same_local_diff_cnt = 0;
    float_type previous_local_diff = INFINITY;
    int swap_local_diff_cnt = 0;
    float_type last_local_diff = INFINITY;
    float_type second_last_local_diff = INFINITY;

    high_resolution_clock::time_point t0,t1;
    double dt0=0.0;

    t0 = high_resolution_clock::now();



    /*
     *
     * 常量打印
     *
     *
     */

    printf("ws_size : %d\n",ws_size);

    TDEF(CHANGE)
    TSTART(CHANGE)

    SparseData sparseData_cpu;
    DenseData denseData_cpu;


    SparseData sparseData_gpu;
    DenseData denseData_gpu;

    printf("---------------------------------------------------------------\n");


    //CSRtoDenseandCSR(k_mat,denseData,sparseData);
    CSRtoDenseandCSR(k_mat, denseData_cpu, sparseData_cpu, denseData_gpu, sparseData_gpu);


    TEND(CHANGE)
    TPRINT(CHANGE,"Change CSR to BCSR time : ")



    TDEF(select)
    TDEF(workset)
    TDEF(matrix)
    long long all_select=0;
    long long all_workset=0;
    long long all_matrix=0;
    float_type obj;
    for (int iter = 0;; ++iter) {
        //select working set
        TDEF(cccp)
        TSTART(cccp)
		//mymemcpy(f_idx2sort.host_data(), f_idx.host_data(), sizeof(int) * f_idx.size());
		//mymemcpy(f_val2sort.host_data(), f_val.host_data(), sizeof(float_type) * f_val.size());	
        f_idx2sort.copy_from(f_idx);
        f_val2sort.copy_from(f_val);
        TEND(cccp)
        TPRINT(cccp, "#$# copy time : ")

        TDEF(sort)
        TSTART(sort)
        sort_f(f_val2sort, f_idx2sort);
        TEND(sort)
        TPRINT(sort, "#$# sort time : ")
        vector<int> ws_indicator(n_instances, 0);

        TSTART(select)
        if (0 == iter) {
            TSTART(workset)
            select_working_set(ws_indicator, f_idx2sort, y, alpha, Cp, Cn, working_set);
            TEND(workset)
            TPRINT(workset,"#$# work set one time : ");
            all_workset+= TINT(workset);

            TSTART(matrix)
            //            k_mat.get_rows(working_set, k_mat_rows);
            //get_rows_intel_gpu(k_mat,sparseData,sparsewithoutdenseData,working_set, k_mat_rows,data_rows);
            get_rows_ColDensity(k_mat,denseData_cpu,sparseData_cpu,denseData_gpu,sparseData_gpu,working_set,k_mat_rows,data_rows);
            TEND(matrix)
            TPRINT(matrix,"#$# matrix one time : ");
            all_matrix+= TINT(matrix);
        } else {
            TSTART(workset)
            working_set_first_half.copy_from(working_set_last_half);
			//mymemcpy(working_set_first_half.host_data(), working_set_last_half.host_data(), sizeof(int) * working_set_last_half.size());

            int *working_set_data = working_set.host_data();
            for (int i = 0; i < q; ++i) {
                ws_indicator[working_set_data[i]] = 1;
            }
            select_working_set(ws_indicator, f_idx2sort, y, alpha, Cp, Cn, working_set_last_half);
            TEND(workset)
            TPRINT(workset,"#$# work set one time : ");
            all_workset+= TINT(workset);

            TSTART(matrix)
            k_mat_rows_first_half.copy_from(k_mat_rows_last_half);
			//mymemcpy(k_mat_rows_first_half.host_data(), k_mat_rows_last_half.host_data(), sizeof(kernel_type) * k_mat_rows_last_half.size());
            //k_mat.get_rows(working_set_last_half, k_mat_rows_last_half);
            //get_rows_intel_gpu(k_mat,sparseData,sparsewithoutdenseData,working_set_last_half, k_mat_rows_last_half,data_rows);
            get_rows_ColDensity(k_mat,denseData_cpu,sparseData_cpu,denseData_gpu,sparseData_gpu,working_set_last_half,k_mat_rows_last_half,data_rows);
            TEND(matrix)
            TPRINT(matrix,"#$# matrix one time : ");
            all_matrix+= TINT(matrix);

        }
        TEND(select)
        all_select+= TINT(select);
        TPRINT(select,"#$# select one time :")
        //local smo
            TDEF(up22)
            TSTART(up22)
        
        smo_kernel(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat.diag(), n_instances, eps, diff,
                   max_iter);
        TEND(up22)
            TPRINT(up22, "#$# up22 time :")
        //update f
        TDEF(up11)
            TSTART(up11)
        update_f(f_val, alpha_diff, k_mat_rows, k_mat.n_instances());
        TEND(up11)
            TPRINT(up11, "#$# up11 time :")
        float_type *diff_data = diff.host_data();
        local_iter += diff_data[1];

        //track unchanged diff
        if (fabs(diff_data[0] - previous_local_diff) < eps * 0.001) {
            same_local_diff_cnt++;
        } else {
            same_local_diff_cnt = 0;
            previous_local_diff = diff_data[0];
        }

        //track unchanged swapping diff
        if(fabs(diff_data[0] - second_last_local_diff) < eps * 0.001){
            swap_local_diff_cnt++;
        } else {
            swap_local_diff_cnt = 0;
        }
        second_last_local_diff = last_local_diff;
        last_local_diff = diff_data[0];

        if (iter % 100 == 0)
            LOG(INFO) << "global iter = " << iter << ", total local iter = " << local_iter << ", diff = "
                      << diff_data[0];
        //todo find some other ways to deal unchanged diff
        //training terminates in three conditions: 1. diff stays unchanged; 2. diff is closed to 0; 3. training reaches the limit of iterations.
        //repeatedly swapping between two diffs
        if ((same_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps) || diff_data[0] < eps ||
            (out_max_iter != -1) && (iter == out_max_iter) ||
            (swap_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps)) {
            rho = calculate_rho(f_val, y, alpha, Cp, Cn);
            LOG(INFO) << "global iter = " << iter << ", total local iter = " << local_iter << ", diff = "
                      << diff_data[0];
            LOG(INFO) << "training finished";
            obj = calculate_obj(f_val, alpha, y);
            LOG(INFO) << "obj = " << obj;
            break;
        }
    }

    printf("#$# All Select Time : %f\n",1.0*all_select/1e6);
    printf("#$# All WorkSet Time : %f\n",1.0*all_workset/1e6);
    printf("#$# All Matrix Time : %f\n",1.0*all_matrix/1e6);

    t1 = high_resolution_clock::now();
    dt0 = dt0 + duration<double>(t1-t0).count();
    printf( "\nFOM: main loop : %11.6lf ms, %11.6lfs \n\n", dt0*1000, dt0 );
    FILE* file = fopen("result.txt", "w");
    fprintf(file, "obj = %.20f\n", obj);
    fclose(file);
}

void
CSMOSolver::select_working_set(vector<int> &ws_indicator, const SyncArray<int> &f_idx2sort, const SyncArray<int> &y,
                               const SyncArray<float_type> &alpha, float_type Cp, float_type Cn,
                               SyncArray<int> &working_set) const {
    int n_instances = ws_indicator.size();
    int p_left = 0;
    int p_right = n_instances - 1;
    int n_selected = 0;
    const int *index = f_idx2sort.host_data();
    const int *y_data = y.host_data();
    const float_type *alpha_data = alpha.host_data();
    int *working_set_data = working_set.host_data();
    while (n_selected < working_set.size()) {
        int i;
        if (p_left < n_instances) {
            i = index[p_left];
            while (ws_indicator[i] == 1 || !is_I_up(alpha_data[i], y_data[i], Cp, Cn)) {
                //construct working set of I_up
                p_left++;
                if (p_left == n_instances) break;
                i = index[p_left];
            }
            if (p_left < n_instances) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }
        if (p_right >= 0) {
            i = index[p_right];
            while (ws_indicator[i] == 1 || !is_I_low(alpha_data[i], y_data[i], Cp, Cn)) {
                //construct working set of I_low
                p_right--;
                if (p_right == -1) break;
                i = index[p_right];
            }
            if (p_right >= 0) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }

    }
}

float_type
CSMOSolver::calculate_rho(const SyncArray<float_type> &f_val, const SyncArray<int> &y, SyncArray<float_type> &alpha,
                          float_type Cp,
                          float_type Cn) const {
    int n_free = 0;
    double sum_free = 0;
    float_type up_value = INFINITY;
    float_type low_value = -INFINITY;
    const float_type *f_val_data = f_val.host_data();
    const int *y_data = y.host_data();
    float_type *alpha_data = alpha.host_data();
    for (int i = 0; i < alpha.size(); ++i) {
        if (is_free(alpha_data[i], y_data[i], Cp, Cn)) {
            n_free++;
            sum_free += f_val_data[i];
        }
        if (is_I_up(alpha_data[i], y_data[i], Cp, Cn)) up_value = min(up_value, f_val_data[i]);
        if (is_I_low(alpha_data[i], y_data[i], Cp, Cn)) low_value = max(low_value, f_val_data[i]);
    }
    return 0 != n_free ? (sum_free / n_free) : (-(up_value + low_value) / 2);
}

void CSMOSolver::init_f(const SyncArray<float_type> &alpha, const SyncArray<int> &y, const KernelMatrix &k_mat,
                        SyncArray<float_type> &f_val) const {
    //todo auto set batch size
    int batch_size = 100;
    vector<int> idx_vec;
    vector<float_type> alpha_diff_vec;
    const int *y_data = y.host_data();
    const float_type *alpha_data = alpha.host_data();
    for (int i = 0; i < alpha.size(); ++i) {
        if (alpha_data[i] != 0) {
            idx_vec.push_back(i);
            alpha_diff_vec.push_back(-alpha_data[i] * y_data[i]);
        }
        if (idx_vec.size() > batch_size || (i == alpha.size() - 1 && !idx_vec.empty())) {
            SyncArray<int> idx(idx_vec.size());
            SyncArray<float_type> alpha_diff(idx_vec.size());
            idx.copy_from(idx_vec.data(), idx_vec.size());
            alpha_diff.copy_from(alpha_diff_vec.data(), idx_vec.size());
            SyncArray<kernel_type> kernel_rows(idx.size() * k_mat.n_instances());
            k_mat.get_rows(idx, kernel_rows);
            update_f(f_val, alpha_diff, kernel_rows, k_mat.n_instances());
            idx_vec.clear();
            alpha_diff_vec.clear();
        }
    }
}

void
CSMOSolver::smo_kernel(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                       SyncArray<float_type> &alpha_diff,
                       const SyncArray<int> &working_set, float_type Cp, float_type Cn,
                       const SyncArray<kernel_type> &k_mat_rows,
                       const SyncArray<kernel_type> &k_mat_diag, int row_len, float_type eps,
                       SyncArray<float_type> &diff,
                       int max_iter) const {
    c_smo_solve(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat_diag, row_len, eps, diff, max_iter);
}

float_type CSMOSolver::calculate_obj(const SyncArray<float_type> &f_val, const SyncArray<float_type> &alpha,
                                     const SyncArray<int> &y) const {
    //todo use parallel reduction for gpu and cpu
    int n_instances = f_val.size();
    float_type obj = 0;
    const float_type *f_val_data = f_val.host_data();
    const float_type *alpha_data = alpha.host_data();
    const int *y_data = y.host_data();
    for (int i = 0; i < n_instances; ++i) {
        obj += alpha_data[i] - (f_val_data[i] + y_data[i]) * alpha_data[i] * y_data[i] / 2;
    }
    return -obj;
}



