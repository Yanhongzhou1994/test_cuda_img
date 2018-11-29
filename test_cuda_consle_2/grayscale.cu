#include "math.h"
#include "stdio.h"
#include "algorithm"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "vector"
#include "vector_functions.hpp"

__global__ void rgba_to_greyscale(const uchar4* const rgbaImage,unsigned char* const greyImage,
	int numRows,int numCols)
{
	int index_x = blockIdx.x*blockDim.x + threadIdx.x;
	int index_y = blockIdx.y*blockDim.y + threadIdx.y;
	int grid_width = gridDim.x*blockDim.x;
	int index = index_y*grid_width + index_x;//index表示图像的指针index
	greyImage[index] = .299f*rgbaImage[index].x + .587f*rgbaImage[index].y + .114f*rgbaImage[index].z;
	
}

extern "C"
void your_rgba_to_greyscale( uchar4* const d_rgbaImage,
	unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
	const int thread = 16;
	const dim3 blockSize(thread,thread,1);
	const dim3 gridSize(ceil(numRows/(float)thread),ceil(numCols/(float)thread),1);
	rgba_to_greyscale<<<gridSize, blockSize >>> (d_rgbaImage,d_greyImage,numRows,numCols);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}