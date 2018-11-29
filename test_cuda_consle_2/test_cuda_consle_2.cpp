// test_cuda_consle_2.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "hw.h"
#include "helper_timer.h"
//#include "grayscale.cu"这一句不能有

extern "C" void your_rgba_to_greyscale(uchar4* const d_rgbaImage,
	unsigned char* const d_greyImage, size_t numRows, size_t numCols);



int main()
{
	uchar4 *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;
	std::string input_file = "test.bmp";
	std::string output_file = "save.bmp";
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	//将rgba转换为grey
	your_rgba_to_greyscale( d_rgbaImage, d_greyImage, numRows(), numCols());
	sdkStopTimer(&timer);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	printf("\n");
	int err = printf("%f msecs.\n", sdkGetTimerValue(&timer));
	if (err < 0)
	{
		std::cerr << "Couldn't print timing information! STDOUT Closed" << std::endl;
		exit(1);
	}
	postProcess(output_file);//保存输出

	sdkDeleteTimer(&timer);
	return 0;
}