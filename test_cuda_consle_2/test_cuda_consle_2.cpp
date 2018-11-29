// test_cuda_consle_2.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "hw.h"
#include "helper_timer.h"
//#include "grayscale.cu"��һ�䲻����

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

	//��rgbaת��Ϊgrey
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
	postProcess(output_file);//�������

	sdkDeleteTimer(&timer);
	return 0;
}