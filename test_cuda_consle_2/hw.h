#pragma once
#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "string"
#include "iostream"


cv::Mat imageRGBA;//四通道图像，有一个透明度通道
cv::Mat imageGrey;//灰度图像

uchar4 *d_rgbaImage_;//device端的rgba图像
unsigned char *d_greyImage_;//device端的灰度图像

size_t numRows() { return imageRGBA.rows; }//无符号64位整型
size_t numCols() { return imageRGBA.cols; }


//初始化device端的显存
void preProcess(uchar4** inputImage, unsigned char** greyImage, uchar4** d_rgbaImage, unsigned char** d_greyImage,
	const std::string &filename)
{
	checkCudaErrors(cudaFree(0));
	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty())
	{
		std::cerr << "couldn't open file:" << filename << std::endl;
		exit(1);
	}
	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);//分成四个通道，是因为没有uchar3
	imageGrey.create(image.rows, image.cols, CV_8UC1);//创建灰度图

	if (!imageRGBA.isContinuous() || !imageGrey.isContinuous())
	{
		std::cerr << "Images aren't continuous!!Exitiong." << std::endl;
		exit(1);
	}

	*inputImage = (uchar4*)imageRGBA.ptr<unsigned char>(0);//给host端的*inputImage指针赋值
	*greyImage = imageGrey.ptr<unsigned char>(0);//给host端的*greyImage指针赋值
	const size_t numPixels = numRows()*numCols();//计算像素值
	checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4)*numPixels));//给device端的rgbaImage分配显存
	checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char)*numPixels));//给device的greyImage分配显存
	checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)));//给d_greyImage初始化
	//把host端的inputImage复制到d_rgbaImage
	checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4)*numPixels, cudaMemcpyHostToDevice));
	d_rgbaImage_ = *d_rgbaImage;//是为了free用
	d_greyImage_ = *d_greyImage;//是为了free用
}

//把计算的结果复制回host端
//释放device端显存
void postProcess(const std::string& output_file)
{
	const int numPixels = numRows()*numCols();
	checkCudaErrors(cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage_, sizeof(unsigned char)*numPixels,
		cudaMemcpyDeviceToHost));
	cv::imwrite(output_file.c_str(), imageGrey);

	cudaFree(d_rgbaImage_);
	cudaFree(d_greyImage_);
}