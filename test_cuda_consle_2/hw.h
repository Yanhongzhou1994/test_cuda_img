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


cv::Mat imageRGBA;//��ͨ��ͼ����һ��͸����ͨ��
cv::Mat imageGrey;//�Ҷ�ͼ��

uchar4 *d_rgbaImage_;//device�˵�rgbaͼ��
unsigned char *d_greyImage_;//device�˵ĻҶ�ͼ��

size_t numRows() { return imageRGBA.rows; }//�޷���64λ����
size_t numCols() { return imageRGBA.cols; }


//��ʼ��device�˵��Դ�
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
	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);//�ֳ��ĸ�ͨ��������Ϊû��uchar3
	imageGrey.create(image.rows, image.cols, CV_8UC1);//�����Ҷ�ͼ

	if (!imageRGBA.isContinuous() || !imageGrey.isContinuous())
	{
		std::cerr << "Images aren't continuous!!Exitiong." << std::endl;
		exit(1);
	}

	*inputImage = (uchar4*)imageRGBA.ptr<unsigned char>(0);//��host�˵�*inputImageָ�븳ֵ
	*greyImage = imageGrey.ptr<unsigned char>(0);//��host�˵�*greyImageָ�븳ֵ
	const size_t numPixels = numRows()*numCols();//��������ֵ
	checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4)*numPixels));//��device�˵�rgbaImage�����Դ�
	checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char)*numPixels));//��device��greyImage�����Դ�
	checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)));//��d_greyImage��ʼ��
	//��host�˵�inputImage���Ƶ�d_rgbaImage
	checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4)*numPixels, cudaMemcpyHostToDevice));
	d_rgbaImage_ = *d_rgbaImage;//��Ϊ��free��
	d_greyImage_ = *d_greyImage;//��Ϊ��free��
}

//�Ѽ���Ľ�����ƻ�host��
//�ͷ�device���Դ�
void postProcess(const std::string& output_file)
{
	const int numPixels = numRows()*numCols();
	checkCudaErrors(cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage_, sizeof(unsigned char)*numPixels,
		cudaMemcpyDeviceToHost));
	cv::imwrite(output_file.c_str(), imageGrey);

	cudaFree(d_rgbaImage_);
	cudaFree(d_greyImage_);
}