#include<opencv2/opencv.hpp>
#include<iostream>
#include<cmath>
using namespace cv;
using namespace std;

#define DISCRETE_FILTER_SIZE 2048
#define LOWPASS_FILTER_LENGTH 10.00000f
#define LINE_SQUARE_CLIP_MAX 100000.0f		//用于辅助找到segLen
#define VECTOR_COMPONENT_MIN 0.050000f

void SynthesizeVectorField(Mat&, Mat&);
void NormalizeVectors(Mat&, Mat&);
void GenerateNoiseImage(Mat&);
void GenBoxFilterLUT(int, float*, float*);
void LinearIntegrateConvolution(Mat&, Mat&, Mat&, Mat&, float*, float*);

void SynthesizeVectorField(Mat& vector_x, Mat& vector_y) {
	int n = vector_x.rows;
	int m = vector_x.cols;
	for (int j = 0; j < m; j++) {
		for (int i = 0; i < n; i++) {
			vector_x.at<float>(i, j) = 0.5f;
			vector_y.at<float>(i, j) = 0.5f;
		}
	}
}

void NormalizeVectors(Mat& vector_x, Mat& vector_y) {
	int n = vector_x.rows;
	int m = vector_y.cols;
	for (int j = 0; j < m; j++) {
		for (int i = 0; i < n; i++) {
			float vecMag = sqrt(vector_x.at<float>(i, j) * vector_x.at<float>(i, j) + vector_y.at<float>(i, j) * vector_y.at<float>(i, j));
			float scale = vecMag == 0.0f ? 0.0f : 1.0f / vecMag;
			vector_x.at<float>(i, j) *= scale;
			vector_y.at<float>(i, j) *= scale;
		}
	}
}

void GenerateNoiseImage(Mat& noiseImage) {
	int n = noiseImage.rows;
	int m = noiseImage.cols;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			int r = rand();
			r = ((r & 0xff) + ((r & 0xff00) >> 8)) & 0xff;
			noiseImage.at<uchar>(i, j) = (unsigned char)r;
		}
	}
}

void GenBoxFilterLUT(int LUTsiz, float* p_LUT0, float* p_LUT1) {
	for (int i = 0; i < LUTsiz; i++) {
		p_LUT0[i] = p_LUT1[i] = i;
	}
}

void LinearIntegrateConvolution(Mat& input, Mat& vector_x, Mat& vector_y, Mat& output, float* p_LUT0, float* p_LUT1) {
	int     advDir;     //标记方向
	int     advcts;     //记录对流次数
	int     ADVCTS = (int)LOWPASS_FILTER_LENGTH * 3;    //最大对流次数是流线长度的三倍

	float		t_acum[2];  //卷积结果
	float		w_acum[2];  //权重加和，用于归一化结果
	float		texVal;			//当前像素值F（floor(P_i)）

	float		curLen;     //当前流线长度
	float		prvLen;     //当前流线长度-segLen_{i-1}
	float		segLen;    //\delta s_i
	float		tmpLen;	//

	float		vct_x;      //当前clip point对应的向量的x方向分量与y方向分量
	float		vct_y;

	float		clp0_x;     //当前位置（位置不是坐标）
	float		clp0_y;
	float		clp1_x;     //下一个位置
	float		clp1_y;

	float		samp_x;	//采样
	float		samp_y;

	float*	wgtLUT;     //对不同方向取对应的LUT
	float		W_ACUM;
	float		len2ID = (DISCRETE_FILTER_SIZE - 1) / LOWPASS_FILTER_LENGTH;
	float		smpWgt;

	int n = input.rows;
	int m = input.cols;
	for (int j = 0; j < m; j++) {
		for (int i = 0; i < n; i++) {
			t_acum[0] = t_acum[1] = w_acum[0] = w_acum[1] = 0.0f;

			for (advDir = 0; advDir < 2; advDir++) {   //一个方向开始
				advcts = 0;
				curLen = 0.0f;
				clp0_x = i + 0.5f;      //初始化
				clp0_y = j + 0.5f;
				wgtLUT = advDir == 0 ? p_LUT0 : p_LUT1;

				while (curLen < LOWPASS_FILTER_LENGTH && advcts < ADVCTS) {     //生成流线，进行卷积运算
					vct_x = vector_x.at<float>(clp0_x, clp0_y);
					vct_y = vector_y.at<float>(clp0_x, clp0_y);

					if (vct_x == 0.0f && vct_y == 0.0f) {
						w_acum[advDir] = advcts == 0 ? 1.0f : w_acum[advDir];	//遇到了x、y分量均为0的点，不能继续前进；如果这只是第一次，那么直接取权值为1
						break;
					}

					vct_x = advDir == 0 ? vct_x : -vct_x;
					vct_y = advDir == 0 ? vct_y : -vct_y;

					segLen = LINE_SQUARE_CLIP_MAX;	//求出segLen（2）式；若使用if语句会慢
					segLen = vct_x < -VECTOR_COMPONENT_MIN ? (int(clp0_x) - clp0_x) / vct_x : segLen;
					segLen = vct_x > VECTOR_COMPONENT_MIN ? (int(int(clp0_x) + 1.5f) - clp0_x) / vct_x : segLen;
					segLen = (vct_y < -VECTOR_COMPONENT_MIN) ? (((tmpLen = (int(clp0_y) - clp0_y) / vct_y) < segLen) ? tmpLen : segLen) : segLen;
					segLen = (vct_y > VECTOR_COMPONENT_MIN) ? (((tmpLen = (int(int(clp0_y) + 1.5f) - clp0_y) / vct_y) < segLen) ? tmpLen : segLen) : segLen;

					prvLen = curLen;
					curLen += segLen;
					segLen += 0.0004f;

					segLen = curLen > LOWPASS_FILTER_LENGTH ? (curLen = LOWPASS_FILTER_LENGTH) - prvLen : segLen;

					clp1_x = clp0_x + vct_x * segLen;	//（1）式
					clp1_y = clp0_y + vct_y * segLen;

					samp_x = (int)clp0_x;	//采样
					samp_y = (int)clp0_y;
					texVal = input.at<uchar>(samp_x, samp_y);

					W_ACUM = wgtLUT[int(curLen * len2ID)];		//（4）式；此处将curLen对应的权重（“曲线质量”）取出
					smpWgt = W_ACUM - w_acum[advDir];		//对每个segLen求出其对应的积分h_i，以作为权值
					w_acum[advDir] = W_ACUM;
					t_acum[advDir] += texVal * smpWgt;

					advcts++;
					clp0_x = clp1_x;
					clp0_y = clp1_y;

					if (clp0_x < 0.0f || clp0_x >= n || clp0_y < 0.0f || clp0_y >= m)  break;	//检查是否越界
				}
			}
			texVal = (t_acum[0] + t_acum[1]) / (w_acum[0] + w_acum[1]);		//（5）式

			texVal = texVal < 0.0f ? 0.0f : texVal;
			texVal = texVal > 255.0f ? 255.0f : texVal;
			output.at<uchar>(i, j) = (unsigned char)texVal;
		}
	}
}


int main() {
	Mat noiseImage(400, 400, CV_8UC1, Scalar(203));
	Mat outputImage(400, 400, CV_8UC1);
	Mat vectorField_x(400, 400, CV_32FC1);
	Mat vectorField_y(400, 400, CV_32FC1);

	float* p_LUT0 = new float[DISCRETE_FILTER_SIZE];
	float* p_LUT1 = new float[DISCRETE_FILTER_SIZE];

	GenerateNoiseImage(noiseImage);
	
	SynthesizeVectorField(vectorField_x, vectorField_y);
	NormalizeVectors(vectorField_x, vectorField_y);
	GenBoxFilterLUT(DISCRETE_FILTER_SIZE, p_LUT0, p_LUT1);

	LinearIntegrateConvolution(noiseImage, vectorField_x, vectorField_y, outputImage, p_LUT0, p_LUT0);


	string w1name = "white noise image";
	namedWindow(w1name);
	imshow(w1name, noiseImage);
	waitKey(0);
	
	string w2name = "processed image";
	
	namedWindow(w2name);
	
	imshow(w2name, outputImage);
	waitKey(0);
	destroyAllWindows();
	
	delete[] p_LUT0;
	delete[] p_LUT1;
}
