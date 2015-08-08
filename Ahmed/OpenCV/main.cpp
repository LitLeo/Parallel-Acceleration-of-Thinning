#include "rirt.h"
#include <iostream>
using namespace std;

int main()
{
	Mat inimg = imread("512-512.bmp", CV_8UC1);
	// printMat(inimg);
	// return 0;
	Mat outimg(inimg.size(), CV_8UC1);
	rirt(inimg, outimg);
	imshow("outimg", outimg);
	imwrite("outimg.bmp", outimg);
	waitKey();
	return 0;
}