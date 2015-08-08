#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#define HIGH 1
#define LOW 0
#define isHigh1(x) (((x) == HIGH) ? 1 : 0)
#define isHigh2(x1, x2) ( (x1) == HIGH && (x2) == HIGH ? 1 : 0)
#define isHigh3(x1, x2, x3) ( (x1) == HIGH && (x2) == HIGH && (x3) == HIGH ? 1 : 0)
#define isHigh4(x1, x2, x3, x4) ( (x1) == HIGH && (x2) == HIGH && (x3) == HIGH && (x4) == HIGH ? 1 : 0)
#define isHigh5(x1, x2, x3, x4, x5) ( (x1) == HIGH && (x2) == HIGH && (x3) == HIGH && (x4) == HIGH && (x5) == HIGH ? 1 : 0)
#define isHigh6(x1, x2, x3, x4, x5, x6) ( (x1) == HIGH && (x2) == HIGH && (x3) == HIGH && (x4) == HIGH && (x5) == HIGH && (x6) == HIGH ? 1 : 0)
#define isHigh7(x1, x2, x3, x4, x5, x6, x7) ( (x1) == HIGH && (x2) == HIGH && (x3) == HIGH && (x4) == HIGH && (x5) == HIGH && (x6) == HIGH && (x7) == HIGH ? 1 : 0)

#define isLow1(x) (((x) != HIGH) ? 1 : 0)
#define isLow2(x1, x2) ( (x1) != HIGH && (x2) != HIGH ? 1 : 0)
#define isLow3(x1, x2, x3) ( (x1) != HIGH && (x2) != HIGH && (x3) != HIGH ? 1 : 0)
#define isLow4(x1, x2, x3, x4) ( (x1) != HIGH && (x2) != HIGH && (x3) != HIGH && (x4) != HIGH ? 1 : 0)
#define isLow5(x1, x2, x3, x4, x5) ( (x1) != HIGH && (x2) != HIGH && (x3) != HIGH && (x4) != HIGH && (x5) != HIGH ? 1 : 0)
#define isLow6(x1, x2, x3, x4, x5, x6) ( (x1) != HIGH && (x2) != HIGH && (x3) != HIGH && (x4) != HIGH && (x5) != HIGH && (x6) != HIGH ? 1 : 0)
#define isLow7(x1, x2, x3, x4, x5, x6, x7) ( (x1) != HIGH && (x2) != HIGH && (x3) != HIGH && (x4) != HIGH && (x5) != HIGH && (x6) != HIGH && (x7) != HIGH ? 1 : 0)
#define isLow8(x1, x2, x3, x4, x5, x6, x7, x8) ( (x1) != HIGH && (x2) != HIGH && (x3) != HIGH && (x4) != HIGH && (x5) != HIGH && (x6) != HIGH && (x7) != HIGH && (x8) != HIGH ? 1 : 0)

void printMat(Mat im)
{
	//im /= 255;
	for (int i = 0; i < im.rows; i++)
	{
		for (int j = 0; j < im.cols; j++)
		{
			cout << (int)im.at<uchar>(i,j);
		}
		cout << endl;
	}

	cout << endl << endl << endl;
}

// x1 x2 x3
// x4 x  x5
// x6 x7 x8


// x 0 x			x 0 x
// 1 w 1  stop,else 1 1 1  delete w and stop
// 1 1 1			1 w 1
// x 0 x			x 0 x
// 
// 0 0 0	0 0 0			1 0 0	0 0 1
// 0 w 0 or 0 w 0 stop, else1 1 0 or0 1 1 stop 
// 1 1 0	0 1 1 			0 w 0	0 w 0
// 1 0 0	0 0 1           0 0 0	0 0 0	

// x 1 1 x		x 1 1 x
// 0 w 1 0 stop 0 1 w 0 delete
// x 1 1 x		x 1 1 x
//
// 0 0 0 0		0 0 1 1				0 0 0 0		1 1 0 0 
// 0 w 1 0 or	0 w 1 0 stop else   0 1 w 0 or  0 1 w 0 stop
// 0 0 1 1		0 0 0 0				1 1 0 0		0 0 0 0

// Ï¸»¯º¯Êý
int rirt(Mat& inimg, Mat& outimg)
{
	Mat tempimg;
	inimg.copyTo(outimg);
	// inimg.copyTo(tempimg);
	
	outimg /= 255;
	// printMat(outimg);
	int rows = inimg.rows;
	int cols = inimg.cols;
	int changecount = 1;

	while (changecount != 0)
	{
		outimg.copyTo(tempimg);
		// printMat(outimg);
		changecount = 0;
		for (int r = 2; r < rows - 2; r++) {
			for (int c = 2; c < cols - 2; c++) {
				uchar x1 = tempimg.at<uchar>(r - 1, c - 1);
				uchar x2 = tempimg.at<uchar>(r - 1, c);
				uchar x3 = tempimg.at<uchar>(r - 1, c + 1);
				uchar x4 = tempimg.at<uchar>(r, c - 1);
				uchar x5 = tempimg.at<uchar>(r, c + 1);
				uchar x6 = tempimg.at<uchar>(r + 1, c - 1);
				uchar x7 = tempimg.at<uchar>(r + 1, c);
				uchar x8 = tempimg.at<uchar>(r + 1, c + 1);
				uchar x9,x10,x11;
				// if w is high
				if(isHigh1(outimg.at<uchar>(r,c))) {
					if(isHigh1(x7)) {
						x9 = tempimg.at<uchar>(r + 2, c - 1);
						x10 = tempimg.at<uchar>(r + 2, c);
						x11 = tempimg.at<uchar>(r + 2, c + 1);

						if (isHigh5(x4,x5,x6,x7,x8) && isLow2(x2,x10) ||
							isHigh3(x6,x7,x9) && isLow8(x1,x2,x3,x4,x5,x8,x10,x11) ||
							isHigh3(x7,x8,x11) && isLow8(x1,x2,x3,x4,x5,x6,x9,x10)) {
								continue ;
						} 
					} 
					if(isHigh1(x2)) {
						// w is down
						x9 = tempimg.at<uchar>(r - 2, c - 1);
						x10 = tempimg.at<uchar>(r - 2, c);
						x11 = tempimg.at<uchar>(r - 2, c + 1);

						if (isHigh5(x1,x2,x3,x4,x5) && isLow2(x7,x10)){
							outimg.at<uchar>(r,c) = LOW;
							changecount ++;
							continue ;
							
						} else if (isHigh3(x1,x2,x9) && isLow8(x3,x4,x5,x6,x7,x8,x10,x11) ||
								   isHigh3(x2,x3,x11) && isLow8(x1,x4,x5,x6,x7,x8,x9,x10)) {
							continue ;
						}
					}
					if(isHigh1(x5)) {
						x9 = tempimg.at<uchar>(r - 1, c + 2);
						x10 = tempimg.at<uchar>(r, c + 2);
						x11 = tempimg.at<uchar>(r + 1, c + 2);

						if (isHigh5(x2,x3,x5,x7,x8) && isLow2(x4,x10) ||
							isHigh3(x5,x8,x11) && isLow8(x1,x2,x3,x4,x6,x7,x9,x10) ||
							isHigh3(x3,x5,x9) && isLow8(x1,x2,x4,x6,x7,x8,x10,x11)) {
							continue ;
						}
					}
					if(isHigh1(x4)){
						x9 = tempimg.at<uchar>(r - 1, c - 2);
						x10 = tempimg.at<uchar>(r, c - 2);
						x11 = tempimg.at<uchar>(r + 1, c - 2);
						if (isHigh5(x1,x2,x4,x6,x7) && isLow2(x5,x10)){
							outimg.at<uchar>(r,c) = LOW;
							changecount ++;
							continue ;
						} else if (isHigh3(x4,x6,x11) && isLow8(x1,x2,x3,x5,x7,x8,x9,x10) ||
								   isHigh3(x1,x4,x9) && isLow8(x2,x3,x5,x6,x7,x8,x10,x11)) {
							continue ;
						}
					}
					
					if (isHigh4(x1, x4, x6, x7) && isLow2(x3, x5) || // 1
						isHigh4(x1, x2, x4, x6) && isLow2(x5, x8) || // 2
						isHigh4(x1, x2, x3, x5) && isLow2(x6, x7) || // 3
						isHigh4(x1, x2, x3, x4) && isLow2(x7, x8) || // 4
						isLow4(x3, x5, x7, x8) && isHigh2(x1, x4) || // 5
						isLow4(x5, x6, x7, x8) && isHigh2(x1, x2) || // 6
						isLow1(x5) && isHigh7(x1, x2, x3, x4, x6, x7, x8) || // 7
						isLow1(x7) && isHigh7(x1, x2, x3, x4, x5, x6, x8) || // 8
						isLow4(x2, x3, x5, x8) && isHigh2(x4, x6) || // 9
						isLow4(x1, x2, x3, x5) && isHigh2(x6, x7) || // 10
						isLow4(x4, x6, x7, x8) && isHigh2(x2, x3) || // 11
						isLow4(x1, x4, x6, x7) && isHigh2(x3, x5) || // 12
						isLow4(x1, x2, x3, x4) && isHigh2(x7, x8) || // 13
						isLow4(x1, x2, x4, x6) && isHigh2(x5, x8) || // 14
						isLow1(x4) && isHigh7(x1, x2, x3, x5, x6, x7, x8) || // 15
						isLow1(x2) && isHigh7(x1, x3, x4, x5, x6, x7, x8) || // 16
						isHigh4(x3, x5, x7, x8) && isLow2(x1, x4) || // 17
						isHigh4(x2, x3, x5, x8) && isLow2(x4, x6) || // 18
						isHigh4(x5, x6, x7, x8) && isLow2(x1, x2) || // 19
						isHigh4(x4, x6, x7, x8) && isLow2(x2, x3) // 20
						) {
							outimg.at<uchar>(r,c) = LOW;
							changecount ++;
					}
				}
					
			}
		}	
	}
	outimg *= 255;
	return 0;
}

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