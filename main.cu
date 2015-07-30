#include <iostream>
#include "Thinning_gpu.h"
#include "Thinning_gpu_pt.h"
#include "Thinning_gpu_pt_con.h"
#include "ErrorCode.h"
#include "Image.h"
using namespace std;

int main(int argc, char const **argv)
{
	if(argc < 2)
	{
		cout << "Please input image!" << endl;
		return 0;
	}
	Thinning_gpu thin_gpu;
    Thinning_gpu_pt thin_gpu_pt;
    Thinning_gpu_pt_con thin_gpu_pt_con;
	Image *inimg;
    ImageBasicOp::newImage(&inimg);
    int errcode;
    errcode = ImageBasicOp::readFromFile(argv[1], inimg);
    if (errcode != NO_ERROR) 
    {
        cout << "error: " << errcode << endl;
        return 0; 
    }
    int piexlcont = 0;
    for(int i = 0; i < inimg->width * inimg->height; i++)
       if(inimg->imgData[i] == 255)
          piexlcont ++;
    cout << "piexlcont = " << piexlcont << endl; 
    Image *outimg;
    ImageBasicOp::newImage(&outimg);
    ImageBasicOp::makeAtHost(outimg, inimg->width, inimg->height);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float runTime;
    cudaEventRecord(start, 0);
    for (int i = 0; i < 100; i++) 
        thin_gpu.thinGuoHall(inimg, outimg);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runTime, start, stop);

	cout << "thinGuoHall() time is " << (runTime) / 100 << " ms" << endl;
    ImageBasicOp::copyToHost(outimg);

    // cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // float runTime;
    cudaEventRecord(start, 0);
    for (int i = 0; i < 100; i++) {
        thin_gpu_pt.thinPattern(inimg, outimg);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runTime, start, stop);

    cout << "thinPattern() time is " << (runTime) / 100 << " ms" << endl;
    ImageBasicOp::copyToHost(outimg);

        // cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // float runTime;
    cudaEventRecord(start, 0);
    for (int i = 0; i < 100; i++) {
        thin_gpu_pt_con.thinPatternCon(inimg, outimg);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runTime, start, stop);

    cout << "thinPatternCon() time is " << (runTime) / 100 << " ms" << endl;
    ImageBasicOp::copyToHost(outimg);
    
    ImageBasicOp::writeToFile("thinningOut.bmp", outimg);  
    ImageBasicOp::deleteImage(inimg);
    ImageBasicOp::deleteImage(outimg);

	return 0;
}
