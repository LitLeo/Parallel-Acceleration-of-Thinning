#include <iostream>
#include "Thinning.h"
#include "ErrorCode.h"
#include "Image.h"
#include "timer.h"

using namespace std;

#define LOOP 1

class Config {
public:
    float runTime;
    int blocksizeX;
    int blocksizeY;
    Config()
    {
        runTime = 10000.0f;
        blocksizeX = -1;
        blocksizeY = -1;
    }

    void update(float _runtime, int _blocksizex, int _blocksizey)
    {
        if(_runtime < runTime) {
            runTime = _runtime;
            blocksizeX = _blocksizex;
            blocksizeY = _blocksizey;
        }
    }

    void print()
    {
        cout << "runTime=" << runTime << ", blocksizeX=" << blocksizeX << ", blocksizeY=" << blocksizeY << endl;
    }
};

int main(int argc, char const **argv)
{
	// if(argc < 2)
	// {
	// 	cout << "Please input image!" << endl;
	// 	return 0;
	// }

    Config config[2];
	Image *inimg;
    ImageBasicOp::newImage(&inimg);
    int errcode;
    errcode = ImageBasicOp::readFromFile("256-256.bmp", inimg);
    // errcode = ImageBasicOp::readFromFile(argv[1], inimg);
    if (errcode != NO_ERROR) {
        cout << "error: " << errcode << endl;
        return 0; 
    }
    cout << "image: " << argv[1] << endl;
    for(int i = 0; i < inimg->width * inimg->height; i++) {
        if(inimg->imgData[i] != 0)
            inimg->imgData[i] = 255;
    }

    // 给每一个设备进行warmup
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    for(unsigned i = 0; i < deviceCount; ++i)
        warmup();

    Thinning thin_gpu;
    Image *outimg1;
    ImageBasicOp::newImage(&outimg1);
    ImageBasicOp::makeAtHost(outimg1, inimg->width, inimg->height);
    
    StartTimer();
    for (int i = 0; i < LOOP; i++) 
       thin_gpu.thinAhmedMultiGPU(inimg, outimg1);
    cout << "thinAhmed() time is " << GetTimer() / LOOP << " ms" << endl;
    /*config[0].update(runTime/LOOP, thin_gpu.DEF_BLOCK_X, thin_gpu.DEF_BLOCK_Y);*/
    ImageBasicOp::copyToHost(outimg1);
    ImageBasicOp::writeToFile("thinAhmed_outimg.bmp", outimg1); 
    ImageBasicOp::deleteImage(outimg1);

    // for (int dev = 0; dev < 1; ++dev) {
    //     cudaSetDevice(dev);
    //     cudaDeviceProp deviceProp;
    //     cudaGetDeviceProperties(&deviceProp, dev);
    //     cout << "Device " << dev << " " << deviceProp.name << endl; // , dev, deviceProp.name);

    //     for (int by = 0; by <= 32; by += 2)
    //     {
	   //      Thinning thin_gpu;
    //         if (by == 0)
    //             thin_gpu.DEF_BLOCK_Y = 1;
    //         else
    //             thin_gpu.DEF_BLOCK_Y = by;

    //         cout << "\nDEF_BLOCK_Y = " << thin_gpu.DEF_BLOCK_Y << " DEF_BLOCK_X = " << thin_gpu.DEF_BLOCK_X << endl;
    //         Image *outimg1;
    //         ImageBasicOp::newImage(&outimg1);
    //         ImageBasicOp::makeAtHost(outimg1, inimg->width, inimg->height);

    //         Image *outimg2;
    //         ImageBasicOp::newImage(&outimg2);
    //         ImageBasicOp::makeAtHost(outimg2, inimg->width, inimg->height);
            
    //         cudaEvent_t start, stop;
    //         float runTime;

    //         // 直接并行
    //         cudaEventCreate(&start);
    //         cudaEventCreate(&stop);
    //         cudaEventRecord(start, 0);
    //         for (int i = 0; i < LOOP; i++) 
    //            thin_gpu.thinAhmedMultiGPU(inimg, outimg1);
    //         cudaEventRecord(stop, 0);
    //         cudaEventSynchronize(stop);
    //         cudaEventElapsedTime(&runTime, start, stop);
    //         cout << "thinAhmed() time is " << (runTime) / LOOP << " ms" << endl;
    //         config[0].update(runTime/LOOP, thin_gpu.DEF_BLOCK_X, thin_gpu.DEF_BLOCK_Y);
    //         ImageBasicOp::copyToHost(outimg1);
    //         ImageBasicOp::writeToFile("thinAhmed_outimg.bmp", outimg1); 

    //         // Pattern 表法，Pattern位于 global 内存
    //         // cudaEventCreate(&start);
    //         // cudaEventCreate(&stop);
    //         // // float runTime;
    //         // cudaEventRecord(start, 0);
    //         // for (int i = 0; i < LOOP; i++) 
    //         //    thin_gpu.thinAhmedPt(inimg, outimg2);
    //         // cudaEventRecord(stop, 0);
    //         // cudaEventSynchronize(stop);
    //         // cudaEventElapsedTime(&runTime, start, stop);
    //         // cout << "thinAhmedPt() time is " << (runTime) / LOOP << " ms" << endl;
    //         // config[1].update(runTime/LOOP, thin_gpu.DEF_BLOCK_X, thin_gpu.DEF_BLOCK_Y);
    //         // ImageBasicOp::copyToHost(outimg2);
    //         // ImageBasicOp::writeToFile("thinAhmedPt_outimg.bmp", outimg2); 

    //         ImageBasicOp::deleteImage(outimg1);
    //         ImageBasicOp::deleteImage(outimg2);
    //     }
    // }

    // cout << "thinAhmed best config:" << endl;
    // config[0].print();
    // cout << "thinAhmedPt best config:" << endl;
    // config[1].print();

    cudaDeviceSynchronize();
    cudaDeviceReset();
    ImageBasicOp::deleteImage(inimg);  

	return 0;
}
