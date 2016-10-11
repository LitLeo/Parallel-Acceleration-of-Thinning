// Thinning.cu
// 实现二值图像的细化算法

#include "Thinning.h"
#include <iostream>
#include <stdio.h>
using namespace std;

#define HIGH 255
#define LOW 0
#define isHigh1(x) (((x) != LOW) ? 1 : 0)
#define isHigh2(x1, x2) ( (x1) != LOW && (x2) != LOW ? 1 : 0)
#define isHigh3(x1, x2, x3) ( (x1) != LOW && (x2) != LOW && (x3) != LOW ? 1 : 0)
#define isHigh4(x1, x2, x3, x4) ( (x1) != LOW && (x2) != LOW && (x3) != LOW && (x4) != LOW ? 1 : 0)
#define isHigh5(x1, x2, x3, x4, x5) ( (x1) != LOW && (x2) != LOW && (x3) != LOW && (x4) != LOW && (x5) != LOW ? 1 : 0)
#define isHigh6(x1, x2, x3, x4, x5, x6) ( (x1) != LOW && (x2) != LOW && (x3) != LOW && (x4) != LOW && (x5) != LOW && (x6) != LOW ? 1 : 0)
#define isHigh7(x1, x2, x3, x4, x5, x6, x7) ( (x1) != LOW && (x2) != LOW && (x3) != LOW && (x4) != LOW && (x5) != LOW && (x6) != LOW && (x7) != LOW ? 1 : 0)

#define isLow1(x) (((x) != HIGH) ? 1 : 0)
#define isLow2(x1, x2) ( (x1) != HIGH && (x2) != HIGH ? 1 : 0)
#define isLow3(x1, x2, x3) ( (x1) != HIGH && (x2) != HIGH && (x3) != HIGH ? 1 : 0)
#define isLow4(x1, x2, x3, x4) ( (x1) != HIGH && (x2) != HIGH && (x3) != HIGH && (x4) != HIGH ? 1 : 0)
#define isLow5(x1, x2, x3, x4, x5) ( (x1) != HIGH && (x2) != HIGH && (x3) != HIGH && (x4) != HIGH && (x5) != HIGH ? 1 : 0)
#define isLow6(x1, x2, x3, x4, x5, x6) ( (x1) != HIGH && (x2) != HIGH && (x3) != HIGH && (x4) != HIGH && (x5) != HIGH && (x6) != HIGH ? 1 : 0)
#define isLow7(x1, x2, x3, x4, x5, x6, x7) ( (x1) != HIGH && (x2) != HIGH && (x3) != HIGH && (x4) != HIGH && (x5) != HIGH && (x6) != HIGH && (x7) != HIGH ? 1 : 0)
#define isLow8(x1, x2, x3, x4, x5, x6, x7, x8) ( (x1) != HIGH && (x2) != HIGH && (x3) != HIGH && (x4) != HIGH && (x5) != HIGH && (x6) != HIGH && (x7) != HIGH && (x8) != HIGH ? 1 : 0)

#define LUT_SIZE 256

static __global__ void _thinAhmedKer(ImageCuda tempimg, ImageCuda outimg,
        unsigned char* input_flag, unsigned char* output_flag, int *devchangecount)
{
    // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    // 两边各有两个点不处理。
    if (c >= outimg.imgMeta.width - 2 || 
         r >= outimg.imgMeta.height - 2 || c < 2 || r < 2)
        return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    // 从左上角第二行第二列开始计算。
    int curpos = (r) * outimg.pitchBytes + c ;
    int flagpos = r * tempimg.imgMeta.width + c;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;
    unsigned char x1, x2, x3, x4, x5, x6, x7, x8;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (isHigh1(*outptr)) {
        // 如果在滚动数组中，发现上一轮发生了删点，说明此时实质上已经是Low了
        // 因此刷新，并返回
        if (isLow1(tempimg.imgMeta.imgData[curpos])){
            outimg.imgMeta.imgData[curpos] = LOW;
            // 因为该修改不属于本次迭代的修改，所以不修改changeCount
            return;
        }
        // 如果上一轮的计算中，没有标志此像素点需要重新计算权值,
        // 虽然像素值是High，仍然略过，因为本次迭代中已知它不会改变（被删）
        // 也不会影响它的八邻域
        if (isLow1(input_flag[flagpos])){
            return;
        }
        x1 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 1];
        x2 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes];
        x3 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 1];
        x4 = tempimg.imgMeta.imgData[curpos - 1];
        x5 = tempimg.imgMeta.imgData[curpos + 1];
        x6 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 1];
        x7 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes];
        x8 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 1];
        unsigned char x9,x10,x11;
        if(isHigh1(x7)) {
            x9 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 + 1];

            if (isHigh5(x4,x5,x6,x7,x8) && isLow2(x2,x10) ||
                isHigh3(x6,x7,x9) && isLow8(x1,x2,x3,x4,x5,x8,x10,x11) ||
                isHigh3(x7,x8,x11) && isLow8(x1,x2,x3,x4,x5,x6,x9,x10))
                    return ;
        } 
        if(isHigh1(x2)) {
            // w is down
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 + 1];

            if (isHigh5(x1,x2,x3,x4,x5) && isLow2(x7,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
                output_flag[flagpos - 1] = HIGH;
                output_flag[flagpos + 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
                return ;
            } else if (isHigh3(x1,x2,x9) && isLow8(x3,x4,x5,x6,x7,x8,x10,x11) ||
                       isHigh3(x2,x3,x11) && isLow8(x1,x4,x5,x6,x7,x8,x9,x10)) 
                return ;
        }
        if(isHigh1(x5)) {
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 2];
            x10 = tempimg.imgMeta.imgData[curpos + 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 2];

            if (isHigh5(x2,x3,x5,x7,x8) && isLow2(x4,x10) ||
                isHigh3(x5,x8,x11) && isLow8(x1,x2,x3,x4,x6,x7,x9,x10) ||
                isHigh3(x3,x5,x9) && isLow8(x1,x2,x4,x6,x7,x8,x10,x11))
                return ;
        }
        if(isHigh1(x4)){
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 2];
            x10 = tempimg.imgMeta.imgData[curpos - 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 2];
            if (isHigh5(x1,x2,x4,x6,x7) && isLow2(x5,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
                output_flag[flagpos - 1] = HIGH;
                output_flag[flagpos + 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
                return ;
            } else if (isHigh3(x4,x6,x11) && isLow8(x1,x2,x3,x5,x7,x8,x9,x10) ||
                       isHigh3(x1,x4,x9) && isLow8(x2,x3,x5,x6,x7,x8,x10,x11)) 
                return ;
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
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;

                output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
                output_flag[flagpos - 1] = HIGH;
                output_flag[flagpos + 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
        }       
    }
}

// 直接并行化
// 线程数，处理多少个点有多少线程数
__host__ int Thinning::thinAhmed(Image *inimg, Image *outimg)
{
    // 局部变量，错误码。
    int errcode;  
    cudaError_t cudaerrcode; 

    // 检查输入图像，输出图像是否为空。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 声明所有中间变量并初始化为空。
    Image *tempimg = NULL;
    unsigned char *flag1 = NULL;
    unsigned char* flag2 = NULL;
    int *devchangecount = NULL;

    // 记录细化点数的变量，位于 host 端。
    int changeCount, iteration;

    // 记录细化点数的变量，位于 device 端。并为其申请空间。
    cudaerrcode = cudaMalloc((void **)&devchangecount, sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 生成暂存图像。
    errcode = ImageBasicOp::newImage(&tempimg);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                inimg->height);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 将输入图像 inimg 完全拷贝到输出图像 outimg ，并将 outimg 拷贝到 
    // device 端。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg, outimg);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 提取输出图像
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 提取暂存图像
    ImageCuda tempsubimgCud;
    errcode = ImageBasicOp::roiSubImage(tempimg, &tempsubimgCud);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 生成flag标志数组
    errcode = cudaMalloc((void**)(&flag1), sizeof(unsigned char) *
            inimg->width * inimg->height);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = cudaMalloc((void**)(&flag2), sizeof(unsigned char) *
            outimg->width * outimg->height);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    /*gridsize.x = 1;*/
    /*gridsize.y = 1;//(outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;*/
    // 赋值为 1，以便开始第一次迭代。
    changeCount = 1;

    // 迭代次数，为了计算滚动的状态
    iteration = 0;
    
    // copy ouimg to tempimg 
     cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                  outimg->imgData, outsubimgCud.deviceId, 
                                  outsubimgCud.pitchBytes * outimg->height);
    
    if (cudaerrcode != cudaSuccess) {
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 初始的输入flag为全HIGH，保证所有点都会计算第一次
    cudaerrcode = cudaMemset(flag1, HIGH, sizeof(unsigned char) * inimg->width * inimg->height);
     if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
     }

     // 第一轮的输出flag初值是全LOW
    cudaerrcode = cudaMemset(flag2, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
     if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
     }
    
    // 开始迭代，当不可再被细化，即记录细化点数的变量 changeCount 的值为 0 时，
    // 停止迭代。 
    while (changeCount > 0) {
        // 将 host 端的变量赋值为 0 ，并将值拷贝到 device 端的 devchangecount。
        changeCount = 0;
        cudaerrcode = cudaMemcpy(devchangecount, &changeCount, sizeof (int),
                                 cudaMemcpyHostToDevice);
        if (cudaerrcode != cudaSuccess) {
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

        // 如果已经进行了奇数次迭代，那么现在源图像是在outimg上，输出图像应该在tempsubimgCud上
        // flag标志数组此刻使用flag1作为输入标志，flag2作为输出标志
        if (iteration & 1){
            // 调用核函数，开始第一步细化操作。
            _thinAhmedKer<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud,
                    flag2, flag1, devchangecount, dev_lut);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }    
            // 每次需要清空下次作为输出flag的数组（即本次迭代的输入flag)
            // 因为每次迭代需要刷新的都是新的“最外一圈”，持续变化中
            cudaerrcode = cudaMemset(flag2, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
            if (cudaerrcode != cudaSuccess) {
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }
        }else{
            // 调用核函数，开始第一步细化操作。
            _thinAhmedKer<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud,
                    flag1, flag2, devchangecount, dev_lut);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }    
            // 每次需要清空下次作为输出flag的数组（即本次迭代的输入flag)
            // 因为每次迭代需要刷新的都是新的“最外一圈”，持续变化中
            cudaerrcode = cudaMemset(flag1, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
            if (cudaerrcode != cudaSuccess) {
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }
        }
        
        // 将位于 device 端的 devchangecount 拷贝到 host 端上的 changeCount 
        // 变量，进行迭代判断。
        cudaerrcode = cudaMemcpy(&changeCount, devchangecount, sizeof (int),
                                 cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

        iteration ++;
   }
    // 如果进行了偶数次迭代,那么outimg现在在tempsubimgCud上，
    // 把tempsubimgCud的内容拷贝到outimg指针的设备端内存中
    if (!(iteration & 1)){
         cudaerrcode = cudaMemcpyPeer(outimg->imgData, outsubimgCud.deviceId, 
                                      tempimg->imgData, tempsubimgCud.deviceId, 
                                      tempsubimgCud.pitchBytes * tempimg->height);
    }
    
    // 细化结束后释放申请的变量空间。
    cudaFree(devchangecount);
    ImageBasicOp::deleteImage(tempimg);
    cudaFree(flag1);
    cudaFree(flag2);
    
    return NO_ERROR;
}

// GPU版本2，优化分支，使用Pattern表法，Pattern表位于global内存中
static __global__ void _thinAhmedPtKer(ImageCuda tempimg, ImageCuda outimg,
        unsigned char* input_flag, unsigned char* output_flag, int *devchangecount, unsigned char *dev_lut)
{
    // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    // 两边各有两个点不处理。
    if (c >= outimg.imgMeta.width - 2 || 
         r >= outimg.imgMeta.height - 2 || c < 2 || r < 2)
        return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    // 从左上角第二行第二列开始计算。
    int curpos = (r) * outimg.pitchBytes + c ;
    int flagpos = r * tempimg.imgMeta.width + c;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;
    unsigned char x1, x2, x3, x4, x5, x6, x7, x8;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (isHigh1(*outptr)) {
        // 如果在滚动数组中，发现上一轮发生了删点，说明此时实质上已经是Low了
        // 因此刷新，并返回
        if (isLow1(tempimg.imgMeta.imgData[curpos])){
            outimg.imgMeta.imgData[curpos] = LOW;
            // 因为该修改不属于本次迭代的修改，所以不修改changeCount
            return;
        }
        // 如果上一轮的计算中，没有标志此像素点需要重新计算权值,
        // 虽然像素值是High，仍然略过，因为本次迭代中已知它不会改变（被删）
        // 也不会影响它的八邻域
        if (isLow1(input_flag[flagpos])){
            return;
        }
        x1 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 1];
        x2 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes];
        x3 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 1];
        x4 = tempimg.imgMeta.imgData[curpos - 1];
        x5 = tempimg.imgMeta.imgData[curpos + 1];
        x6 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 1];
        x7 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes];
        x8 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 1];
        unsigned char x9,x10,x11;
        if(isHigh1(x7)) {
            x9 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 + 1];

            if (isHigh5(x4,x5,x6,x7,x8) && isLow2(x2,x10) ||
                isHigh3(x6,x7,x9) && isLow8(x1,x2,x3,x4,x5,x8,x10,x11) ||
                isHigh3(x7,x8,x11) && isLow8(x1,x2,x3,x4,x5,x6,x9,x10))
                    return ;
        } 
        if(isHigh1(x2)) {
            // w is down
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 + 1];

            if (isHigh5(x1,x2,x3,x4,x5) && isLow2(x7,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
                output_flag[flagpos - 1] = HIGH;
                output_flag[flagpos + 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
                return ;
            } else if (isHigh3(x1,x2,x9) && isLow8(x3,x4,x5,x6,x7,x8,x10,x11) ||
                       isHigh3(x2,x3,x11) && isLow8(x1,x4,x5,x6,x7,x8,x9,x10)) 
                return ;
        }
        if(isHigh1(x5)) {
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 2];
            x10 = tempimg.imgMeta.imgData[curpos + 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 2];

            if (isHigh5(x2,x3,x5,x7,x8) && isLow2(x4,x10) ||
                isHigh3(x5,x8,x11) && isLow8(x1,x2,x3,x4,x6,x7,x9,x10) ||
                isHigh3(x3,x5,x9) && isLow8(x1,x2,x4,x6,x7,x8,x10,x11))
                return ;
        }
        if(isHigh1(x4)){
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 2];
            x10 = tempimg.imgMeta.imgData[curpos - 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 2];
            if (isHigh5(x1,x2,x4,x6,x7) && isLow2(x5,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
                output_flag[flagpos - 1] = HIGH;
                output_flag[flagpos + 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
                return ;
            } else if (isHigh3(x4,x6,x11) && isLow8(x1,x2,x3,x5,x7,x8,x9,x10) ||
                       isHigh3(x1,x4,x9) && isLow8(x2,x3,x5,x6,x7,x8,x10,x11)) 
                return ;
        }

        // 1   2   4
        // 8       16
        // 32  64  128
        unsigned char index = isHigh1(x1) * 1 + isHigh1(x2) * 2 + isHigh1(x3) * 4 + isHigh1(x4) * 8 +
                  isHigh1(x5) * 16 + isHigh1(x6) * 32 + isHigh1(x7) * 64 + isHigh1(x8) * 128;
        
        if (dev_lut[index] == 1) {
            outimg.imgMeta.imgData[curpos] = LOW;
            *devchangecount = 1;
            output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
            output_flag[flagpos - outimg.imgMeta.width] = HIGH;
            output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
            output_flag[flagpos - 1] = HIGH;
            output_flag[flagpos + 1] = HIGH;
            output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
            output_flag[flagpos + outimg.imgMeta.width] = HIGH;
            output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
        }       
    }
}

__host__ int Thinning::thinAhmedPt(Image *inimg, Image *outimg)
{
 // 局部变量，错误码。
    int errcode;  
    cudaError_t cudaerrcode; 

    // 检查输入图像，输出图像是否为空。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 声明所有中间变量并初始化为空。
    Image *tempimg = NULL;
    unsigned char *flag1 = NULL;
    unsigned char *flag2 = NULL;
    int *devchangecount = NULL;
    unsigned char *dev_lut;
    unsigned char lut[256] = { 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 
    1, 1, 0, 0, 1, 1, 0, 0 };
    
    cudaerrcode = cudaMalloc((void **)&dev_lut, sizeof (unsigned char) * 256);
    if (cudaerrcode != cudaSuccess) 
        return CUDA_ERROR;

    cudaerrcode = cudaMemcpy(dev_lut, lut, sizeof(unsigned char) * 256, cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess) 
        return CUDA_ERROR;

    // 记录细化点数的变量，迭代次数，位于 host 端。
    int changeCount, iteration;

    // 记录细化点数的变量，位于 device 端。并为其申请空间。
    cudaerrcode = cudaMalloc((void **)&devchangecount, sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 生成暂存图像。
    errcode = ImageBasicOp::newImage(&tempimg);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                inimg->height);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 将输入图像 inimg 完全拷贝到输出图像 outimg ，并将 outimg 拷贝到 
    // device 端。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg, outimg);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 提取输出图像
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 提取暂存图像
    ImageCuda tempsubimgCud;
    errcode = ImageBasicOp::roiSubImage(tempimg, &tempsubimgCud);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 生成flag标志数组
    errcode = cudaMalloc((void**)(&flag1), sizeof(unsigned char) *
            inimg->width * inimg->height);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = cudaMalloc((void**)(&flag2), sizeof(unsigned char) *
            outimg->width * outimg->height);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 赋值为 1，以便开始第一次迭代。
    changeCount = 1;

    // 迭代次数，为了计算滚动的状态
    iteration = 0;

    // copy ouimg to tempimg 
     cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                  outimg->imgData, outsubimgCud.deviceId, 
                                  outsubimgCud.pitchBytes * outimg->height);
    
    if (cudaerrcode != cudaSuccess) {
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 初始的输入flag为全HIGH，保证所有点都会计算第一次
    cudaerrcode = cudaMemset(flag1, HIGH, sizeof(unsigned char) * inimg->width * inimg->height);
     if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
     }

     // 第一轮的输出flag初值是全LOW
    cudaerrcode = cudaMemset(flag2, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
     if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
     }

    // 开始迭代，当不可再被细化，即记录细化点数的变量 changeCount 的值为 0 时，
    // 停止迭代。
    while (changeCount > 0) {
        // 将 host 端的变量赋值为 0 ，并将值拷贝到 device 端的 devchangecount。
        changeCount = 0;
        cudaerrcode = cudaMemcpy(devchangecount, &changeCount, sizeof (int),
                                 cudaMemcpyHostToDevice);
        if (cudaerrcode != cudaSuccess) {
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

        // 如果已经进行了奇数次迭代，那么现在源图像是在outimg上，输出图像应该在tempsubimgCud上
        // flag标志数组此刻使用flag1作为输入标志，flag2作为输出标志
        if (iteration & 1){
            // 调用核函数，开始第一步细化操作。
            _thinAhmedPtKer<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud,
                    flag2, flag1, devchangecount, dev_lut);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }    
            // 每次需要清空下次作为输出flag的数组（即本次迭代的输入flag)
            // 因为每次迭代需要刷新的都是新的“最外一圈”，持续变化中
            cudaerrcode = cudaMemset(flag2, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
            if (cudaerrcode != cudaSuccess) {
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }
        }else{
            // 调用核函数，开始第一步细化操作。
            _thinAhmedPtKer<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud,
                    flag1, flag2, devchangecount, dev_lut);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }    
            // 每次需要清空下次作为输出flag的数组（即本次迭代的输入flag)
            // 因为每次迭代需要刷新的都是新的“最外一圈”，持续变化中
            cudaerrcode = cudaMemset(flag1, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
            if (cudaerrcode != cudaSuccess) {
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }
        }
        
        // 将位于 device 端的 devchangecount 拷贝到 host 端上的 changeCount 
        // 变量，进行迭代判断。
        cudaerrcode = cudaMemcpy(&changeCount, devchangecount, sizeof (int),
                                 cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

        iteration ++;
   }
    // 如果进行了偶数次迭代,那么outimg现在在tempsubimgCud上，
    // 把tempsubimgCud的内容拷贝到outimg指针的设备端内存中
    if (!(iteration & 1)){
         cudaerrcode = cudaMemcpyPeer(outimg->imgData, outsubimgCud.deviceId, 
                                      tempimg->imgData, tempsubimgCud.deviceId, 
                                      tempsubimgCud.pitchBytes * tempimg->height);
    }

    // 细化结束后释放申请的变量空间。
    cudaFree(devchangecount);
    ImageBasicOp::deleteImage(tempimg);
    cudaFree(flag1);
    cudaFree(flag2);

    return NO_ERROR;
}

// GPU版本2，优化分支，使用Pattern表法，Pattern表位于shared内存中
static __global__ void _thinAhmedPtSharedKer(ImageCuda tempimg, ImageCuda outimg,
        unsigned char* input_flag, unsigned char* output_flag, int *devchangecount, unsigned char *dev_lut)
{
    // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    // 两边各有两个点不处理。
    if (c >= outimg.imgMeta.width - 2 || 
         r >= outimg.imgMeta.height - 2 || c < 2 || r < 2)
        return;

    // 申请共享内存，lut size = 256
    __shared__ unsigned char shared_lut[LUT_SIZE]; 
    int shared_index = threadIdx.y * blockDim.y + threadIdx.x;
    if (shared_index < 256) 
        shared_lut[shared_index] = dev_lut[shared_index];
    __syncthreads();

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    // 从左上角第二行第二列开始计算。
    int curpos = (r) * outimg.pitchBytes + c ;
    int flagpos = r * tempimg.imgMeta.width + c;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;
    unsigned char x1, x2, x3, x4, x5, x6, x7, x8;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (isHigh1(*outptr)) {
        // 如果在滚动数组中，发现上一轮发生了删点，说明此时实质上已经是Low了
        // 因此刷新，并返回
        if (isLow1(tempimg.imgMeta.imgData[curpos])){
            outimg.imgMeta.imgData[curpos] = LOW;
            // 因为该修改不属于本次迭代的修改，所以不修改changeCount
            return;
        }
        // 如果上一轮的计算中，没有标志此像素点需要重新计算权值,
        // 虽然像素值是High，仍然略过，因为本次迭代中已知它不会改变（被删）
        // 也不会影响它的八邻域
        if (isLow1(input_flag[flagpos])){
            return;
        }
        x1 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 1];
        x2 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes];
        x3 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 1];
        x4 = tempimg.imgMeta.imgData[curpos - 1];
        x5 = tempimg.imgMeta.imgData[curpos + 1];
        x6 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 1];
        x7 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes];
        x8 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 1];
        unsigned char x9,x10,x11;
        if(isHigh1(x7)) {
            x9 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 + 1];

            if (isHigh5(x4,x5,x6,x7,x8) && isLow2(x2,x10) ||
                isHigh3(x6,x7,x9) && isLow8(x1,x2,x3,x4,x5,x8,x10,x11) ||
                isHigh3(x7,x8,x11) && isLow8(x1,x2,x3,x4,x5,x6,x9,x10))
                    return ;
        } 
        if(isHigh1(x2)) {
            // w is down
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 + 1];

            if (isHigh5(x1,x2,x3,x4,x5) && isLow2(x7,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
                output_flag[flagpos - 1] = HIGH;
                output_flag[flagpos + 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
                return ;
            } else if (isHigh3(x1,x2,x9) && isLow8(x3,x4,x5,x6,x7,x8,x10,x11) ||
                       isHigh3(x2,x3,x11) && isLow8(x1,x4,x5,x6,x7,x8,x9,x10)) 
                return ;
        }
        if(isHigh1(x5)) {
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 2];
            x10 = tempimg.imgMeta.imgData[curpos + 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 2];

            if (isHigh5(x2,x3,x5,x7,x8) && isLow2(x4,x10) ||
                isHigh3(x5,x8,x11) && isLow8(x1,x2,x3,x4,x6,x7,x9,x10) ||
                isHigh3(x3,x5,x9) && isLow8(x1,x2,x4,x6,x7,x8,x10,x11))
                return ;
        }
        if(isHigh1(x4)){
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 2];
            x10 = tempimg.imgMeta.imgData[curpos - 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 2];
            if (isHigh5(x1,x2,x4,x6,x7) && isLow2(x5,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
                output_flag[flagpos - 1] = HIGH;
                output_flag[flagpos + 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
                return ;
            } else if (isHigh3(x4,x6,x11) && isLow8(x1,x2,x3,x5,x7,x8,x9,x10) ||
                       isHigh3(x1,x4,x9) && isLow8(x2,x3,x5,x6,x7,x8,x10,x11)) 
                return ;
        }

        // 1   2   4
        // 8       16
        // 32  64  128
        unsigned char index = isHigh1(x1) * 1 + isHigh1(x2) * 2 + isHigh1(x3) * 4 + isHigh1(x4) * 8 +
                              isHigh1(x5) * 16 + isHigh1(x6) * 32 + isHigh1(x7) * 64 + isHigh1(x8) * 128;
        
        if (shared_lut[index] == 1) {
            outimg.imgMeta.imgData[curpos] = LOW;
            *devchangecount = 1;
            output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
            output_flag[flagpos - outimg.imgMeta.width] = HIGH;
            output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
            output_flag[flagpos - 1] = HIGH;
            output_flag[flagpos + 1] = HIGH;
            output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
            output_flag[flagpos + outimg.imgMeta.width] = HIGH;
            output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
        }       
    }
}

__host__ int Thinning::thinAhmedPtShared(Image *inimg, Image *outimg)
{
 // 局部变量，错误码。
    int errcode;  
    cudaError_t cudaerrcode; 

    // 检查输入图像，输出图像是否为空。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 声明所有中间变量并初始化为空。
    Image *tempimg = NULL;
    unsigned char *flag1 = NULL;
    unsigned char *flag2 = NULL;
    int *devchangecount = NULL;
    unsigned char *dev_lut;
    unsigned char lut[256] = { 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 
    1, 1, 0, 0, 1, 1, 0, 0 };
    
    cudaerrcode = cudaMalloc((void **)&dev_lut, sizeof (unsigned char) * 256);
    if (cudaerrcode != cudaSuccess) 
        return CUDA_ERROR;

    cudaerrcode = cudaMemcpy(dev_lut, lut, sizeof(unsigned char) * 256, cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess) 
        return CUDA_ERROR;

    // 记录细化点数的变量，位于 host 端。
    int changeCount, iteration;

    // 记录细化点数的变量，位于 device 端。并为其申请空间。
    cudaerrcode = cudaMalloc((void **)&devchangecount, sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 生成暂存图像。
    errcode = ImageBasicOp::newImage(&tempimg);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                inimg->height);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 将输入图像 inimg 完全拷贝到输出图像 outimg ，并将 outimg 拷贝到 
    // device 端。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg, outimg);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 提取输出图像
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 提取暂存图像
    ImageCuda tempsubimgCud;
    errcode = ImageBasicOp::roiSubImage(tempimg, &tempsubimgCud);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 生成flag标志数组
    errcode = cudaMalloc((void**)(&flag1), sizeof(unsigned char) *
            inimg->width * inimg->height);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = cudaMalloc((void**)(&flag2), sizeof(unsigned char) *
            outimg->width * outimg->height);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 赋值为 1，以便开始第一次迭代。
    changeCount = 1;

    // 迭代次数，为了计算滚动的状态
    iteration = 0;

    // copy ouimg to tempimg 
     cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                  outimg->imgData, outsubimgCud.deviceId, 
                                  outsubimgCud.pitchBytes * outimg->height);
    
    if (cudaerrcode != cudaSuccess) {
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 初始的输入flag为全HIGH，保证所有点都会计算第一次
    cudaerrcode = cudaMemset(flag1, HIGH, sizeof(unsigned char) * inimg->width * inimg->height);
     if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
     }

     // 第一轮的输出flag初值是全LOW
    cudaerrcode = cudaMemset(flag2, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
     if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
     }

    // 开始迭代，当不可再被细化，即记录细化点数的变量 changeCount 的值为 0 时，
    // 停止迭代。 
    while (changeCount > 0) {
        // 将 host 端的变量赋值为 0 ，并将值拷贝到 device 端的 devchangecount。
        changeCount = 0;
        cudaerrcode = cudaMemcpy(devchangecount, &changeCount, sizeof (int),
                                 cudaMemcpyHostToDevice);
        if (cudaerrcode != cudaSuccess) {
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

        // 如果已经进行了奇数次迭代，那么现在源图像是在outimg上，输出图像应该在tempsubimgCud上
        // flag标志数组此刻使用flag1作为输入标志，flag2作为输出标志
        if (iteration & 1){
            // 调用核函数，开始第一步细化操作。
            _thinAhmedPtSharedKer<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud,
                    flag2, flag1, devchangecount, dev_lut);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }    
            // 每次需要清空下次作为输出flag的数组（即本次迭代的输入flag)
            // 因为每次迭代需要刷新的都是新的“最外一圈”，持续变化中
            cudaerrcode = cudaMemset(flag2, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
            if (cudaerrcode != cudaSuccess) {
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }
        }else{
            // 调用核函数，开始第一步细化操作。
            _thinAhmedPtSharedKer<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud,
                    flag1, flag2, devchangecount, dev_lut);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }    
            // 每次需要清空下次作为输出flag的数组（即本次迭代的输入flag)
            // 因为每次迭代需要刷新的都是新的“最外一圈”，持续变化中
            cudaerrcode = cudaMemset(flag1, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
            if (cudaerrcode != cudaSuccess) {
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }
        }
        
        // 将位于 device 端的 devchangecount 拷贝到 host 端上的 changeCount 
        // 变量，进行迭代判断。
        cudaerrcode = cudaMemcpy(&changeCount, devchangecount, sizeof (int),
                                 cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

        iteration ++;
   }
    // 如果进行了偶数次迭代,那么outimg现在在tempsubimgCud上，
    // 把tempsubimgCud的内容拷贝到outimg指针的设备端内存中
    if (!(iteration & 1)){
         cudaerrcode = cudaMemcpyPeer(outimg->imgData, outsubimgCud.deviceId, 
                                      tempimg->imgData, tempsubimgCud.deviceId, 
                                      tempsubimgCud.pitchBytes * tempimg->height);
    }

    // 细化结束后释放申请的变量空间。
    cudaFree(devchangecount);
    ImageBasicOp::deleteImage(tempimg);
    cudaFree(flag1);
    cudaFree(flag2);
    
    return NO_ERROR;
}

// 常量内存空间
__constant__ unsigned char constant_lut[256];
// GPU版本2，优化分支，使用Pattern表法，Pattern表位于global内存中
static __global__ void _thinAhmedPtConstantKer(ImageCuda tempimg, ImageCuda outimg,
        unsigned char* input_flag, unsigned char* output_flag, int *devchangecount, unsigned char *dev_lut)
{
    // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    // 两边各有两个点不处理。
    if (c >= outimg.imgMeta.width - 2 || 
         r >= outimg.imgMeta.height - 2 || c < 2 || r < 2)
        return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    // 从左上角第二行第二列开始计算。
    int curpos = (r) * outimg.pitchBytes + c ;
    int flagpos = r * tempimg.imgMeta.width + c;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;
    unsigned char x1, x2, x3, x4, x5, x6, x7, x8;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (isHigh1(*outptr)) {
        // 如果在滚动数组中，发现上一轮发生了删点，说明此时实质上已经是Low了
        // 因此刷新，并返回
        if (isLow1(tempimg.imgMeta.imgData[curpos])){
            outimg.imgMeta.imgData[curpos] = LOW;
            // 因为该修改不属于本次迭代的修改，所以不修改changeCount
            return;
        }
        // 如果上一轮的计算中，没有标志此像素点需要重新计算权值,
        // 虽然像素值是High，仍然略过，因为本次迭代中已知它不会改变（被删）
        // 也不会影响它的八邻域
        if (isLow1(input_flag[flagpos])){
            return;
        }
        x1 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 1];
        x2 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes];
        x3 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 1];
        x4 = tempimg.imgMeta.imgData[curpos - 1];
        x5 = tempimg.imgMeta.imgData[curpos + 1];
        x6 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 1];
        x7 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes];
        x8 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 1];
        unsigned char x9,x10,x11;
        if(isHigh1(x7)) {
            x9 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 + 1];

            if (isHigh5(x4,x5,x6,x7,x8) && isLow2(x2,x10) ||
                isHigh3(x6,x7,x9) && isLow8(x1,x2,x3,x4,x5,x8,x10,x11) ||
                isHigh3(x7,x8,x11) && isLow8(x1,x2,x3,x4,x5,x6,x9,x10))
                    return ;
        } 
        if(isHigh1(x2)) {
            // w is down
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 + 1];

            if (isHigh5(x1,x2,x3,x4,x5) && isLow2(x7,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
                output_flag[flagpos - 1] = HIGH;
                output_flag[flagpos + 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
                return ;
            } else if (isHigh3(x1,x2,x9) && isLow8(x3,x4,x5,x6,x7,x8,x10,x11) ||
                       isHigh3(x2,x3,x11) && isLow8(x1,x4,x5,x6,x7,x8,x9,x10)) 
                return ;
        }
        if(isHigh1(x5)) {
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 2];
            x10 = tempimg.imgMeta.imgData[curpos + 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 2];

            if (isHigh5(x2,x3,x5,x7,x8) && isLow2(x4,x10) ||
                isHigh3(x5,x8,x11) && isLow8(x1,x2,x3,x4,x6,x7,x9,x10) ||
                isHigh3(x3,x5,x9) && isLow8(x1,x2,x4,x6,x7,x8,x10,x11))
                return ;
        }
        if(isHigh1(x4)){
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 2];
            x10 = tempimg.imgMeta.imgData[curpos - 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 2];
            if (isHigh5(x1,x2,x4,x6,x7) && isLow2(x5,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
                output_flag[flagpos - 1] = HIGH;
                output_flag[flagpos + 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
                return ;
            } else if (isHigh3(x4,x6,x11) && isLow8(x1,x2,x3,x5,x7,x8,x9,x10) ||
                       isHigh3(x1,x4,x9) && isLow8(x2,x3,x5,x6,x7,x8,x10,x11)) 
                return ;
        }

        // 1   2   4
        // 8       16
        // 32  64  128
        unsigned char index = isHigh1(x1) * 1 + isHigh1(x2) * 2 + isHigh1(x3) * 4 + isHigh1(x4) * 8 +
                              isHigh1(x5) * 16 + isHigh1(x6) * 32 + isHigh1(x7) * 64 + isHigh1(x8) * 128;
        
        if (constant_lut[index] == 1) {
            outimg.imgMeta.imgData[curpos] = LOW;
            *devchangecount = 1;
            output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
            output_flag[flagpos - outimg.imgMeta.width] = HIGH;
            output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
            output_flag[flagpos - 1] = HIGH;
            output_flag[flagpos + 1] = HIGH;
            output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
            output_flag[flagpos + outimg.imgMeta.width] = HIGH;
            output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
        }       
    }
}

__host__ int Thinning::thinAhmedPtConstant(Image *inimg, Image *outimg)
{
 // 局部变量，错误码。
    int errcode;  
    cudaError_t cudaerrcode; 

    // 检查输入图像，输出图像是否为空。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 声明所有中间变量并初始化为空。
    Image *tempimg = NULL;
    unsigned char *flag1 = NULL;
    unsigned char *flag2 = NULL;
    int *devchangecount = NULL;
    // unsigned char *dev_lut;
    unsigned char lut[256] = { 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 
    1, 1, 0, 0, 1, 1, 0, 0 };
    
    // cudaerrcode = cudaMalloc((void **)&dev_lut, sizeof (unsigned char) * 256);
    // if (cudaerrcode != cudaSuccess) 
    //     return CUDA_ERROR;

    // cudaerrcode = cudaMemcpy(dev_lut, lut, sizeof(unsigned char) * 256, cudaMemcpyHostToDevice);
    // if (cudaerrcode != cudaSuccess) 
    //     return CUDA_ERROR;

    cudaerrcode = cudaMemcpyToSymbol(constant_lut, lut, sizeof(unsigned char) * 256);
    if (cudaerrcode != cudaSuccess) 
        return CUDA_ERROR;

    // 记录细化点数的变量，位于 host 端。
    int changeCount, iteration;

    // 记录细化点数的变量，位于 device 端。并为其申请空间。
    cudaerrcode = cudaMalloc((void **)&devchangecount, sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 生成暂存图像。
    errcode = ImageBasicOp::newImage(&tempimg);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                inimg->height);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 将输入图像 inimg 完全拷贝到输出图像 outimg ，并将 outimg 拷贝到 
    // device 端。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg, outimg);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 提取输出图像
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 提取暂存图像
    ImageCuda tempsubimgCud;
    errcode = ImageBasicOp::roiSubImage(tempimg, &tempsubimgCud);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 生成flag标志数组
    errcode = cudaMalloc((void**)(&flag1), sizeof(unsigned char) *
            inimg->width * inimg->height);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = cudaMalloc((void**)(&flag2), sizeof(unsigned char) *
            outimg->width * outimg->height);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 赋值为 1，以便开始第一次迭代。
    changeCount = 1;
    
    // 迭代次数，为了计算滚动的状态
    iteration = 0;
    
    // copy ouimg to tempimg 
     cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                  outimg->imgData, outsubimgCud.deviceId, 
                                  outsubimgCud.pitchBytes * outimg->height);
    
    if (cudaerrcode != cudaSuccess) {
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 初始的输入flag为全HIGH，保证所有点都会计算第一次
    cudaerrcode = cudaMemset(flag1, HIGH, sizeof(unsigned char) * inimg->width * inimg->height);
     if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
     }

     // 第一轮的输出flag初值是全LOW
    cudaerrcode = cudaMemset(flag2, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
     if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
     }

    // 开始迭代，当不可再被细化，即记录细化点数的变量 changeCount 的值为 0 时，
    // 停止迭代。 
    while (changeCount > 0) {
        // 将 host 端的变量赋值为 0 ，并将值拷贝到 device 端的 devchangecount。
        changeCount = 0;
        cudaerrcode = cudaMemcpy(devchangecount, &changeCount, sizeof (int),
                                 cudaMemcpyHostToDevice);
        if (cudaerrcode != cudaSuccess) {
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

        // 如果已经进行了奇数次迭代，那么现在源图像是在outimg上，输出图像应该在tempsubimgCud上
        // flag标志数组此刻使用flag1作为输入标志，flag2作为输出标志
        if (iteration & 1){
            // 调用核函数，开始第一步细化操作。
            _thinAhmedPtConstantKer<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud,
                    flag2, flag1, devchangecount, dev_lut);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }    
            // 每次需要清空下次作为输出flag的数组（即本次迭代的输入flag)
            // 因为每次迭代需要刷新的都是新的“最外一圈”，持续变化中
            cudaerrcode = cudaMemset(flag2, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
            if (cudaerrcode != cudaSuccess) {
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }
        }else{
            // 调用核函数，开始第一步细化操作。
            _thinAhmedPtConstantKer<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud,
                    flag1, flag2, devchangecount, dev_lut);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }    
            // 每次需要清空下次作为输出flag的数组（即本次迭代的输入flag)
            // 因为每次迭代需要刷新的都是新的“最外一圈”，持续变化中
            cudaerrcode = cudaMemset(flag1, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
            if (cudaerrcode != cudaSuccess) {
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }
        }
        
        // 将位于 device 端的 devchangecount 拷贝到 host 端上的 changeCount 
        // 变量，进行迭代判断。
        cudaerrcode = cudaMemcpy(&changeCount, devchangecount, sizeof (int),
                                 cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

        iteration ++;
   }
    // 如果进行了偶数次迭代,那么outimg现在在tempsubimgCud上，
    // 把tempsubimgCud的内容拷贝到outimg指针的设备端内存中
    if (!(iteration & 1)){
         cudaerrcode = cudaMemcpyPeer(outimg->imgData, outsubimgCud.deviceId, 
                                      tempimg->imgData, tempsubimgCud.deviceId, 
                                      tempsubimgCud.pitchBytes * tempimg->height);
    }

    // 细化结束后释放申请的变量空间。
    cudaFree(devchangecount);
    ImageBasicOp::deleteImage(tempimg);
    cudaFree(flag1);
    cudaFree(flag2);

    return NO_ERROR;
}

texture<unsigned char, 1, cudaReadModeElementType> tex_lut;
static __global__ void _thinAhmedPtTextureKer(ImageCuda tempimg, ImageCuda outimg,
        unsigned char* input_flag, unsigned char* output_flag, int *devchangecount, unsigned char *dev_lut)
{
    // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    // 两边各有两个点不处理。
    if (c >= outimg.imgMeta.width - 2 || 
         r >= outimg.imgMeta.height - 2 || c < 2 || r < 2)
        return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    // 从左上角第二行第二列开始计算。
    int curpos = (r) * outimg.pitchBytes + c ;
    int flagpos = r * tempimg.imgMeta.width + c;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;
    unsigned char x1, x2, x3, x4, x5, x6, x7, x8;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (isHigh1(*outptr)) {
        // 如果在滚动数组中，发现上一轮发生了删点，说明此时实质上已经是Low了
        // 因此刷新，并返回
        if (isLow1(tempimg.imgMeta.imgData[curpos])){
            outimg.imgMeta.imgData[curpos] = LOW;
            // 因为该修改不属于本次迭代的修改，所以不修改changeCount
            return;
        }
        // 如果上一轮的计算中，没有标志此像素点需要重新计算权值,
        // 虽然像素值是High，仍然略过，因为本次迭代中已知它不会改变（被删）
        // 也不会影响它的八邻域
        if (isLow1(input_flag[flagpos])){
            return;
        }
        x1 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 1];
        x2 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes];
        x3 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 1];
        x4 = tempimg.imgMeta.imgData[curpos - 1];
        x5 = tempimg.imgMeta.imgData[curpos + 1];
        x6 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 1];
        x7 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes];
        x8 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 1];
        unsigned char x9,x10,x11;
        if(isHigh1(x7)) {
            x9 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 + 1];

            if (isHigh5(x4,x5,x6,x7,x8) && isLow2(x2,x10) ||
                isHigh3(x6,x7,x9) && isLow8(x1,x2,x3,x4,x5,x8,x10,x11) ||
                isHigh3(x7,x8,x11) && isLow8(x1,x2,x3,x4,x5,x6,x9,x10))
                    return ;
        } 
        if(isHigh1(x2)) {
            // w is down
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 + 1];

            if (isHigh5(x1,x2,x3,x4,x5) && isLow2(x7,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
                output_flag[flagpos - 1] = HIGH;
                output_flag[flagpos + 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
                return ;
            } else if (isHigh3(x1,x2,x9) && isLow8(x3,x4,x5,x6,x7,x8,x10,x11) ||
                       isHigh3(x2,x3,x11) && isLow8(x1,x4,x5,x6,x7,x8,x9,x10)) 
                return ;
        }
        if(isHigh1(x5)) {
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 2];
            x10 = tempimg.imgMeta.imgData[curpos + 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 2];

            if (isHigh5(x2,x3,x5,x7,x8) && isLow2(x4,x10) ||
                isHigh3(x5,x8,x11) && isLow8(x1,x2,x3,x4,x6,x7,x9,x10) ||
                isHigh3(x3,x5,x9) && isLow8(x1,x2,x4,x6,x7,x8,x10,x11))
                return ;
        }
        if(isHigh1(x4)){
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 2];
            x10 = tempimg.imgMeta.imgData[curpos - 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 2];
            if (isHigh5(x1,x2,x4,x6,x7) && isLow2(x5,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width] = HIGH;
                output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
                output_flag[flagpos - 1] = HIGH;
                output_flag[flagpos + 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width] = HIGH;
                output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
                return ;
            } else if (isHigh3(x4,x6,x11) && isLow8(x1,x2,x3,x5,x7,x8,x9,x10) ||
                       isHigh3(x1,x4,x9) && isLow8(x2,x3,x5,x6,x7,x8,x10,x11)) 
                return ;
        }

        // 1   2   4
        // 8       16
        // 32  64  128
        unsigned char index = isHigh1(x1) * 1 + isHigh1(x2) * 2 + isHigh1(x3) * 4 + isHigh1(x4) * 8 +
                              isHigh1(x5) * 16 + isHigh1(x6) * 32 + isHigh1(x7) * 64 + isHigh1(x8) * 128;
        
        if (tex1Dfetch(tex_lut, index) == 1) {
            outimg.imgMeta.imgData[curpos] = LOW;
            *devchangecount = 1;
            output_flag[flagpos - outimg.imgMeta.width - 1] = HIGH;
            output_flag[flagpos - outimg.imgMeta.width] = HIGH;
            output_flag[flagpos - outimg.imgMeta.width + 1] = HIGH;
            output_flag[flagpos - 1] = HIGH;
            output_flag[flagpos + 1] = HIGH;
            output_flag[flagpos + outimg.imgMeta.width - 1] = HIGH;
            output_flag[flagpos + outimg.imgMeta.width] = HIGH;
            output_flag[flagpos + outimg.imgMeta.width + 1] = HIGH;
        }       
    }
}

__host__ int Thinning::thinAhmedPtTexture(Image *inimg, Image *outimg)
{
 // 局部变量，错误码。
    int errcode;  
    cudaError_t cudaerrcode; 

    // 检查输入图像，输出图像是否为空。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 声明所有中间变量并初始化为空。
    Image *tempimg = NULL;
    unsigned char *flag1 = NULL;
    unsigned char *flag2 = NULL;
    int *devchangecount = NULL;
    // unsigned char *dev_lut;
    unsigned char lut[256] = { 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 
    1, 1, 0, 0, 1, 1, 0, 0 };
    
    // cudaerrcode = cudaMalloc((void **)&dev_lut, sizeof (unsigned char) * 256);
    // if (cudaerrcode != cudaSuccess) 
    //     return CUDA_ERROR;

    // cudaerrcode = cudaMemcpy(dev_lut, lut, sizeof(unsigned char) * 256, cudaMemcpyHostToDevice);
    // if (cudaerrcode != cudaSuccess) 
    //     return CUDA_ERROR;

    unsigned char *dev_lut;

    cudaMalloc((void **) &dev_lut, 256*sizeof(unsigned char));
    cudaMemcpy((void *)dev_lut, (void *)lut, 256*sizeof(unsigned char), cudaMemcpyHostToDevice);
    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaBindTexture(0, tex_lut, dev_lut);

    // 记录细化点数的变量，位于 host 端。
    int changeCount, iteration;

    // 记录细化点数的变量，位于 device 端。并为其申请空间。
    cudaerrcode = cudaMalloc((void **)&devchangecount, sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 生成暂存图像。
    errcode = ImageBasicOp::newImage(&tempimg);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                inimg->height);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 将输入图像 inimg 完全拷贝到输出图像 outimg ，并将 outimg 拷贝到 
    // device 端。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg, outimg);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 提取输出图像
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 提取暂存图像
    ImageCuda tempsubimgCud;
    errcode = ImageBasicOp::roiSubImage(tempimg, &tempsubimgCud);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 生成flag标志数组
    errcode = cudaMalloc((void**)(&flag1), sizeof(unsigned char) *
            inimg->width * inimg->height);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = cudaMalloc((void**)(&flag2), sizeof(unsigned char) *
            outimg->width * outimg->height);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 赋值为 1，以便开始第一次迭代。
    changeCount = 1;

    // 迭代次数，为了计算滚动的状态
    iteration = 0;

    // copy ouimg to tempimg 
     cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                  outimg->imgData, outsubimgCud.deviceId, 
                                  outsubimgCud.pitchBytes * outimg->height);
    
    if (cudaerrcode != cudaSuccess) {
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 初始的输入flag为全HIGH，保证所有点都会计算第一次
    cudaerrcode = cudaMemset(flag1, HIGH, sizeof(unsigned char) * inimg->width * inimg->height);
     if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
     }

     // 第一轮的输出flag初值是全LOW
    cudaerrcode = cudaMemset(flag2, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
     if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
     }

    // 开始迭代，当不可再被细化，即记录细化点数的变量 changeCount 的值为 0 时，
    // 停止迭代。 
    while (changeCount > 0) {
        // 将 host 端的变量赋值为 0 ，并将值拷贝到 device 端的 devchangecount。
        changeCount = 0;
        cudaerrcode = cudaMemcpy(devchangecount, &changeCount, sizeof (int),
                                 cudaMemcpyHostToDevice);
        if (cudaerrcode != cudaSuccess) {
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }
        // 如果已经进行了奇数次迭代，那么现在源图像是在outimg上，输出图像应该在tempsubimgCud上
        // flag标志数组此刻使用flag1作为输入标志，flag2作为输出标志
        if (iteration & 1){
            // 调用核函数，开始第一步细化操作。
            _thinAhmedPtTextureKer<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud,
                    flag2, flag1, devchangecount, dev_lut);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }    
            // 每次需要清空下次作为输出flag的数组（即本次迭代的输入flag)
            // 因为每次迭代需要刷新的都是新的“最外一圈”，持续变化中
            cudaerrcode = cudaMemset(flag2, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
            if (cudaerrcode != cudaSuccess) {
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }
        }else{
            // 调用核函数，开始第一步细化操作。
            _thinAhmedPtTextureKer<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud,
                    flag1, flag2, devchangecount, dev_lut);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }    
            // 每次需要清空下次作为输出flag的数组（即本次迭代的输入flag)
            // 因为每次迭代需要刷新的都是新的“最外一圈”，持续变化中
            cudaerrcode = cudaMemset(flag1, LOW, sizeof(unsigned char) * outimg->width * outimg->height);
            if (cudaerrcode != cudaSuccess) {
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }
        }
        
        // 将位于 device 端的 devchangecount 拷贝到 host 端上的 changeCount 
        // 变量，进行迭代判断。
        cudaerrcode = cudaMemcpy(&changeCount, devchangecount, sizeof (int),
                                 cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

        iteration ++;
   }
    // 如果进行了偶数次迭代,那么outimg现在在tempsubimgCud上，
    // 把tempsubimgCud的内容拷贝到outimg指针的设备端内存中
    if (!(iteration & 1)){
         cudaerrcode = cudaMemcpyPeer(outimg->imgData, outsubimgCud.deviceId, 
                                      tempimg->imgData, tempsubimgCud.deviceId, 
                                      tempsubimgCud.pitchBytes * tempimg->height);
    }

    // 细化结束后释放申请的变量空间。
    cudaFree(devchangecount);
    ImageBasicOp::deleteImage(tempimg);
    cudaFree(flag1);
    cudaFree(flag2);
    
    return NO_ERROR;
}

