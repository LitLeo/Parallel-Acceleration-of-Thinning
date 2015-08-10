// Thinning.cu
// 实现二值图像的细化算法

#include "Thinning.h"
#include <iostream>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：DEF_PATTERN_SIZE
// 定义了 PATTERN 表的默认大小。
#define DEF_PATTERN_SIZE  512

#define HIGH 255
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



__constant__ unsigned char lutthin[2048];

// 256
unsigned char lutthin[] = { 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 
                            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
                            1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,  };


static __global__ void _thinAhmed(ImageCuda tempimg, ImageCuda outimg, int *devchangecount)
{
    // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    // 两边各有两个点不处理。
    if (c >= outimg.imgMeta.width - 5 || 
        r >= outimg.imgMeta.height - 5)
        return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    // 从左上角第二行第二列开始计算。
    int curpos = (r + 2) * outimg.pitchBytes + c + 2;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (isHigh1(*outptr)) {
        unsigned char x1 = tempimg.imgMeta.imgData[r - 1, c - 1];
        unsigned char x2 = tempimg.imgMeta.imgData[r - 1, c];
        unsigned char x3 = tempimg.imgMeta.imgData[r - 1, c + 1];
        unsigned char x4 = tempimg.imgMeta.imgData[r, c - 1];
        unsigned char x5 = tempimg.imgMeta.imgData[r, c + 1];
        unsigned char x6 = tempimg.imgMeta.imgData[r + 1, c - 1];
        unsigned char x7 = tempimg.imgMeta.imgData[r + 1, c];
        unsigned char x8 = tempimg.imgMeta.imgData[r + 1, c + 1];
        unsigned char x9,x10,x11;
        if(isHigh1(x7)) {
            x9 = tempimg.imgMeta.imgData[r + 2, c - 1];
            x10 = tempimg.imgMeta.imgData[r + 2, c];
            x11 = tempimg.imgMeta.imgData[r + 2, c + 1];

            if (isHigh5(x4,x5,x6,x7,x8) && isLow2(x2,x10) ||
                isHigh3(x6,x7,x9) && isLow8(x1,x2,x3,x4,x5,x8,x10,x11) ||
                isHigh3(x7,x8,x11) && isLow8(x1,x2,x3,x4,x5,x6,x9,x10)) {
                    continue ;
            } 
        } 
        if(isHigh1(x2)) {
            // w is down
            x9 = tempimg.imgMeta.imgData[r - 2, c - 1];
            x10 = tempimg.imgMeta.imgData[r - 2, c];
            x11 = tempimg.imgMeta.imgData[r - 2, c + 1];

            if (isHigh5(x1,x2,x3,x4,x5) && isLow2(x7,x10)){
                outimg.imgMeta.imgData[r,c] = LOW;
                *changecount = 1;
                continue ;
                
            } else if (isHigh3(x1,x2,x9) && isLow8(x3,x4,x5,x6,x7,x8,x10,x11) ||
                       isHigh3(x2,x3,x11) && isLow8(x1,x4,x5,x6,x7,x8,x9,x10)) {
                continue ;
            }
        }
        if(isHigh1(x5)) {
            x9 = tempimg.imgMeta.imgData[r - 1, c + 2];
            x10 = tempimg.imgMeta.imgData[r, c + 2];
            x11 = tempimg.imgMeta.imgData[r + 1, c + 2];

            if (isHigh5(x2,x3,x5,x7,x8) && isLow2(x4,x10) ||
                isHigh3(x5,x8,x11) && isLow8(x1,x2,x3,x4,x6,x7,x9,x10) ||
                isHigh3(x3,x5,x9) && isLow8(x1,x2,x4,x6,x7,x8,x10,x11)) {
                continue ;
            }
        }
        if(isHigh1(x4)){
            x9 = tempimg.imgMeta.imgData[r - 1, c - 2];
            x10 = tempimg.imgMeta.imgData[r, c - 2];
            x11 = tempimg.imgMeta.imgData[r + 1, c - 2];
            if (isHigh5(x1,x2,x4,x6,x7) && isLow2(x5,x10)){
                outimg.imgMeta.imgData[r,c] = LOW;
                *changecount = 1;
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
                outimg.imgMeta.imgData[r,c] = LOW;
                *changecount = 1;
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
    int *devchangecount = NULL;

    // 记录细化点数的变量，位于 host 端。
    int changeCount;

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

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1 - 4) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1 - 4) / blocksize.y;

    // 赋值为 1，以便开始第一次迭代。
    changeCount = 1;

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

        errcode = ImageBasicOp::copyToCurrentDevice(outimg, tempimg);
        if (errcode != NO_ERROR) {
            // FAIL_THIN_IMAGE_FREE;
            return errcode;
        }
            
        // 调用核函数，开始第一步细化操作。
        _thinAhmed<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud, devchangecount);
        if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束迭代函数，释放申请的变量空间。
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }    
        
        // 将位于 device 端的 devchangecount 拷贝到 host 端上的 changeCount 
        // 变量，进行迭代判断。
        cudaerrcode = cudaMemcpy(&changeCount, devchangecount, sizeof (int),
                                 cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

   }

    // 细化结束后释放申请的变量空间。
    cudaFree(devchangecount);
    ImageBasicOp::deleteImage(tempimg);

    return NO_ERROR;
}
__host__ int Thinning::thinAhmedFour(Image *inimg, Image *outimg)
{
    return NO_ERROR;
}
__host__ int Thinning::thinAhmedPt(Image *inimg, Image *outimg)
{
    return NO_ERROR;
}
__host__ int Thinning::thinAhmedPtFour(Image *inimg, Image *outimg)
{
    return NO_ERROR;
}