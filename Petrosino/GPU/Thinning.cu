// Thinning.cu
// 实现二值图像的细化算法

#include "Thinning.h"
#include <iostream>
#include <stdio.h>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

#define uchar unsigned char

// 宏：DEF_PATTERN_SIZE
// 定义了 PATTERN 表的默认大小。
#define DEF_PATTERN_SIZE  512

#define uchar unsigned char

#define HIGH 255
#define LOW 0

static __global__ void _thinPet1Ker(ImageCuda tempimg, ImageCuda outimg, int *devchangecount)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 2 || 
     dstr >= tempimg.imgMeta.height - 2 || dstc < 2 || dstr < 2)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * tempimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = tempimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
        int posColumn2 = posColumn1 + tempimg.pitchBytes;
        int posColumn3 = posColumn2 + tempimg.pitchBytes;
        int posColumn4 = posColumn3 + tempimg.pitchBytes;

        uchar x1 = tempimg.imgMeta.imgData[posColumn2 + 1 + dstc] == HIGH;
        uchar x2 = tempimg.imgMeta.imgData[posColumn1 + 1 + dstc] == HIGH;
        uchar x3 = tempimg.imgMeta.imgData[posColumn1     + dstc] == HIGH;
        uchar x4 = tempimg.imgMeta.imgData[posColumn1 + 1 + dstc] == HIGH;
        uchar x5 = tempimg.imgMeta.imgData[posColumn2 + 1 + dstc] == HIGH;
        uchar x6 = tempimg.imgMeta.imgData[posColumn3 + 1 + dstc] == HIGH;
        uchar x7 = tempimg.imgMeta.imgData[posColumn3     + dstc] == HIGH;
        uchar x8 = tempimg.imgMeta.imgData[posColumn3 + 1 + dstc] == HIGH;
        // uchar y1 = tempimg.imgMeta.imgData[posColumn4 + 1 + dstc] == HIGH;
        uchar y2 = tempimg.imgMeta.imgData[posColumn4     + dstc] == HIGH;
        // uchar y3 = tempimg.imgMeta.imgData[posColumn4 + 1 + dstc] == HIGH;
        // uchar y4 = tempimg.imgMeta.imgData[posColumn1 + 2 + dstc] == HIGH;
        uchar y5 = tempimg.imgMeta.imgData[posColumn2 + 2 + dstc] == HIGH;
        // uchar y6 = tempimg.imgMeta.imgData[posColumn3 + 2 + dstc] == HIGH;

        int A  = (x2 ^ x3) + (x3 ^ x4) + (x4 ^ x5) + (x5 ^ x6) + 
                 (x6 ^ x7) + (x7 ^ x8) + (x8 ^ x1) + (x1 ^ x2);
        int B  = x2 + x3 + x4 + x5 + x6 + x7 + x8 + x1;
        int R = x1 && x7 && x8 &&
               ((!y5 && x2 && x3 && !x5) || (!y2 && !x3 && x5 && x6));
        if (A == 2 && B >= 2 && B <= 6 && R == 0) {
            outimg.imgMeta.imgData[curpos] = LOW;
            // 记录删除点数的 devchangecount 值加 1 。
            *devchangecount = 1;
        }
    }
}

static __global__ void _thinPet2Ker(ImageCuda tempimg, ImageCuda outimg, int *devchangecount)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 2 || 
     dstr >= tempimg.imgMeta.height - 2 || dstc < 2 || dstr < 2)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * tempimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = tempimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
        int posColumn2 = posColumn1 + tempimg.pitchBytes;
        int posColumn3 = posColumn2 + tempimg.pitchBytes;
        // int posColumn4 = posColumn3 + tempimg.pitchBytes;

        uchar x1 = tempimg.imgMeta.imgData[posColumn2 + 1 + dstc] == HIGH;
        uchar x2 = tempimg.imgMeta.imgData[posColumn1 + 1 + dstc] == HIGH;
        uchar x3 = tempimg.imgMeta.imgData[posColumn1     + dstc] == HIGH;
        uchar x4 = tempimg.imgMeta.imgData[posColumn1 + 1 + dstc] == HIGH;
        uchar x5 = tempimg.imgMeta.imgData[posColumn2 + 1 + dstc] == HIGH;
        uchar x6 = tempimg.imgMeta.imgData[posColumn3 + 1 + dstc] == HIGH;
        uchar x7 = tempimg.imgMeta.imgData[posColumn3     + dstc] == HIGH;
        uchar x8 = tempimg.imgMeta.imgData[posColumn3 + 1 + dstc] == HIGH;
        // uchar y1 = tempimg.imgMeta.imgData[posColumn4 + 1 + dstc] == HIGH;
        // uchar y2 = tempimg.imgMeta.imgData[posColumn4     + dstc] == HIGH;
        // uchar y3 = tempimg.imgMeta.imgData[posColumn4 + 1 + dstc] == HIGH;
        // uchar y4 = tempimg.imgMeta.imgData[posColumn1 + 2 + dstc] == HIGH;
        // uchar y5 = tempimg.imgMeta.imgData[posColumn2 + 2 + dstc] == HIGH;
        // uchar y6 = tempimg.imgMeta.imgData[posColumn3 + 2 + dstc] == HIGH;

        int S0 = (x3&&x7) || (x5&&x1);
        int S1 = (x1 && !x6 && (!x4 || x3)) || (x3 && !x8 && (!x6 || x5)) ||
            (x7 && !x4 && (!x2 || x1)) || (x5 && !x2 && (!x8 || x7));
        int B  = x2 + x3 + x4 + x5 + x6 + x7 + x8 + x1;
        int R = (x3 && (x1&&!x8 || x5&&!x6)) || (x7 && (!x5&&!x8 || !x1&&!x6));
        if ((!S0 && S1) && R == 0 && B >= 3) {
            outimg.imgMeta.imgData[curpos] = LOW;
            // 记录删除点数的 devchangecount 值加 1 。
            *devchangecount = 1;
        }
    }
}

// 直接并行化
// 线程数，处理多少个点有多少线程数
__host__ int Thinning::thinPet(Image *inimg, Image *outimg)
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
         return CUDA_ERROR;
     }

     // 生成暂存图像。
     errcode = ImageBasicOp::newImage(&tempimg);
     if (errcode != NO_ERROR)
         return errcode;
     errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                 inimg->height);
     if (errcode != NO_ERROR) {
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
     gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
     gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

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
             return CUDA_ERROR;
         }

         // copy ouimg to tempimg 
         cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                      outimg->imgData, outsubimgCud.deviceId, 
                                      outsubimgCud.pitchBytes * outimg->height);
        
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // 调用核函数，开始第一步细化操作。
         _thinPet1Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
         if (cudaGetLastError() != cudaSuccess) {
             // 核函数出错，结束迭代函数，释放申请的变量空间。
             // FAIL_THIN_IMAGE_FREE;
             return CUDA_ERROR;
         }

         // copy ouimg to tempimg 
         cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                      outimg->imgData, outsubimgCud.deviceId, 
                                      outsubimgCud.pitchBytes * outimg->height);
        
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // 调用核函数，开始第二步细化操作。
         _thinPet2Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
         if (cudaGetLastError() != cudaSuccess) {
             // 核函数出错，结束迭代函数，释放申请的变量空间 。
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

/*
static __global__ void _thinPetFour1Ker(ImageCuda tempimg, ImageCuda outimg, int *devchangecount)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 1 || 
     dstr >= tempimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * tempimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = tempimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
        int posColumn2 = posColumn1 + tempimg.pitchBytes;
        int posColumn3 = posColumn2 + tempimg.pitchBytes;

        // p1 p2 p3
        // p8    p4
        // p7 p6 p5
        unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + posColumn1] == HIGH;
        unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1] == HIGH;
        unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1] == HIGH;
        unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2] == HIGH;
        unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3] == HIGH;
        unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3] == HIGH;
        unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3] == HIGH;
        unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2] == HIGH;

        int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                 (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                 (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                 (p8 == 0 && p1 == 1) + (p1 == 0 && p2 == 1);
        int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p1;
        int m1 = (p2 * p4 * p6);
        int m2 = (p4 * p6 * p8);

        if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
            outimg.imgMeta.imgData[curpos] = LOW;
            // 记录删除点数的 devchangecount 值加 1 。
            *devchangecount = 1;
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (++dstr >= outimg.imgMeta.height - 1)
            return ;
        curpos += outimg.pitchBytes;  

        // 获取当前像素点在图像中的绝对位置。
        outptr = outimg.imgMeta.imgData + curpos;
        if (*outptr != LOW) {
            // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
            // 防止下面细化处理时重复计算。
            int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
            int posColumn2 = posColumn1 + tempimg.pitchBytes;
            int posColumn3 = posColumn2 + tempimg.pitchBytes;

            // p1 p2 p3
            // p8    p4
            // p7 p6 p5
            unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + posColumn1] == HIGH;
            unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1] == HIGH;
            unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1] == HIGH;
            unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2] == HIGH;
            unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3] == HIGH;
            unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3] == HIGH;
            unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3] == HIGH;
            unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2] == HIGH;

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p1 == 1) + (p1 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p1;
            int m1 = (p2 * p4 * p6);
            int m2 = (p4 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
                outimg.imgMeta.imgData[curpos] = LOW;
                // 记录删除点数的 devchangecount 值加 1 。
                *devchangecount = 1;
            }
        }
    }
}

static __global__ void _thinPetFour2Ker(ImageCuda tempimg, ImageCuda outimg, int *devchangecount)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 1 || 
     dstr >= tempimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * tempimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = tempimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
        int posColumn2 = posColumn1 + tempimg.pitchBytes;
        int posColumn3 = posColumn2 + tempimg.pitchBytes;

        // p1 p2 p3
        // p8    p4
        // p7 p6 p5
        unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + posColumn1] == HIGH;
        unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1] == HIGH;
        unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1] == HIGH;
        unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2] == HIGH;
        unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3] == HIGH;
        unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3] == HIGH;
        unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3] == HIGH;
        unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2] == HIGH;

        int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                 (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                 (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                 (p8 == 0 && p1 == 1) + (p1 == 0 && p2 == 1);
        int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p1;
        int m1 = (p2 * p4 * p8);
        int m2 = (p2 * p6 * p8);

        if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
            outimg.imgMeta.imgData[curpos] = LOW;
            // 记录删除点数的 devchangecount 值加 1 。
            *devchangecount = 1;
        }
    }
    for (int i = 0; i < 3; ++i) {
        if (++dstr >= outimg.imgMeta.height - 1)
            return ;
        curpos += outimg.pitchBytes;  

        // 获取当前像素点在图像中的绝对位置。
        outptr = outimg.imgMeta.imgData + curpos;
        if (*outptr != LOW) {
            // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
            // 防止下面细化处理时重复计算。
            int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
            int posColumn2 = posColumn1 + tempimg.pitchBytes;
            int posColumn3 = posColumn2 + tempimg.pitchBytes;

            // p1 p2 p3
            // p8    p4
            // p7 p6 p5
            unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + posColumn1] == HIGH;
            unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1] == HIGH;
            unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1] == HIGH;
            unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2] == HIGH;
            unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3] == HIGH;
            unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3] == HIGH;
            unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3] == HIGH;
            unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2] == HIGH;

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p1 == 1) + (p1 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p1;
            int m1 = (p2 * p4 * p8);
            int m2 = (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
                outimg.imgMeta.imgData[curpos] = LOW;
                // 记录删除点数的 devchangecount 值加 1 。
                *devchangecount = 1;
            }
        }
    }
}

// 直接并行化
// 线程数，处理多少个点有多少线程数
__host__ int Thinning::thinPetFour(Image *inimg, Image *outimg)
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
         return CUDA_ERROR;
     }

     // 生成暂存图像。
     errcode = ImageBasicOp::newImage(&tempimg);
     if (errcode != NO_ERROR)
         return errcode;
     errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                 inimg->height);
     if (errcode != NO_ERROR) {
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
     gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
     gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y*3 - 1) / blocksize.y*3;

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
             return CUDA_ERROR;
         }

         // copy ouimg to tempimg 
         cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                      outimg->imgData, outsubimgCud.deviceId, 
                                      outsubimgCud.pitchBytes * outimg->height);
        
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // 调用核函数，开始第一步细化操作。
         _thinPetFour1Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
         if (cudaGetLastError() != cudaSuccess) {
             // 核函数出错，结束迭代函数，释放申请的变量空间。
             // FAIL_THIN_IMAGE_FREE;
             return CUDA_ERROR;
         }

         // copy ouimg to tempimg 
         cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                      outimg->imgData, outsubimgCud.deviceId, 
                                      outsubimgCud.pitchBytes * outimg->height);
        
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // 调用核函数，开始第二步细化操作。
         _thinPetFour2Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
         if (cudaGetLastError() != cudaSuccess) {
             // 核函数出错，结束迭代函数，释放申请的变量空间 。
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

static __global__ void _thinPetPt1Ker(ImageCuda tempimg, ImageCuda outimg, 
                                     int *devchangecount, uchar *dev_lut)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 1 || 
     dstr >= tempimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * tempimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = tempimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
        int posColumn2 = posColumn1 + tempimg.pitchBytes;
        int posColumn3 = posColumn2 + tempimg.pitchBytes;

        // p1 p2 p3
        // p8    p4
        // p7 p6 p5
        unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + posColumn1];
        unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1];
        unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1];
        unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2];
        unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3];
        unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3];
        unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3];
        unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2];

        uchar index= (p1==HIGH)*1 + (p2==HIGH)*2 + (p3==HIGH)*4 + (p4==HIGH)*8 + 
                     (p5==HIGH)*16 + (p6==HIGH)*32 + (p7==HIGH)*64 + (p8==HIGH)*128;

        if (dev_lut[index]) {
            outimg.imgMeta.imgData[curpos] = LOW;
            // 记录删除点数的 devchangecount 值加 1 。
            *devchangecount = 1;
        }
    }
}

static __global__ void _thinPetPt2Ker(ImageCuda tempimg, ImageCuda outimg, int *devchangecount, uchar *dev_lut)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 1 || 
     dstr >= tempimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * tempimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = tempimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
        int posColumn2 = posColumn1 + tempimg.pitchBytes;
        int posColumn3 = posColumn2 + tempimg.pitchBytes;

        // p1 p2 p3
        // p8    p4
        // p7 p6 p5
        unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + posColumn1];
        unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1];
        unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1];
        unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2];
        unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3];
        unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3];
        unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3];
        unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2];

        uchar index= (p1==HIGH)*1 + (p2==HIGH)*2 + (p3==HIGH)*4 + (p4==HIGH)*8 + 
                     (p5==HIGH)*16 + (p6==HIGH)*32 + (p7==HIGH)*64 + (p8==HIGH)*128;

        if (dev_lut[index + 256]) {
            outimg.imgMeta.imgData[curpos] = LOW;
            // 记录删除点数的 devchangecount 值加 1 。
            *devchangecount = 1;
        }
    }
}

__host__ int Thinning::thinPetPt(Image *inimg, Image *outimg)
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
        return CUDA_ERROR;
    }

    uchar lut[512] = 
    {
        0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
        0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 

        0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 
        0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0
    };

    uchar *dev_lut;
    cudaerrcode = cudaMalloc((void **)&dev_lut, sizeof (uchar) * 512);
    if (cudaerrcode != cudaSuccess) 
        return CUDA_ERROR;

    cudaerrcode = cudaMemcpy(dev_lut, lut, sizeof (uchar) * 512,
                          cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess) 
        return CUDA_ERROR;

     // 生成暂存图像。
     errcode = ImageBasicOp::newImage(&tempimg);
     if (errcode != NO_ERROR)
         return errcode;
     errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                 inimg->height);
     if (errcode != NO_ERROR) {
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
     gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
     gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

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
         return CUDA_ERROR;
        }

         // copy ouimg to tempimg 
         cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                      outimg->imgData, outsubimgCud.deviceId, 
                                      outsubimgCud.pitchBytes * outimg->height);
        
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // 调用核函数，开始第一步细化操作。
         _thinPetPt1Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount, dev_lut);
         if (cudaGetLastError() != cudaSuccess) {
             // 核函数出错，结束迭代函数，释放申请的变量空间。
             // FAIL_THIN_IMAGE_FREE;
             return CUDA_ERROR;
         }

         // copy ouimg to tempimg 
         cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                      outimg->imgData, outsubimgCud.deviceId, 
                                      outsubimgCud.pitchBytes * outimg->height);
        
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // 调用核函数，开始第二步细化操作。
         _thinPetPt2Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount, dev_lut);
         if (cudaGetLastError() != cudaSuccess) {
             // 核函数出错，结束迭代函数，释放申请的变量空间 。
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

__constant__ uchar con_lut[512] = 
{
    0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
    0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 

    0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 
    0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0
};

static __global__ void _thinPetPtCon1Ker(ImageCuda tempimg, ImageCuda outimg, 
                                     int *devchangecount)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 1 || 
     dstr >= tempimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * tempimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = tempimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
        int posColumn2 = posColumn1 + tempimg.pitchBytes;
        int posColumn3 = posColumn2 + tempimg.pitchBytes;

        // p1 p2 p3
        // p8    p4
        // p7 p6 p5
        unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + posColumn1];
        unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1];
        unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1];
        unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2];
        unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3];
        unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3];
        unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3];
        unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2];

        uchar index= (p1==HIGH)*1 + (p2==HIGH)*2 + (p3==HIGH)*4 + (p4==HIGH)*8 + 
                     (p5==HIGH)*16 + (p6==HIGH)*32 + (p7==HIGH)*64 + (p8==HIGH)*128;

        if (con_lut[index]) {
            outimg.imgMeta.imgData[curpos] = LOW;
            // 记录删除点数的 devchangecount 值加 1 。
            *devchangecount = 1;
        }
    }
}

static __global__ void _thinPetPtCon2Ker(ImageCuda tempimg, ImageCuda outimg, 
                                     int *devchangecount)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 1 || 
     dstr >= tempimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * tempimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = tempimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
        int posColumn2 = posColumn1 + tempimg.pitchBytes;
        int posColumn3 = posColumn2 + tempimg.pitchBytes;

        // p1 p2 p3
        // p8    p4
        // p7 p6 p5
        unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + posColumn1];
        unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1];
        unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1];
        unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2];
        unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3];
        unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3];
        unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3];
        unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2];

        uchar index= (p1==HIGH)*1 + (p2==HIGH)*2 + (p3==HIGH)*4 + (p4==HIGH)*8 + 
                     (p5==HIGH)*16 + (p6==HIGH)*32 + (p7==HIGH)*64 + (p8==HIGH)*128;

        if (con_lut[index]) {
            outimg.imgMeta.imgData[curpos] = LOW;
            // 记录删除点数的 devchangecount 值加 1 。
            *devchangecount = 1;
        }
    }
}

__host__ int Thinning::thinPetPtCon(Image *inimg, Image *outimg)
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
        return CUDA_ERROR;
    }

     // 生成暂存图像。
     errcode = ImageBasicOp::newImage(&tempimg);
     if (errcode != NO_ERROR)
         return errcode;
     errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                 inimg->height);
     if (errcode != NO_ERROR) {
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
     gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
     gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

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
         return CUDA_ERROR;
        }

         // copy ouimg to tempimg 
         cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                      outimg->imgData, outsubimgCud.deviceId, 
                                      outsubimgCud.pitchBytes * outimg->height);
        
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // 调用核函数，开始第一步细化操作。
         _thinPetPtCon1Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
         if (cudaGetLastError() != cudaSuccess) {
             // 核函数出错，结束迭代函数，释放申请的变量空间。
             // FAIL_THIN_IMAGE_FREE;
             return CUDA_ERROR;
         }

         // copy ouimg to tempimg 
         cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                      outimg->imgData, outsubimgCud.deviceId, 
                                      outsubimgCud.pitchBytes * outimg->height);
        
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // 调用核函数，开始第二步细化操作。
         _thinPetPtCon2Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
         if (cudaGetLastError() != cudaSuccess) {
             // 核函数出错，结束迭代函数，释放申请的变量空间 。
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

static __global__ void _thinPetPtConFour1Ker(ImageCuda tempimg, ImageCuda outimg, 
                                     int *devchangecount)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 1 || 
     dstr >= tempimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * tempimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = tempimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
        int posColumn2 = posColumn1 + tempimg.pitchBytes;
        int posColumn3 = posColumn2 + tempimg.pitchBytes;

        // p1 p2 p3
        // p8    p4
        // p7 p6 p5
        unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + posColumn1];
        unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1];
        unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1];
        unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2];
        unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3];
        unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3];
        unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3];
        unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2];

        uchar index= (p1==HIGH)*1 + (p2==HIGH)*2 + (p3==HIGH)*4 + (p4==HIGH)*8 + 
                     (p5==HIGH)*16 + (p6==HIGH)*32 + (p7==HIGH)*64 + (p8==HIGH)*128;

        if (con_lut[index]) {
            outimg.imgMeta.imgData[curpos] = LOW;
            // 记录删除点数的 devchangecount 值加 1 。
            *devchangecount = 1;
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (++dstr >= outimg.imgMeta.height - 1)
            return ;
        curpos += outimg.pitchBytes;  

        // 获取当前像素点在图像中的绝对位置。
        outptr = outimg.imgMeta.imgData + curpos;
        if (*outptr != LOW) {
            // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
            // 防止下面细化处理时重复计算。
            int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
            int posColumn2 = posColumn1 + tempimg.pitchBytes;
            int posColumn3 = posColumn2 + tempimg.pitchBytes;

            // p1 p2 p3
            // p8    p4
            // p7 p6 p5
            unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + posColumn1];
            unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1];
            unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1];
            unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2];
            unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3];
            unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3];
            unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3];
            unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2];

            uchar index= (p1==HIGH)*1 + (p2==HIGH)*2 + (p3==HIGH)*4 + (p4==HIGH)*8 + 
                         (p5==HIGH)*16 + (p6==HIGH)*32 + (p7==HIGH)*64 + (p8==HIGH)*128;

            if (con_lut[index]) {
                outimg.imgMeta.imgData[curpos] = LOW;
                // 记录删除点数的 devchangecount 值加 1 。
                *devchangecount = 1;
            }
        }
    }
}

static __global__ void _thinPetPtConFour2Ker(ImageCuda tempimg, ImageCuda outimg, 
                                     int *devchangecount)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 1 || 
     dstr >= tempimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * tempimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = tempimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
        int posColumn2 = posColumn1 + tempimg.pitchBytes;
        int posColumn3 = posColumn2 + tempimg.pitchBytes;

        // p1 p2 p3
        // p8    p4
        // p7 p6 p5
        unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + posColumn1];
        unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1];
        unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1];
        unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2];
        unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3];
        unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3];
        unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3];
        unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2];

        uchar index= (p1==HIGH)*1 + (p2==HIGH)*2 + (p3==HIGH)*4 + (p4==HIGH)*8 + 
                     (p5==HIGH)*16 + (p6==HIGH)*32 + (p7==HIGH)*64 + (p8==HIGH)*128;

        if (con_lut[index]) {
            outimg.imgMeta.imgData[curpos] = LOW;
            // 记录删除点数的 devchangecount 值加 1 。
            *devchangecount = 1;
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (++dstr >= outimg.imgMeta.height - 1)
            return ;
        curpos += outimg.pitchBytes;  

        // 获取当前像素点在图像中的绝对位置。
        outptr = outimg.imgMeta.imgData + curpos;
        if (*outptr != LOW) {
            // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
            // 防止下面细化处理时重复计算。
            int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
            int posColumn2 = posColumn1 + tempimg.pitchBytes;
            int posColumn3 = posColumn2 + tempimg.pitchBytes;

            // p1 p2 p3
            // p8    p4
            // p7 p6 p5
            unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + posColumn1];
            unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1];
            unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1];
            unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2];
            unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3];
            unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3];
            unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3];
            unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2];

            uchar index= (p1==HIGH)*1 + (p2==HIGH)*2 + (p3==HIGH)*4 + (p4==HIGH)*8 + 
                         (p5==HIGH)*16 + (p6==HIGH)*32 + (p7==HIGH)*64 + (p8==HIGH)*128;

            if (con_lut[index + 256]) {
                outimg.imgMeta.imgData[curpos] = LOW;
                // 记录删除点数的 devchangecount 值加 1 。
                *devchangecount = 1;
            }
        }
    }
}

__host__ int Thinning::thinPetPtConFour(Image *inimg, Image *outimg)
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
        return CUDA_ERROR;
    }

     // 生成暂存图像。
     errcode = ImageBasicOp::newImage(&tempimg);
     if (errcode != NO_ERROR)
         return errcode;
     errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                 inimg->height);
     if (errcode != NO_ERROR) {
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
     gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
     gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 3 - 1) / blocksize.y * 3;

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
         return CUDA_ERROR;
        }

         // copy ouimg to tempimg 
         cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                      outimg->imgData, outsubimgCud.deviceId, 
                                      outsubimgCud.pitchBytes * outimg->height);
        
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // 调用核函数，开始第一步细化操作。
         _thinPetPtConFour1Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
         if (cudaGetLastError() != cudaSuccess) {
             // 核函数出错，结束迭代函数，释放申请的变量空间。
             // FAIL_THIN_IMAGE_FREE;
             return CUDA_ERROR;
         }

         // copy ouimg to tempimg 
         cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                      outimg->imgData, outsubimgCud.deviceId, 
                                      outsubimgCud.pitchBytes * outimg->height);
        
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // 调用核函数，开始第二步细化操作。
         _thinPetPtConFour2Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
         if (cudaGetLastError() != cudaSuccess) {
             // 核函数出错，结束迭代函数，释放申请的变量空间 。
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
}*/