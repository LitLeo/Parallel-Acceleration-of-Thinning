// Thinning.cu
// 实现二值图像的细化算法

#include "Thinning_gpu_four.h"
#include <iostream>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

 static __global__ void _thinGuoHallIter1Ker(ImageCuda outimg, ImageCuda tempimg, int *devchangecount,
                                      unsigned char highpixel, unsigned char lowpixel)
 {
     // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
     // column，r 表示 row ）。
     int dstc = blockIdx.x * blockDim.x + threadIdx.x;
     int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

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
     if (*outptr != lowpixel) {
         // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
         // 防止下面细化处理时重复计算。
         int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
         int posColumn2 = posColumn1 + tempimg.pitchBytes;
         int posColumn3 = posColumn2 + tempimg.pitchBytes;
    
         unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1] == highpixel;
         unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1] == highpixel;
         unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2] == highpixel;
         unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3] == highpixel;
         unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3] == highpixel;
         unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3] == highpixel;
         unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2] == highpixel;
         unsigned char p9 = tempimg.imgMeta.imgData[dstc-1 + posColumn1] == highpixel;
    
         int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                  (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
         int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
         int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
         int N  = N1 < N2 ? N1 : N2;
         int m  = ((p6 | p7 | !p9) & p8);

         if (C == 1 && (N >= 2 && N <= 3) & m == 0) {
             outimg.imgMeta.imgData[curpos] = lowpixel;
             // 记录删除点数的 devchangecount 值加 1 。
             //atomicAdd(devchangecount, 1);
             *devchangecount = 1;
         } 
     }

          // 处理剩下的3个点
     for (int i = 0; i < 3; ++i) {
        if (++dstr > tempimg.imgMeta.height - 1)
            return ;
        curpos += tempimg.pitchBytes;  

        // 获取当前像素点在图像中的绝对位置。
         outptr = tempimg.imgMeta.imgData + curpos;

         // 如果目标像素点的像素值为低像素, 则不进行细化处理。
         if (*outptr != lowpixel) {
             // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
             // 防止下面细化处理时重复计算。
             int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
             int posColumn2 = posColumn1 + tempimg.pitchBytes;
             int posColumn3 = posColumn2 + tempimg.pitchBytes;
        
             unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1] == highpixel;
             unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1] == highpixel;
             unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2] == highpixel;
             unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3] == highpixel;
             unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3] == highpixel;
             unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3] == highpixel;
             unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2] == highpixel;
             unsigned char p9 = tempimg.imgMeta.imgData[dstc-1 + posColumn1] == highpixel;
        
        
             int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                      (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
             int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
             int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
             int N  = N1 < N2 ? N1 : N2;
             int m  = ((p6 | p7 | !p9) & p8);

             if (C == 1 && (N >= 2 && N <= 3) & m == 0) {
                 outimg.imgMeta.imgData[curpos] = lowpixel;
                 // 记录删除点数的 devchangecount 值加 1 。
                 //atomicAdd(devchangecount, 1);
                 *devchangecount = 1;
             }
         }
     }
 }

 static __global__ void _thinGuoHallIter2Ker(ImageCuda tempimg, ImageCuda outimg, 
                                        int *devchangecount, unsigned char highpixel,
                                        unsigned char lowpixel)
 {
     // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
     // column，r 表示 row ）。
     int dstc = blockIdx.x * blockDim.x + threadIdx.x;
     int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

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
     if (*outptr != lowpixel) {
         // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
         // 防止下面细化处理时重复计算。
         int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
         int posColumn2 = posColumn1 + tempimg.pitchBytes;
         int posColumn3 = posColumn2 + tempimg.pitchBytes;
    
         unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1] == highpixel;
         unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1] == highpixel;
         unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2] == highpixel;
         unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3] == highpixel;
         unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3] == highpixel;
         unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3] == highpixel;
         unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2] == highpixel;
         unsigned char p9 = tempimg.imgMeta.imgData[dstc-1 + posColumn1] == highpixel;
    
    
         int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                  (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
         int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
         int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
         int N  = N1 < N2 ? N1 : N2;
         int m  = ((p2 | p3 | !p5) & p4);

         if (C == 1 && (N >= 2 && N <= 3) & m == 0) {
             outimg.imgMeta.imgData[curpos] = lowpixel;
             // 记录删除点数的 devchangecount 值加 1 。
             //atomicAdd(devchangecount, 1);
             *devchangecount = 1;
         }
     }

     // 处理剩下的3个点
     for (int i = 0; i < 3; ++i) {
        if (++dstr > tempimg.imgMeta.height - 1)
            return ;
        curpos += tempimg.pitchBytes;  

        // 获取当前像素点在图像中的绝对位置。
         outptr = tempimg.imgMeta.imgData + curpos;

         // 如果目标像素点的像素值为低像素, 则不进行细化处理。
         if (*outptr != lowpixel) {
             // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
             // 防止下面细化处理时重复计算。
             int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
             int posColumn2 = posColumn1 + tempimg.pitchBytes;
             int posColumn3 = posColumn2 + tempimg.pitchBytes;
        
             unsigned char p2 = tempimg.imgMeta.imgData[dstc+    posColumn1] == highpixel;
             unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + posColumn1] == highpixel;
             unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + posColumn2] == highpixel;
             unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + posColumn3] == highpixel;
             unsigned char p6 = tempimg.imgMeta.imgData[dstc+    posColumn3] == highpixel;
             unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + posColumn3] == highpixel;
             unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + posColumn2] == highpixel;
             unsigned char p9 = tempimg.imgMeta.imgData[dstc-1 + posColumn1] == highpixel;
        
        
             int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                      (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
             int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
             int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
             int N  = N1 < N2 ? N1 : N2;
             int m  = ((p2 | p3 | !p5) & p4);

             if (C == 1 && (N >= 2 && N <= 3) & m == 0) {
                 outimg.imgMeta.imgData[curpos] = lowpixel;
                 // 记录删除点数的 devchangecount 值加 1 。
                 //atomicAdd(devchangecount, 1);
                 *devchangecount = 1;
             }
         }
     }
 }

 __host__ int Thinning_gpu_four::thinGpuFour(Image *inimg, Image *outimg)
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
     gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) / blocksize.y * 4;

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
         _thinGuoHallIter1Ker<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud, devchangecount,
                                                         highPixel, lowPixel);
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
         _thinGuoHallIter2Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud,
                                                         devchangecount, highPixel, lowPixel);
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