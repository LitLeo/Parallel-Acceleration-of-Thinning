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
                                     int *devchangecount)
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
        if (iteration & 1){
            // 调用核函数，开始第一步细化操作。
            _thinAhmedKer<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud,
                                                   devchangecount);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }
        }else{
            // 调用核函数，开始第一步细化操作。
            _thinAhmedKer<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud,
                                                   devchangecount);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
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

    return NO_ERROR;
}

// 直接并行化
// 线程数，处理多少个点有多少线程数
__host__ int Thinning::thinAhmedMultiGPU(Image *inimg, Image *outimg)
{
    // 局部变量，错误码。
    int errcode;

    // 检查输入图像，输出图像是否为空。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;
    // 获得设备数量
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    Image **d_outsubimg, **tempsubimg;
    d_outsubimg = new Image*[deviceCount];
    tempsubimg = new Image*[deviceCount];

    ImageCuda *insubimgCud, *d_outsubimgCud, *tempsubimgCud, *h_outsubimgCud;
    insubimgCud = new ImageCuda[deviceCount];
    h_outsubimgCud = new ImageCuda[deviceCount];

    tempsubimgCud = new ImageCuda[deviceCount];
    d_outsubimgCud = new ImageCuda[deviceCount];

    int *changeCount = new int[deviceCount];

    int **devchangecount = new int*[deviceCount];

    cudaStream_t *stream = new cudaStream_t[deviceCount];

    // 对图像进行分割和内存申请
    int height = inimg->height / deviceCount;
    for(int i = 0; i < deviceCount; ++i) {
        changeCount[i] = 1;
        ImageBasicOp::newImage(&d_outsubimg[i]);
        ImageBasicOp::newImage(&tempsubimg[i]);

        // 计算每个设备上需要处理的图像大小，对图像进行分割。
        outimg->roiY1 = inimg->roiY1 = i * height;
        outimg->roiY2 = inimg->roiY2 = (i+1) * height;

        // 如果图像高度不是设备数的倍数，则将余数放到最后一个设备上处理。
        if(i == (deviceCount - 1) && inimg->height % deviceCount != 0) {
            outimg->roiY2 = inimg->roiY2 = (i+1) * height + inimg->height % deviceCount;
        } else
            outimg->roiY2 = inimg->roiY2 = (i+1) * height;

        errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud[i]);
        if (errcode != NO_ERROR) {
            return errcode;
        }
        errcode = ImageBasicOp::roiSubImage(outimg, &h_outsubimgCud[i]);
        if (errcode != NO_ERROR) {
            return errcode;
        }

        // 切换设备，在不同设备内进行内存申请和拷贝。
        cudaSetDevice(i);

        // 为tempimg和outing在device端申请内存。
        errcode = ImageBasicOp::makeAtCurrentDevice(tempsubimg[i],
                                                    insubimgCud[i].imgMeta.width,
                                                    insubimgCud[i].imgMeta.height);
        if (errcode != NO_ERROR) {
            return errcode;
        }

        errcode = ImageBasicOp::makeAtCurrentDevice(d_outsubimg[i],
                                                    insubimgCud[i].imgMeta.width,
                                                    insubimgCud[i].imgMeta.height);
        if (errcode != NO_ERROR) {
            return errcode;
        }

        errcode = ImageBasicOp::roiSubImage(d_outsubimg[i], &d_outsubimgCud[i]);
        if (errcode != NO_ERROR) {
            return errcode;
        }
        errcode = ImageBasicOp::roiSubImage(tempsubimg[i], &tempsubimgCud[i]);
        if (errcode != NO_ERROR) {
            return errcode;
        }

        // 在不同设备上为devchangecount申请空间并初始化为0.
        checkCudaErrors(cudaMalloc(&devchangecount[i], sizeof(int)));

        cudaStreamCreate(&stream[i]);
    }

    for(int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        // 将inimg拷贝到outimg中
        checkCudaErrors(cudaMemcpy2DAsync (d_outsubimgCud[i].imgMeta.imgData, d_outsubimgCud[i].pitchBytes,
                                    insubimgCud[i].imgMeta.imgData, insubimgCud[i].imgMeta.width,
                                    insubimgCud[i].imgMeta.width * sizeof (unsigned char),
                                    insubimgCud[i].imgMeta.height,
                                    cudaMemcpyHostToDevice, stream[i]));

        // 将outinimg 拷贝到 tempimg中
        checkCudaErrors(cudaMemcpy2DAsync (tempsubimgCud[i].imgMeta.imgData, tempsubimgCud[i].pitchBytes,
                                    d_outsubimgCud[i].imgMeta.imgData, d_outsubimgCud[i].pitchBytes,
                                    d_outsubimgCud[i].pitchBytes * sizeof (unsigned char),
                                    d_outsubimgCud[i].imgMeta.height,
                                    cudaMemcpyDeviceToDevice, stream[i]));
    }

    bool all_done_flag = true;
    int iteration = 0;
    // 记录细化点数的变量，位于 host 端。
    do {
        all_done_flag = true;
        // 在每个GPU上执行迭代细化算法
        for(int i = 0; i < deviceCount; ++i) {
            // 如果上次迭代删除的点数不为0。
            if(changeCount != 0) {
                cudaSetDevice(i);

                // 每次迭代开始前将 devchangecount 赋值为 0
                checkCudaErrors(cudaMemset(devchangecount[i], 0, sizeof(int)));

                // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
                dim3 gridsize, blocksize;
                blocksize.x = DEF_BLOCK_X;
                blocksize.y = DEF_BLOCK_Y;
                gridsize.x = (d_outsubimgCud[i].imgMeta.width + blocksize.x - 1) / blocksize.x;
                gridsize.y = (d_outsubimgCud[i].imgMeta.height + blocksize.y - 1) / blocksize.y;

                // 如果已经进行了奇数次迭代，那么现在源图像是在outimg上，输出图像应该在tempsubimgCud上
                if (iteration & 1){
                    // 调用核函数，开始第一步细化操作。
                    _thinAhmedKer<<<gridsize, blocksize, 0, stream[i]>>>(d_outsubimgCud[i], tempsubimgCud[i],
                                                                         devchangecount[i]);
                    if (cudaGetLastError() != cudaSuccess) {
                        // 核函数出错，结束迭代函数，释放申请的变量空间。
                        return CUDA_ERROR;
                    }
                }else{
                    // 调用核函数，开始第一步细化操作。
                    _thinAhmedKer<<<gridsize, blocksize, 0, stream[i]>>>(tempsubimgCud[i], d_outsubimgCud[i],
                                                           devchangecount[i]);
                    if (cudaGetLastError() != cudaSuccess) {
                        // 核函数出错，结束迭代函数，释放申请的变量空间。
                        return CUDA_ERROR;
                    }
                }

                // 将位于 device 端的 devchangecount 拷贝到 host 端上的 changeCount
                // 变量，进行迭代判断。
                checkCudaErrors(cudaMemcpy(&changeCount[i], devchangecount[i], sizeof (int),
                                                         cudaMemcpyDeviceToHost));

                // 如果被删除的点数不为0，则代表还有下一次迭代
                if(changeCount[i] != 0) {
                    all_done_flag = false;
                }
            }
        }

        iteration ++;

    } while (!all_done_flag);

    // 细化结束后释放申请的变量空间。
    for(unsigned i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(stream[i]);
    }
    for(unsigned i = 0; i < deviceCount; ++i) {
        // 将d_outimg拷贝到h_outimg中
        checkCudaErrors(cudaMemcpy2DAsync (h_outsubimgCud[i].imgMeta.imgData, h_outsubimgCud[i].pitchBytes,
                                    d_outsubimgCud[i].imgMeta.imgData, d_outsubimgCud[i].pitchBytes,
                                    d_outsubimgCud[i].imgMeta.width * sizeof (unsigned char),
                                    d_outsubimgCud[i].imgMeta.height,
                                    cudaMemcpyHostToDevice, stream[i]));
        cudaFree(tempsubimgCud[i].imgMeta.imgData);
        cudaFree(devchangecount[i]);
        cudaStreamDestroy(stream[i]);
    }

    delete [] insubimgCud;
    delete [] d_outsubimgCud;
    delete [] h_outsubimgCud;
    delete [] tempsubimgCud;
    delete [] changeCount;
    delete [] devchangecount;
    delete [] stream;

    return NO_ERROR;
}

// GPU版本2，优化分支，使用Pattern表法，Pattern表位于global内存中
static __global__ void _thinAhmedPtKer(ImageCuda tempimg, ImageCuda outimg,
                                       int *devchangecount, unsigned char *dev_lut)
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

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (isHigh1(*outptr)) {
        // 如果在滚动数组中，发现上一轮发生了删点，说明此时实质上已经是Low了
        // 因此刷新，并返回
        if (isLow1(tempimg.imgMeta.imgData[curpos])){
            outimg.imgMeta.imgData[curpos] = LOW;
            // 因为该修改不属于本次迭代的修改，所以不修改changeCount
            return;
        }
        unsigned char x1, x2, x3, x4, x5, x6, x7, x8;
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
                                                     devchangecount, dev_lut);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }
        }else{
            // 调用核函数，开始第一步细化操作。
            _thinAhmedPtKer<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud,
                                                     devchangecount, dev_lut);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
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
    return NO_ERROR;
}


/*
// 共享内存优化版本，Pattern表位于global内存中
static __global__ void _thinAhmedSharedKer(ImageCuda tempimg, ImageCuda outimg,
                                       int *devchangecount, unsigned char *dev_lut,
                                       unsigned s_size)
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

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;

    // 线程在线程块内的相对坐标
    int thread_index = threadIdx.x + threadIdx.y * blockDim.x;
    // 线程块大小
    int block_size = blockDim.x * blockDim.y;

    // 对于AW细化算法，线程块所需共享内存的宽度为线程块宽度+4
    int s_width = blockDim.x + 2;

    int idx_of_thread_in_shared = threadIdx.x + 1 + (threadIdx.y + 1) * blockDim.x;

    unsigned char* block_pointer = tempimg.imgMeta.imgData + blockIdx.x * blockDim.x +
                                   blockIdx.y * blockDim.y;

    // 共享内存申请和拷贝
    extern __shared__ unsigned char s_imgData[];

    for(unsigned k = thread_index; k < s_size; k += block_size) {
        int i = k / s_width;
        int j = k % s_width;

        s_imgData[k] = block_pointer[i * outimg.pitchBytes + j];
    }
    __syncthreads();

     // 共享内存拷贝数据判断
    if(threadIdx.x == 31 && threadIdx.y  == 7) {
        // for(unsigned w = 0; w < s_width; ++w) {
            // for(unsigned h = 0; h < s_width + 2; ++h) {
            //     if(s_imgData[h * s_width + w] != block_pointer[h * outimg.pitchBytes + w]) {
            //         printf("shared copy failed! id = %d\n", h * s_width + w);
            //         break;
            //     }
            // }
            for(unsigned i = 0; i < s_size; ++i) {
                printf("%d ", s_imgData[i]);
            }
            printf("\n");
            printf("\n");
            printf("\n");
        // }
    }
    return ;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (isHigh1(*outptr)) {
        // 如果在滚动数组中，发现上一轮发生了删点，说明此时实质上已经是Low了
        // 因此刷新，并返回
        if (isLow1(tempimg.imgMeta.imgData[curpos])){
            outimg.imgMeta.imgData[curpos] = LOW;
            // 因为该修改不属于本次迭代的修改，所以不修改changeCount
            return;
        }

        // s_imgData[thread_index] = tempimg.imgMeta.imgData[r - 1 + thread_index / s_width
        //                           * outimg.pitchBytes + c - 1 + thread_index % s_width];

        // int next_thread_index = thread_index + block_size;
        // if(next_thread_index < s_size) {
        //     s_imgData[next_thread_index] = tempimg.imgMeta.imgData[r - 1 + next_thread_index / s_width
        //                           * outimg.pitchBytes + c - 1 + next_thread_index % s_width];
        // }

        unsigned char x1, x2, x3, x4, x5, x6, x7, x8;
        x1 = tempimg.imgMeta.imgData[idx_of_thread_in_shared - s_width - 1];
        x2 = tempimg.imgMeta.imgData[idx_of_thread_in_shared - s_width];
        x3 = tempimg.imgMeta.imgData[idx_of_thread_in_shared - s_width + 1];
        x4 = tempimg.imgMeta.imgData[idx_of_thread_in_shared - 1];
        x5 = tempimg.imgMeta.imgData[idx_of_thread_in_shared + 1];
        x6 = tempimg.imgMeta.imgData[idx_of_thread_in_shared + s_width - 1];
        x7 = tempimg.imgMeta.imgData[idx_of_thread_in_shared + s_width];
        x8 = tempimg.imgMeta.imgData[idx_of_thread_in_shared + s_width + 1];
        unsigned char x9,x10,x11;
        if(isHigh1(x7)) {
            x9 = tempimg.imgMeta.imgData[idx_of_thread_in_shared + s_width * 2 - 1];
            x10 = tempimg.imgMeta.imgData[idx_of_thread_in_shared + s_width * 2];
            x11 = tempimg.imgMeta.imgData[idx_of_thread_in_shared + s_width * 2 + 1];

            if (isHigh5(x4,x5,x6,x7,x8) && isLow2(x2,x10) ||
                isHigh3(x6,x7,x9) && isLow8(x1,x2,x3,x4,x5,x8,x10,x11) ||
                isHigh3(x7,x8,x11) && isLow8(x1,x2,x3,x4,x5,x6,x9,x10))
                    return ;
        }
        if(isHigh1(x2)) {
            // w is down
            x9 = tempimg.imgMeta.imgData[idx_of_thread_in_shared - s_width * 2 - 1];
            x10 = tempimg.imgMeta.imgData[idx_of_thread_in_shared - s_width * 2];
            x11 = tempimg.imgMeta.imgData[idx_of_thread_in_shared - s_width * 2 + 1];

            if (isHigh5(x1,x2,x3,x4,x5) && isLow2(x7,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                return ;
            } else if (isHigh3(x1,x2,x9) && isLow8(x3,x4,x5,x6,x7,x8,x10,x11) ||
                       isHigh3(x2,x3,x11) && isLow8(x1,x4,x5,x6,x7,x8,x9,x10))
                return ;
        }
        if(isHigh1(x5)) {
            x9 = tempimg.imgMeta.imgData[idx_of_thread_in_shared - s_width + 2];
            x10 = tempimg.imgMeta.imgData[idx_of_thread_in_shared + 2];
            x11 = tempimg.imgMeta.imgData[idx_of_thread_in_shared + s_width + 2];

            if (isHigh5(x2,x3,x5,x7,x8) && isLow2(x4,x10) ||
                isHigh3(x5,x8,x11) && isLow8(x1,x2,x3,x4,x6,x7,x9,x10) ||
                isHigh3(x3,x5,x9) && isLow8(x1,x2,x4,x6,x7,x8,x10,x11))
                return ;
        }
        if(isHigh1(x4)){
            x9 = tempimg.imgMeta.imgData[idx_of_thread_in_shared - s_width - 2];
            x10 = tempimg.imgMeta.imgData[idx_of_thread_in_shared - 2];
            x11 = tempimg.imgMeta.imgData[idx_of_thread_in_shared + s_width - 2];
            if (isHigh5(x1,x2,x4,x6,x7) && isLow2(x5,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
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
        }
    }
}

__host__ int Thinning::thinAhmedShared(Image *inimg, Image *outimg)
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

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    gridsize.x = 1;
    gridsize.y = 1;

    // 赋值为 1，以便开始第一次迭代。
    changeCount = 1;

    // 迭代次数，为了计算滚动的状态
    iteration = 0;

    int s_size = (blocksize.x + 2) * (blocksize.y + 4); 

    // copy ouimg to tempimg
    cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId,
                                  outimg->imgData, outsubimgCud.deviceId,
                                  outsubimgCud.pitchBytes * outimg->height);

    if (cudaerrcode != cudaSuccess) {
        // FAIL_THIN_IMAGE_FREE;
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
            _thinAhmedSharedKer<<<gridsize, blocksize, s_size>>>(outsubimgCud, tempsubimgCud,
                                                     devchangecount, dev_lut, s_size);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
                // FAIL_THIN_IMAGE_FREE;
                return CUDA_ERROR;
            }
        }else{
            // 调用核函数，开始第一步细化操作。
            _thinAhmedSharedKer<<<gridsize, blocksize, s_size>>>(tempsubimgCud, outsubimgCud,
                                                     devchangecount, dev_lut, s_size);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束迭代函数，释放申请的变量空间。
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
    return NO_ERROR;
}
*/