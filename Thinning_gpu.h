﻿#ifndef __THINNING_GPU_H__
#define __THINNING_GPU_H__

#include "Image.h"
#include "ErrorCode.h"

class Thinning_gpu {

protected:

    unsigned char highPixel;
    
    unsigned char lowPixel;
    
public:

    // 构造函数：Thinning
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    Thinning_gpu()
    {
        this->highPixel = 255;  // 高像素值默认为 255。
        this->lowPixel = 0;     // 低像素值默认为 0。 
    }

    // 构造函数：Thinning
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    Thinning_gpu(
            unsigned char highpixel,  // 高像素 
            unsigned char lowpixel    // 低像素
    ) {
        this->highPixel = 255;        // 高像素值默认为 255。
        this->lowPixel = 0;           // 低像素值默认为 0。

        // 根据参数列表中的值设定成员变量的初值。
        this->setHighLowPixel(highPixel, lowPixel);
    }

    // 成员函数：getHighPixel（获取高像素的值）
    // 获取成员变量 highPixel 的值。
    __host__ __device__ unsigned char  // 返回值：返回 hignPixel 的值。
    getHighPixel() const
    { 
        // 返回 highPixel 成员变量的值。
        return highPixel;   
    }

    // 成员函数：setHighPixel（设置高像素）
    // 设置成员变量 highPixel 的值。
    __host__ __device__ int          // 返回值：若函数正确执行，返回 NO_ERROR。
    setHighPixel(                                                       
            unsigned char highpixel  // 高像素的像素值
    ) {
        // 如果高像素和低像素相等，则报错。
        if (highpixel == this->lowPixel)
            return INVALID_DATA;

        // 将转换器内的原先高像素位置为 false。此做法可使转换器的只有一个高像素
        // 位。
        // this->imgCon.clearConvertFlag(this->highPixel);

        // 将 highPixel 成员变量赋成新值
        this->highPixel = highpixel;

        // 将转换器的高像素与本算法同步。
        // this->imgCon.setHighPixel(this->highPixel);

        // 将转换器的标记数组高像素位置为 true。
        // this->imgCon.setConvertFlag(this->highPixel);

        return NO_ERROR;
    }

    // 成员函数：getLowPixel（获取低像素的值）
    // 获取成员变量 lowPixel 的值。
    __host__ __device__ unsigned char  // 返回值：返回 lowPixel 的值。
    getLowPixel() const
    { 
        // 返回 lowPixel 成员变量的值。
        return lowPixel;   
    }

    // 成员函数：setLowPixel（设置低像素）
    // 设置成员变量 lowPixel 的值。
    __host__ __device__ int         // 返回值：若函数正确执行，返回 NO_ERROR。
    setLowPixel(
            unsigned char lowpixel  // 低像素的像素值
    ) {
        // 如果高像素和低像素相等，则报错。
        if (this->highPixel == lowpixel)
            return INVALID_DATA;

        // 将 lowPixel 成员变量赋成新值。
        this->lowPixel = lowpixel;

        // 将转换器的低像素与本算法同步。
        // this->imgCon.setLowPixel(this->lowPixel);

        return NO_ERROR;
    }

    // 成员函数：setHighLowPixel（设置高低像素）
    // 设置成员变量 highPixel 和 lowPixel 的值。
    __host__ __device__ int           // 返回值：函数正确执行，返回 NO_ERROR。
    setHighLowPixel(
            unsigned char highpixel,  // 高像素的像素值
            unsigned char lowpixel    // 低像素的像素值
    ) {
        // 如果高像素和低像素相等，则报错。
        if (highpixel == lowpixel)
            return INVALID_DATA;

        // 将转换器内的原先高像素位置为 false。此做法可使转换器的只有一个高像素
        // 位。
        // this->imgCon.clearConvertFlag(this->highPixel);

        // 将 highPixel 成员变量赋成新值。
        this->highPixel = highpixel;

        // 将 lowPixel 成员变量赋成新值。
        this->lowPixel = lowpixel;

        // 将转换器的高像素和低像素与本算法同步。
        // this->imgCon.setHighPixel(this->highPixel); 
        // this->imgCon.setLowPixel(this->lowPixel);

        // 将转换器的标记数组高像素位置为 true。
        // this->imgCon.setConvertFlag(this->highPixel);

        return NO_ERROR;
    }

	__host__ int 
	thinGuoHall(
			Image *inimg,
			Image *outimg
	);


};

#endif

