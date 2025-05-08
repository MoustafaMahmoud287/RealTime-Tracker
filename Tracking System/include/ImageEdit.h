#ifndef IMAGEEDIT_H_INCLUDED
#define IMAGEEDIT_H_INCLUDED

#include <iostream>
#include <windows.h>
#include <math.h>
#include "model.h"

using namespace std;



#define M_PI 22.0/7.0
#define H_SIZE 256
#define ROWS 720
#define COLUMNS 1280
#define RegionSide 20

/*********************************************/


/**************Image Matrices && other variables******************/
class Color {// assume 24 bit format of BitMap
private:
    BYTE Blue;
    BYTE Green;
    BYTE Red;
public:
    Color() : Blue(0), Green(0), Red(0) {}
    Color(BYTE B, BYTE G, BYTE R) : Blue(B), Green(G), Red(R) {}
    Color& operator=(const Color& c) { Blue = c.Blue; Green = c.Green; Red = c.Red; return *this; }
    BOOL operator ==(const Color& c)const { return Blue == c.Blue && Green == c.Green && Red == c.Red; }
    BYTE& operator[](BYTE index) {
        if (index > 2) cout << "index out of bounds index % 3 returned ";
        index %= 3;
        if (index == 0) return Blue; if (index == 1) return Green; return Red;
    }
};

struct LUV {
    float L;
    float U;
    float V;
    __host__ __device__ void operator = (LUV luv) {
        L = luv.L;
        U = luv.U;
        V = luv.V;
    }
};

struct U_V_ {
    float U;
    float V;
    __host__ __device__ void operator = (U_V_ uv) {
        U = uv.U;
        V = uv.V;
    }
};

class Image {

public:

    /************data members*************/
    model Mask;
    LUV AvgColorLuv;
    U_V_ AvgColorUV;
    BYTE* OriginalImage;
    BYTE* BinaryImage;
    INT32* Histogram;
    LUV* ColorSum1;
    U_V_* ColorSum2;
    UINT32* PixelCount;
    point* arr;
    point* norm_arr;
    bool first;
    /************************************/

    /************Constructor && Destructor***************/
    Image();
    ~Image();
    /***************************************************/

    /*****************main Functions*******************/
    void AccGetAverageColor1();

private:
};

#endif