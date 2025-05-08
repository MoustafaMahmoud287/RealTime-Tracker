#include "../include/ImageEdit.h"
#include <iostream>
#include <thread>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <texture_indirect_functions.h>
#include <crt/device_functions.h>

using namespace std;
int MaxLength = 256 * 256 * 256;
float* DataL = new float[256 * 256 * 256];
float* DataU = new float[256 * 256 * 256];
float* DataV = new float[256 * 256 * 256];
float* DataUdash = new float[256 * 256 * 256];
float* DataVdash = new float[256 * 256 * 256];

/*********************************************/
double fff(double ft) {
    ft = (ft > 0.008856f) ? powf(ft, 1.0f / 3.0f) : (7.787f * ft + (16.0f / 116.0f));
    return ft;
}
float clamp(float val, float mi, float ma) {
    if (val > ma) return ma;
    if (val < mi) return mi;
    return val;
}
float gammaCorrect(float val) {
    constexpr float kGammaThreshold = 0.04045f;
    constexpr float kGammaDivisor = 12.92f;
    constexpr float kGammaPower = 2.4f;
    constexpr float kEpsilon = 1e-6f;
    return (val <= kGammaThreshold) ?
        (val / kGammaDivisor) :
        powf((val + 0.055f) / 1.055f, kGammaPower);
};
void ReadFile() {
    constexpr float kGammaThreshold = 0.04045f;
    constexpr float kGammaDivisor = 12.92f;
    constexpr float kGammaPower = 2.4f;
    constexpr float kEpsilon = 1e-6f;

    for (int yy = 0; yy < 256; yy++) {
        for (int uu = 0; uu < 256; uu++) {
            for (int vv = 0; vv < 256; vv++) {
                // YCbCr (BT.601) to RGB
                float R = yy + 1.402f * (vv - 128);
                float G = yy - 0.34414f * (uu - 128) - 0.71414f * (vv - 128);
                float B = yy + 1.772f * (uu - 128);

                // Clamp and round
                R = roundf(clamp(R, 0.0f, 255.0f));
                G = roundf(clamp(G, 0.0f, 255.0f));
                B = roundf(clamp(B, 0.0f, 255.0f));

                // sRGB to linear RGB
                R /= 255.0f;
                G /= 255.0f;
                B /= 255.0f;

                R = gammaCorrect(R);
                G = gammaCorrect(G);
                B = gammaCorrect(B);

                // Linear RGB to XYZ (D65)
                float X = R * 0.412453f + G * 0.35758f + B * 0.180423f;
                float Y = R * 0.212671f + G * 0.71516f + B * 0.072169f;
                float Z = R * 0.019334f + G * 0.119193f + B * 0.950227f;

                // XYZ to u'v'
                float denom = X + 15.0f * Y + 3.0f * Z;
                denom = (denom < kEpsilon) ? kEpsilon : denom;
                float u_prime = 4.0f * X / denom;
                float v_prime = 9.0f * Y / denom;

                // XYZ to LUV (D65 white point)
                double Xn = 0.95047, Yn = 1.0, Zn = 1.08883;
                double fx = fff(X / Xn);
                double fy = fff(Y / Yn);
                double fz = fff(Z / Zn);
                float L = 116.0f * fy - 16.0f;
                float u = 500.0 * (fx - fy);
                float v = 200.0 * (fy - fz);

                // Store results
                DataUdash[256 * 256 * yy + 256 * uu + vv] = u_prime;
                DataVdash[256 * 256 * yy + 256 * uu + vv] = v_prime;
                DataL[256 * 256 * yy + 256 * uu + vv] = L;
                DataU[256 * 256 * yy + 256 * uu + vv] = u;
                DataV[256 * 256 * yy + 256 * uu + vv] = v;
            }
        }
    }
}
/************Cuda Global Variables**************/
__constant__ LUV CudaAvgColorLuv;
__constant__ U_V_ CudaAvgColorUV;
int BlocksPerGrid = 32;
int max_mask_size = 30;
float* __restrict__  lTable;
float* __restrict__  uTable;
float* __restrict__  vTable;
float* __restrict__  _vTable;
float* __restrict__  _uTable;
LUV* DevImageLuv = nullptr;
LUV* DevColorSum1 = nullptr;
U_V_* DevImageUV = nullptr;
U_V_* DevColorSum2 = nullptr;
UINT32* DevPixelCount = nullptr;
BYTE* DevBinaryImage = nullptr;
BYTE* DevYUVImage = nullptr;
point* DevArr = nullptr;
point* DevNorm = nullptr;
INT32* DevHist = nullptr;
/**********************************************/

/***********Cuda Initialization Function*************/
void CudaInitForImage() {

    cudaMalloc((void**)&DevImageLuv, sizeof(LUV) * ROWS * COLUMNS);
    cudaMalloc((void**)&DevImageUV, sizeof(U_V_) * ROWS * COLUMNS);
    cudaMalloc((void**)&DevBinaryImage, sizeof(BYTE) * ROWS * COLUMNS);
    cudaMalloc((void**)&DevYUVImage, sizeof(BYTE) * ROWS * COLUMNS * 2);
    cudaMalloc((void**)&DevHist, sizeof(INT32) * 256);
    cudaMalloc((void**)&lTable, sizeof(float) * 256 * 256 * 256);
    cudaMalloc((void**)&uTable, sizeof(float) * 256 * 256 * 256);
    cudaMalloc((void**)&vTable, sizeof(float) * 256 * 256 * 256);
    cudaMalloc((void**)&_uTable, sizeof(float) * 256 * 256 * 256);
    cudaMalloc((void**)&_vTable, sizeof(float) * 256 * 256 * 256);
    cudaMalloc((void**)&DevColorSum1, sizeof(LUV) * BlocksPerGrid);
    cudaMalloc((void**)&DevColorSum2, sizeof(U_V_) * BlocksPerGrid);
    cudaMalloc((void**)&DevPixelCount, sizeof(UINT32) * BlocksPerGrid);
    cudaMalloc((void**)&DevArr, sizeof(point) * 16 * max_mask_size);
    cudaMalloc((void**)&DevNorm, sizeof(point) * 8 * max_mask_size);

    cudaMemcpy(lTable, DataL, 256 * 256 * 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(uTable, DataU, 256 * 256 * 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vTable, DataV, 256 * 256 * 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(_uTable, DataUdash, 256 * 256 * 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(_vTable, DataVdash, 256 * 256 * 256 * sizeof(float), cudaMemcpyHostToDevice);
}

void CudaForImageEnd() {

    cudaFree(DevColorSum1);
    cudaFree(DevColorSum2);
    cudaFree(DevImageLuv);
    cudaFree(DevImageUV);
    cudaFree(DevPixelCount);
    cudaFree(DevBinaryImage);
    cudaFree(DevYUVImage);
    cudaFree(DevArr);
    cudaFree(DevNorm);
    cudaFree(DevHist);
    cudaFree(lTable);
    cudaFree(uTable);
    cudaFree(vTable);
    cudaFree(_uTable);
    cudaFree(_vTable);
}
/****************************************************/


/************Constructor && Destructor***************/
Image::Image() :Mask() {
    ReadFile();
    CudaInitForImage();
    AvgColorUV.U = 0;
    AvgColorUV.V = 0;
    AvgColorLuv.L = 0;
    AvgColorLuv.U = 0;
    AvgColorLuv.V = 0;
    first = true;
    OriginalImage = new BYTE[ROWS * COLUMNS * 2];
    BinaryImage = new BYTE[ROWS * COLUMNS];
    Histogram = new INT32[H_SIZE];
    ColorSum1 = new LUV[BlocksPerGrid];
    ColorSum2 = new U_V_[BlocksPerGrid];
    PixelCount = new UINT32[BlocksPerGrid];
    arr = new point[16 * max_mask_size];
    norm_arr = new point[8 * max_mask_size];
}

Image::~Image() {
    delete[] OriginalImage;
    delete[] BinaryImage;
    delete[] Histogram;
    delete[] ColorSum1;
    delete[] ColorSum2;
    delete[] PixelCount;
    delete[] arr;
    delete[] norm_arr;
    CudaForImageEnd();
}
/****************************************************/


__device__ float Area(point& p1, point& p2, point& p3) {
    return abs((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2.0);
}

__device__ bool InRangeOfAverageColor(LUV* ImageLuv, U_V_* ImageUV, int index, bool binary = true) {
    // Reference color (average)
    double L1 = static_cast<double>(CudaAvgColorLuv.L);
    double a1 = static_cast<double>(CudaAvgColorLuv.U);
    double b1 = static_cast<double>(CudaAvgColorLuv.V);

    // Sample color (current pixel)
    double L2 = static_cast<double>(ImageLuv[index].L);
    double a2 = static_cast<double>(ImageLuv[index].U);
    double b2 = static_cast<double>(ImageLuv[index].V);

    // --- CIEDE2000 Calculation ---
    // Step 1: Convert Lab to LCh (Lightness, Chroma, Hue)
    const double C1 = sqrt(a1 * a1 + b1 * b1);
    const double C2 = sqrt(a2 * a2 + b2 * b2);
    const double C_avg = (C1 + C2) / 2.0;

    const double G = 0.5 * (1.0 - sqrt(pow(C_avg, 7.0) / (pow(C_avg, 7.0) + pow(25.0, 7.0))));
    const double a1_prime = a1 * (1.0 + G);
    const double a2_prime = a2 * (1.0 + G);

    const double C1_prime = sqrt(a1_prime * a1_prime + b1 * b1);
    const double C2_prime = sqrt(a2_prime * a2_prime + b2 * b2);

    // Ensure consistent double-precision in atan2
    double h1_prime = (b1 == 0.0 && a1_prime == 0.0) ? 0.0 : atan2(b1, a1_prime);
    double h2_prime = (b2 == 0.0 && a2_prime == 0.0) ? 0.0 : atan2(b2, a2_prime);

    // Convert to degrees and ensure positive angle
    h1_prime = h1_prime * (180.0 / M_PI);
    h2_prime = h2_prime * (180.0 / M_PI);
    if (h1_prime < 0.0) h1_prime += 360.0;
    if (h2_prime < 0.0) h2_prime += 360.0;

    // Step 2: Compute Delta L', C', H'
    const double delta_L_prime = L2 - L1;
    const double delta_C_prime = C2_prime - C1_prime;

    double delta_h_prime;
    if (C1_prime * C2_prime == 0.0) {
        delta_h_prime = 0.0;
    }
    else if (fabs(h2_prime - h1_prime) <= 180.0) {
        delta_h_prime = h2_prime - h1_prime;
    }
    else if (h2_prime - h1_prime > 180.0) {
        delta_h_prime = h2_prime - h1_prime - 360.0;
    }
    else {
        delta_h_prime = h2_prime - h1_prime + 360.0;
    }

    const double delta_H_prime = 2.0 * sqrt(C1_prime * C2_prime) * sin(delta_h_prime * (M_PI / 360.0));

    // Step 3: Calculate weighted differences
    const double L_avg_prime = (L1 + L2) / 2.0;
    const double C_avg_prime = (C1_prime + C2_prime) / 2.0;

    double H_avg_prime;
    if (C1_prime * C2_prime == 0.0) {
        H_avg_prime = h1_prime + h2_prime;
    }
    else if (fabs(h1_prime - h2_prime) <= 180.0) {
        H_avg_prime = (h1_prime + h2_prime) / 2.0;
    }
    else if (h1_prime + h2_prime < 360.0) {
        H_avg_prime = (h1_prime + h2_prime + 360.0) / 2.0;
    }
    else {
        H_avg_prime = (h1_prime + h2_prime - 360.0) / 2.0;
    }

    const double T = 1.0 - 0.17 * cos((H_avg_prime - 30.0) * (M_PI / 180.0)) +
        0.24 * cos(2.0 * H_avg_prime * (M_PI / 180.0)) +
        0.32 * cos((3.0 * H_avg_prime + 6.0) * (M_PI / 180.0)) -
        0.20 * cos((4.0 * H_avg_prime - 63.0) * (M_PI / 180.0));

    const double S_L = 1.0 + ((0.015 * pow(L_avg_prime - 50.0, 2.0)) / sqrt(20.0 + pow(L_avg_prime - 50.0, 2.0)));
    const double S_C = 1.0 + 0.045 * C_avg_prime;
    const double S_H = 1.0 + 0.015 * C_avg_prime * T;

    const double R_T = -2.0 * sqrt(pow(C_avg_prime, 7.0) / (pow(C_avg_prime, 7.0) + pow(25.0, 7.0))) *
        sin(60.0 * exp(-pow((H_avg_prime - 275.0) / 25.0, 2.0)) * (M_PI / 180.0));

    // Final CIEDE2000 Delta E
    const double delta_E = sqrt(
        pow(delta_L_prime / S_L, 2.0) +
        pow(delta_C_prime / S_C, 2.0) +
        pow(delta_H_prime / S_H, 2.0) +
        R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
    );

    // Return true if color difference is within threshold (e.g., <= 8)
    return (delta_E <= 12.0);
}

__global__ void GetLUVfromYUV(float* __restrict__ lTable, float* __restrict__ uTable, float* __restrict__ vTable, float* __restrict__ _uTable, float* __restrict__ _vTable, LUV* LUVImage, U_V_* UVImage, BYTE* YUVImage) {

    int y = blockIdx.x;
    int x = threadIdx.x * 4;
    int y1 = YUVImage[y * COLUMNS * 2 + x + 0];
    int u = YUVImage[y * COLUMNS * 2 + x + 1];
    int y2 = YUVImage[y * COLUMNS * 2 + x + 2];
    int v = YUVImage[y * COLUMNS * 2 + x + 3];

    x = threadIdx.x * 2;
    LUVImage[y * COLUMNS + x].L = lTable[256 * 256 * y1 + 256 * u + v];
    LUVImage[y * COLUMNS + x].U = uTable[256 * 256 * y1 + 256 * u + v];
    LUVImage[y * COLUMNS + x].V = vTable[256 * 256 * y1 + 256 * u + v];
    UVImage[y * COLUMNS + x].U = _uTable[256 * 256 * y1 + 256 * u + v];
    UVImage[y * COLUMNS + x].V = _vTable[256 * 256 * y1 + 256 * u + v];

    x++;
    LUVImage[y * COLUMNS + x].L = lTable[256 * 256 * y2 + 256 * u + v];
    LUVImage[y * COLUMNS + x].U = uTable[256 * 256 * y2 + 256 * u + v];
    LUVImage[y * COLUMNS + x].V = vTable[256 * 256 * y2 + 256 * u + v];
    UVImage[y * COLUMNS + x].U = _uTable[256 * 256 * y2 + 256 * u + v];
    UVImage[y * COLUMNS + x].V = _vTable[256 * 256 * y2 + 256 * u + v];

}
__global__ void AverageColorKernel(LUV* LUVImage, U_V_* UVImage, LUV* ColorSum1, U_V_* ColorSum2, UINT32* PixelCount, point Min, point Max, bool first) {
    __shared__ int cachedPixels[512];
    __shared__ LUV cachedColors1[512];
    __shared__ U_V_ cachedColors2[512];


    int yIndex = blockIdx.x + Min.y;
    int xIndex = threadIdx.x + Min.x;

    LUV tempSum1 = { 0.0, 0.0, 0.0 };
    U_V_ tempSum2 = { 0.0, 0.0 };

    UINT32 tempCount = 0;
    int index = 0;
    while (yIndex <= Max.y) {
        while (xIndex <= Max.x) {
            int index = yIndex * COLUMNS + xIndex;
           // if ((first) || InRangeOfAverageColor(LUVImage, UVImage, index, false)) {
                LUV TrueColor1 = LUVImage[index];
                U_V_ TrueColor2 = UVImage[index];

                tempSum1.L += TrueColor1.L;
                tempSum1.U += TrueColor1.U;
                tempSum1.V += TrueColor1.V;

                tempSum2.U += TrueColor2.U;
                tempSum2.V += TrueColor2.V;

                tempCount++;
            //}
            xIndex += blockDim.x;
        }
        yIndex += gridDim.x;
    }

    cachedColors1[threadIdx.x] = tempSum1;
    cachedColors2[threadIdx.x] = tempSum2;
    cachedPixels[threadIdx.x] = tempCount;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            cachedPixels[threadIdx.x] += cachedPixels[threadIdx.x + i];
            cachedColors1[threadIdx.x].L += cachedColors1[threadIdx.x + i].L;
            cachedColors1[threadIdx.x].U += cachedColors1[threadIdx.x + i].U;
            cachedColors1[threadIdx.x].V += cachedColors1[threadIdx.x + i].V;

            cachedColors2[threadIdx.x].U += cachedColors2[threadIdx.x + i].U;
            cachedColors2[threadIdx.x].V += cachedColors2[threadIdx.x + i].V;
        }
        __syncthreads();
        i /= 2;
    }
    if (threadIdx.x == 0) {
        ColorSum1[blockIdx.x] = cachedColors1[0];
        ColorSum2[blockIdx.x] = cachedColors2[0];
        PixelCount[blockIdx.x] = cachedPixels[0];
    }
}

__global__ void BinaryImageKernel(LUV* LUVImage, U_V_* UVImage, BYTE* BinaryImage) {
    int i = blockIdx.x;
    for (int j = threadIdx.x; j < COLUMNS; j += blockDim.x) {
        int index = i * COLUMNS + j;
        if (InRangeOfAverageColor(LUVImage, UVImage, index)) BinaryImage[index] = 1;
        else BinaryImage[index] = 0;
    }
}

__global__ void RegionsKernel(INT32* DevHist, BYTE* BinaryImage, point* Arr, point* Norm) {

    __shared__ int InObject[256];
    __shared__ int OutObject[256];

    point Min = Norm[2 * blockIdx.x + 0];
    point Max = Norm[2 * blockIdx.x + 1];
    point regoinPoints[4] = { Arr[4 * blockIdx.x + 0], Arr[4 * blockIdx.x + 1], Arr[4 * blockIdx.x + 2], Arr[4 * blockIdx.x + 3] };
    int HistVal = (blockIdx.x / 4) * 8 + (blockIdx.x % 4) * 2 + 2;
    int xcord = Min.x + threadIdx.x;
    int ycord = Min.y + threadIdx.y;
    int InCount = 0;
    int OutCount = 0;
    while (ycord <= Max.y) {
        while (xcord <= Max.x) {
            float totalArea = Area(regoinPoints[0], regoinPoints[1], regoinPoints[2]) + Area(regoinPoints[0], regoinPoints[3], regoinPoints[2]);
            float tri1Area = Area(point(xcord, ycord), regoinPoints[0], regoinPoints[1]);
            float tri2Area = Area(point(xcord, ycord), regoinPoints[1], regoinPoints[2]);
            float tri3Area = Area(point(xcord, ycord), regoinPoints[2], regoinPoints[3]);
            float tri4Area = Area(point(xcord, ycord), regoinPoints[0], regoinPoints[3]);
            if (abs(totalArea - (tri1Area + tri2Area + tri3Area + tri4Area)) < 0.1) {
                if (BinaryImage[ycord * COLUMNS + xcord] == 1) InCount++;
                else OutCount++;
            }
            xcord += 16;
        }
        xcord = Min.x + threadIdx.x;
        ycord += 16;
    }
    InObject[threadIdx.y * 16 + threadIdx.x] = InCount;
    OutObject[threadIdx.y * 16 + threadIdx.x] = OutCount;
    __syncthreads();

    int i = 128;
    while (i != 0) {
        if (threadIdx.y * 16 + threadIdx.x < i) {
            InObject[threadIdx.y * 16 + threadIdx.x] += InObject[threadIdx.y * 16 + threadIdx.x + i];
            OutObject[threadIdx.y * 16 + threadIdx.x] += OutObject[threadIdx.y * 16 + threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        DevHist[HistVal + 1] = InObject[0];
        DevHist[HistVal] = OutObject[0];
    }

}


void Image::AccGetAverageColor1() {

    int MaxIt = 20;
    cudaMemcpy(DevYUVImage, OriginalImage, sizeof(BYTE) * ROWS * COLUMNS * 2, cudaMemcpyHostToDevice);

    INT32 sz = Mask.get_size();
    point MinPoint = Mask[0];
    point MaxPoint = Mask[0];

    point c = Mask.centroid();
    //for (INT32 i = 0; i < sz; i++) {
        /*MinPoint.x = min(Mask[i].x, MinPoint.x);
        MinPoint.y = min(Mask[i].y, MinPoint.y);
        MaxPoint.x = max(Mask[i].x, MaxPoint.x);
        MaxPoint.y = max(Mask[i].y, MaxPoint.y);*/
        MinPoint.x = max(c.x - 10, 0);
        MinPoint.y = max(c.y - 10, 0);
        MaxPoint.x = min(c.x + 10, 1279);
        MaxPoint.y = min(c.y, 719);
    //}

    GetLUVfromYUV << <ROWS, (COLUMNS / 2) >> > (lTable, uTable, vTable, _uTable, _vTable, DevImageLuv, DevImageUV, DevYUVImage);

    int ActualBlocksUsed = min(32, MaxPoint.x - MinPoint.x);
    int ThreadesPerBlock = 512;
    AverageColorKernel << <ActualBlocksUsed, ThreadesPerBlock >> > (DevImageLuv, DevImageUV, DevColorSum1, DevColorSum2, DevPixelCount, MinPoint, MaxPoint, first);
    if (first) {
        MaxIt = 20;
        first = false;
    }

    cudaMemcpy(ColorSum1, DevColorSum1, sizeof(LUV) * ActualBlocksUsed, cudaMemcpyDeviceToHost);
    cudaMemcpy(ColorSum2, DevColorSum2, sizeof(U_V_) * ActualBlocksUsed, cudaMemcpyDeviceToHost);
    cudaMemcpy(PixelCount, DevPixelCount, sizeof(UINT32) * ActualBlocksUsed, cudaMemcpyDeviceToHost);

    LUV summingColor1 = { 0.0, 0.0, 0.0 };
    LUV summingColor2 = { 0.0, 0.0 };
    UINT32 TotalPixels = 0;

    for (int i = 0; i < ActualBlocksUsed; i++) {
        TotalPixels += PixelCount[i];

        summingColor1.L += ColorSum1[i].L;
        summingColor1.U += ColorSum1[i].U;
        summingColor1.V += ColorSum1[i].V;

        summingColor2.U += ColorSum2[i].U;
        summingColor2.V += ColorSum2[i].V;
    }
    if (TotalPixels != 0)
    {
        AvgColorLuv.L = summingColor1.L / TotalPixels;
        AvgColorLuv.U = summingColor1.U / TotalPixels;
        AvgColorLuv.V = summingColor1.V / TotalPixels;

        AvgColorUV.U = summingColor2.U / TotalPixels;
        AvgColorUV.V = summingColor2.V / TotalPixels;
    }

    cudaMemcpyToSymbol(CudaAvgColorLuv, &AvgColorLuv, sizeof(LUV), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(CudaAvgColorUV, &AvgColorUV, sizeof(U_V_), 0, cudaMemcpyHostToDevice);
    BinaryImageKernel << <ROWS, 512 >> > (DevImageLuv, DevImageUV, DevBinaryImage);

    INT32 state = UNSTABLE;
    int iterations = 0;
    while (state != STABLE && (iterations <= MaxIt || first)) {
        sz = Mask.get_size();

        point center = Mask.centroid();
        int counter = 0;
        for (INT32 i = 0; i < sz; i++) {

            //get the line points
            point p1 = Mask[i], p2 = Mask[(i + 1) % sz];
            point mid = p1.mid_point(p2);
            //getting the regions points
            point regionPoints[6];

            //find the direction
            float PerpUnitVectorX = (p1.y - p2.y) / p1.length(p2);
            float PerpUnitVectorY = -1 * (p1.x - p2.x) / p1.length(p2);

            point checkPoint = point(round(mid.x - PerpUnitVectorX * 2), round(mid.y - PerpUnitVectorY * 2));


            regionPoints[0] = point(round(p1.x - PerpUnitVectorX * RegionSide), round(p1.y - PerpUnitVectorY * RegionSide));
            regionPoints[1] = point(round(mid.x - PerpUnitVectorX * RegionSide), round(mid.y - PerpUnitVectorY * RegionSide));
            regionPoints[2] = point(round(p2.x - PerpUnitVectorX * RegionSide), round(p2.y - PerpUnitVectorY * RegionSide));
            regionPoints[3] = point(round(p1.x + PerpUnitVectorX * RegionSide), round(p1.y + PerpUnitVectorY * RegionSide));
            regionPoints[4] = point(round(mid.x + PerpUnitVectorX * RegionSide), round(mid.y + PerpUnitVectorY * RegionSide));
            regionPoints[5] = point(round(p2.x + PerpUnitVectorX * RegionSide), round(p2.y + PerpUnitVectorY * RegionSide));

            for (int j = 0; j < 6; j++) {
                if (regionPoints[j].y < 0) regionPoints[j].y = 0;
                if (regionPoints[j].x < 0) regionPoints[j].x = 0;
                if (regionPoints[j].y >= ROWS) regionPoints[j].y = ROWS - 1;
                if (regionPoints[j].x >= COLUMNS) regionPoints[j].x = COLUMNS - 1;
            }
            if (((checkPoint.x - center.x) * PerpUnitVectorX + (checkPoint.y - center.y) * PerpUnitVectorY) < 0) {
                int j = 0;
                arr[i * 16 + j++] = p1; arr[i * 16 + j++] = regionPoints[0]; arr[i * 16 + j++] = regionPoints[1]; arr[i * 16 + j++] = mid;
                arr[i * 16 + j++] = mid; arr[i * 16 + j++] = regionPoints[1]; arr[i * 16 + j++] = regionPoints[2]; arr[i * 16 + j++] = p2;
                arr[i * 16 + j++] = mid; arr[i * 16 + j++] = regionPoints[4]; arr[i * 16 + j++] = regionPoints[5]; arr[i * 16 + j++] = p2;
                arr[i * 16 + j++] = p1; arr[i * 16 + j++] = regionPoints[3]; arr[i * 16 + j++] = regionPoints[4]; arr[i * 16 + j++] = mid;
            }
            else {
                int j = 0;
                arr[i * 16 + j++] = p1; arr[i * 16 + j++] = regionPoints[3]; arr[i * 16 + j++] = regionPoints[4]; arr[i * 16 + j++] = mid;
                arr[i * 16 + j++] = mid; arr[i * 16 + j++] = regionPoints[4]; arr[i * 16 + j++] = regionPoints[5]; arr[i * 16 + j++] = p2;
                arr[i * 16 + j++] = mid; arr[i * 16 + j++] = regionPoints[1]; arr[i * 16 + j++] = regionPoints[2]; arr[i * 16 + j++] = p2;
                arr[i * 16 + j++] = p1; arr[i * 16 + j++] = regionPoints[0]; arr[i * 16 + j++] = regionPoints[1]; arr[i * 16 + j++] = mid;
            }

            for (int j = 0; j < 4; j++) {
                point Min = arr[16 * i + j * 4];
                point Max = Min;
                for (int k = 0; k < 4; k++) {
                    Min.x = min(Min.x, arr[16 * i + j * 4 + k].x);
                    Min.y = min(Min.y, arr[16 * i + j * 4 + k].y);
                    Max.x = max(Max.x, arr[16 * i + j * 4 + k].x);
                    Max.y = max(Max.y, arr[16 * i + j * 4 + k].y);
                }
                norm_arr[counter++] = Min;
                norm_arr[counter++] = Max;
            }

        }


        cudaMemcpy(DevArr, arr, sizeof(point) * 16 * sz, cudaMemcpyHostToDevice);
        cudaMemcpy(DevNorm, norm_arr, sizeof(point) * 8 * sz, cudaMemcpyHostToDevice);
        cudaMemset(DevHist, 0, sizeof(INT32) * 256);

        dim3 blockDim(16, 16);
        RegionsKernel << <sz * 4, blockDim >> > (DevHist, DevBinaryImage, DevArr, DevNorm);

        cudaMemcpy(Histogram, DevHist, sizeof(INT32) * 256, cudaMemcpyDeviceToHost);
        /* cout << iterations << endl;
         for (int i = 0; i < sz; i++) {
             cout << Mask[i].x << " " << Mask[i].y << endl;
             cout << "Min Point :" << norm_arr[8 * i + 0 + 0].x << " " << norm_arr[8 * i + 0 + 0].y << endl;
             cout << "Max Point :" << norm_arr[8 * i + 0 + 1].x << " " << norm_arr[8 * i + 0 + 1].y << endl;
             cout << "A: " << Histogram[8 * i + 2] << " " << Histogram[8 * i + 3] << endl;

             cout << "Min Point :" << norm_arr[8 * i + 2 + 0].x << " " << norm_arr[8 * i + 2 + 0].y << endl;
             cout << "Max Point :" << norm_arr[8 * i + 2 + 1].x << " " << norm_arr[8 * i + 2 + 1].y << endl;
             cout << "B: " << Histogram[8 * i + 4] << " " << Histogram[8 * i + 5] << endl;

             cout << "Min Point :" << norm_arr[8 * i + 4 + 0].x << " " << norm_arr[8 * i + 4 + 0].y << endl;
             cout << "Max Point :" << norm_arr[8 * i + 4 + 1].x << " " << norm_arr[8 * i + 4 + 1].y << endl;
             cout << "C: " << Histogram[8 * i + 6] << " " << Histogram[8 * i + 7] << endl;

             cout << "Min Point :" << norm_arr[8 * i + 6 + 0].x << " " << norm_arr[8 * i + 6 + 0].y << endl;
             cout << "Max Point :" << norm_arr[8 * i + 6 + 1].x << " " << norm_arr[8 * i + 6 + 1].y << endl;
             cout << "D: " << Histogram[8 * i + 8] << " " << Histogram[8 * i + 9] << endl;

             cout << endl << endl;
         }
         cout << endl << endl;*/

        state = Mask.get_state(Histogram);
        switch (state) {

        case DEFORM: {
            Mask.deform(Histogram);

        } break;

        case INSERT: {
            Mask.insert_vertecies_at_position(Histogram);
        }  break;

        case REINIT: {
            Mask.reinit(Mask.centroid());
        }break;
                   //after deformation and insertion remove extra vertecies if needed
        default: {
            Mask.remove_extra_vertices();
            std::cout << iterations << endl;
            //iterations = 50;

        }
        }
        iterations++;
    }
}
