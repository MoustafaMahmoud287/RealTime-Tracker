#ifndef CAMERA_HEADER
#define CAMERA_HEADER

#include <iostream>
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <winuser.h>
#include <mutex>
#include <fstream>
#include <chrono>
#include <thread>

using namespace std::chrono;
using namespace std;



#define ImageWidth      1280UL
#define ImageHeight     720
#define Imagelegth      ImageWidth * ImageHeight

#define CameraFrame     60

#define CameraName      L"AnkerWork C310 Webcam"



class Camera {
private:
    IMFMediaSource* pCamera_Obj; //Object for camera
    IMFSourceReader* pReader;    //Object to read data form buffer
    UINT cbImage;                //Size of image in buffer 
    HBITMAP hBitmap;             //Image in bitmap
    BYTE* ArrayRGB;              //Image array in RGB for bitmap
    HDC screenDC;                //Handle to Device Context for display
    HDC memDC;                   //Handle to Device Context Memory Device
    BYTE* pData;                 //Array of data that gets from buffer
    BYTE* image;
    BYTE* imageCopy;
    BYTE* BufferImage;

    HRESULT InitializeCamera();
    HRESULT CreateCameraObject(IMFMediaSource** ppCamera);
    HRESULT ConfigureMediaType();

public:
    Camera();
    ~Camera();
    void CaptureImage(BYTE*& imagebuffer , mutex& m , bool& newImage);
    void ReadBuffer();
    void UpdateBitmap();
    HBITMAP GetHbitmap();
};

class ThreadHandler {
    public:
    static void ReadFile(char FileName);
    static void UpdateImageDisplay(BYTE*& ArrayRGB, BYTE*& pData);
    //static void UpdateImageHSI(HSI*& ImageHSI, BYTE*& pData);
    static void ReadData(BYTE*& pData, BYTE*& image, BYTE*& imageCopy, size_t start, size_t end);

};

inline BYTE Clamp(int value);

#endif 