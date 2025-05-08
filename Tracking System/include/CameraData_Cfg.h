//#ifndef CAMERA_DATA_CFG_HEADER
//#define CAMERA_DATA_CFG_HEADER
//
//#include <iostream>
//#include <windows.h>
//#include <mfapi.h>
//#include <mfidl.h>
//#include <mfreadwrite.h>
//#include <winuser.h>
//#include <cmath>
//#include <algorithm>
//#include <mutex>
//#include <functional>
//#include <chrono>
//#include <fstream>
//
//using namespace std;
//using namespace std::chrono;
//
//#define ImageWidth      1280
//#define ImageHeight     720
//
//#define CameraFrame     60
//
//#define CameraName      L"AnkerWork C310 Webcam"
//
//
//
//#define dataH           0
//#define dataS           1
//#define dataI           2
//
//#define MaxLength       16777216            //256 * 256 * 256
//
//extern volatile INT16 DataHue[256][256][256];
//extern volatile BYTE  DataSat[256][256][256];
//extern volatile BYTE  DataInt[256][256][256];
//
//
//struct HSI {
//    INT16 H;
//    BYTE  S;
//    BYTE  I;
//
//    HSI() {
//        H = 0;
//        I = 0;
//        S = 0;
//    }
//
//    void operator = (HSI hsi) {
//        H = hsi.H;
//        S = hsi.S;
//        I = hsi.I;
//    }
//
//};
//
//void ThreadDataRead(char num);
//void UpdateImageDisplay(BYTE*& ArrayRGB, BYTE*& pData);
//void UpdateImageHSI(HSI**& ImageHSI, BYTE*& pData);
//inline BYTE Clamp(int value);
//
//#endif // !CAMERA_DATA_CFG_HEADER