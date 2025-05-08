//#include "../include/CameraData_Cfg.h"
//
//
//volatile INT16 DataHue[256][256][256];
//volatile BYTE  DataSat[256][256][256];
//volatile BYTE  DataInt[256][256][256];
//
//
///// <summary>
///// To Read Data from files 
///// </summary>
///// <param name="num">flag to name of files</param>
//void ThreadDataRead(char num) {
//    fstream file;
//    switch (num) {
//    case dataH:
//        file.open("dataH.txt", ios::app | ios::in | ios::out | ios::binary);
//        file.close();
//        file.open("dataH.txt", ios::in | ios::out | ios::binary);
//        file.seekg(0, ios::beg);
//        file.read((char*)DataHue, (size_t)MaxLength * sizeof(INT16));
//        break;
//
//    case dataS:
//        file.open("dataS.txt", ios::app | ios::in | ios::out | ios::binary);
//        file.close();
//        file.open("dataS.txt", ios::in | ios::out | ios::binary);
//        file.seekg(0, ios::beg);
//        file.read((char*)DataSat, (size_t)MaxLength * sizeof(BYTE));
//        break;
//
//    case dataI:
//        file.open("dataI.txt", ios::app | ios::in | ios::out | ios::binary);
//        file.close();
//        file.open("dataI.txt", ios::in | ios::out | ios::binary);
//        file.seekg(0, ios::beg);
//        file.read((char*)DataInt, (size_t)MaxLength * sizeof(BYTE));
//        break;
//
//    default:
//        MessageBoxW(NULL, L"Failed in Data name", L"Camera Error", MB_OK | MB_ICONERROR);
//    }
//}
//
//
///// <summary>
///// Helper function to clamp values between 0 and 255
///// </summary>
///// <param name="value"></param>
///// <returns></returns>
//inline BYTE Clamp(int value) {
//    return (BYTE)(value < 0 ? 0 : (value > 255 ? 255 : value));
//}
//
//void UpdateImageDisplay(BYTE*& ArrayRGB , BYTE*& pData) {
//    size_t index = { 0 }, counter = { 0 };
//    for (int y = 0; y < ImageHeight; y++) {
//        index = y * (ImageWidth * 2);
//        for (int x = 0; x < ImageWidth; x += 2) {
//
//            int y1 = pData[index];
//            int u = pData[index + 1];
//            int y2 = pData[index + 2];
//            int v = pData[index + 3];
//
//            // Convert first pixel from YUV to RGB
//            int c1 = y1 - 16;
//            int d = u - 128;
//            int e = v - 128;
//
//            ArrayRGB[counter++] = Clamp((298 * c1 + 516 * d + 128) >> 8);                //BLUE
//            ArrayRGB[counter++] = Clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8);      //GREEN
//            ArrayRGB[counter++] = Clamp((298 * c1 + 409 * e + 128) >> 8);                //RED
//
//            if (x + 1 < ImageWidth) {
//                c1 = y2 - 16;
//
//                ArrayRGB[counter++] = Clamp((298 * c1 + 516 * d + 128) >> 8);                //BLUE
//                ArrayRGB[counter++] = Clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8);      //GREEN
//                ArrayRGB[counter++] = Clamp((298 * c1 + 409 * e + 128) >> 8);                //RED
//            }
//            index += 4;
//
//        }
//    }
//}
//
//void UpdateImageHSI(HSI**& ImageHSI, BYTE*& pData) {
//    size_t index = { 0 };
//    for (int y = 0; y < ImageHeight; y++) {
//        index = y * (ImageWidth * 2);
//        for (int x = 0; x < ImageWidth; x += 2) {
//            ImageHSI[y][x].H = DataHue[pData[index]][pData[index + 1]][pData[index + 3]];
//            ImageHSI[y][x].S = DataSat[pData[index]][pData[index + 1]][pData[index + 3]];
//            ImageHSI[y][x].I = DataInt[pData[index]][pData[index + 1]][pData[index + 3]];
//            if (x + 1 < ImageWidth) {
//                ImageHSI[y][x + 1].S = DataSat[pData[index + 2]][pData[index + 1]][pData[index + 3]];
//                ImageHSI[y][x + 1].I = DataInt[pData[index + 2]][pData[index + 1]][pData[index + 3]];
//                ImageHSI[y][x + 1].H = DataHue[pData[index + 2]][pData[index + 1]][pData[index + 3]];
//            }
//            index += 4;
//        }
//    }
//}
