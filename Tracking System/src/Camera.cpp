#include "../include/Camera.h"

#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")


/// <summary>
/// Helper function to clamp values between 0 and 255
/// </summary>
/// <param name="value"></param>
/// <returns></returns>
inline BYTE Clamp(int value) {
    return (BYTE)(value < 0 ? 0 : (value > 255 ? 255 : value));
}

void ThreadHandler::UpdateImageDisplay(BYTE*& ArrayRGB, BYTE*& pData) {
    size_t index = { 0 }, counter = { 0 };
    for (int y = 0; y < ImageHeight; y++) {
        index = y * (ImageWidth * 2);
        for (int x = 0; x < ImageWidth; x += 2) {

            int y1 = pData[index];
            int u = pData[index + 1];
            int y2 = pData[index + 2];
            int v = pData[index + 3];

            // Convert first pixel from YUV to RGB
            int c1 = y1 - 16;
            int d = u - 128;
            int e = v - 128;

            ArrayRGB[counter++] = Clamp((298 * c1 + 516 * d + 128) >> 8);                //BLUE
            ArrayRGB[counter++] = Clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8);      //GREEN
            ArrayRGB[counter++] = Clamp((298 * c1 + 409 * e + 128) >> 8);                //RED

            if (x + 1 < ImageWidth) {
                c1 = y2 - 16;

                ArrayRGB[counter++] = Clamp((298 * c1 + 516 * d + 128) >> 8);                //BLUE
                ArrayRGB[counter++] = Clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8);      //GREEN
                ArrayRGB[counter++] = Clamp((298 * c1 + 409 * e + 128) >> 8);                //RED
            }
            index += 4;
        }
    }
}

void ThreadHandler::ReadData(BYTE*& pData, BYTE*& image, BYTE*& imageCopy, size_t start, size_t end) {
    while (start < end) {
        image[start] = imageCopy[start] = pData[start];
        start++;
    }
}




/// <summary>
/// To Read Data from files 
/// </summary>
/// <param name="num">flag to name of files</param>
void ThreadHandler::ReadFile(char FileName) {
   /* fstream file;
    switch (FileName) {
    case dataU:
        file.open("dataU.txt", ios::app | ios::in | ios::out | ios::binary);
        file.close();
        file.open("dataU.txt", ios::in | ios::out | ios::binary);
        file.seekg(0, ios::beg);
        file.read((char*)DataU, (size_t)MaxLength * sizeof(float));
        break;

    case dataV:
        file.open("dataV.txt", ios::app | ios::in | ios::out | ios::binary);
        file.close();
        file.open("dataV.txt", ios::in | ios::out | ios::binary);
        file.seekg(0, ios::beg);
        file.read((char*)DataV, (size_t)MaxLength * sizeof(float));
        break;
    default:
        MessageBoxW(NULL, L"Failed in Data name", L"Camera Error", MB_OK | MB_ICONERROR);
    }*/
}


/// <summary>
/// Constructor and initialize Camera
/// </summary>
Camera::Camera()
    : pCamera_Obj(NULL), pReader(NULL), cbImage(0), hBitmap(NULL), 
    ArrayRGB(nullptr) , pData(nullptr) {

    HDC screenDC = GetDC(NULL);
    HDC memDC = CreateCompatibleDC(screenDC);

    //initialize camera configuration
    this->InitializeCamera();

    //initialize image array 1D of hsi "buffer"
    image = new BYTE [Imagelegth * 2];
    imageCopy = new BYTE [Imagelegth * 2];
    BufferImage = new BYTE[Imagelegth * 2];
}


/// <summary>
/// Destructor
/// </summary>
Camera::~Camera() {
    if (pCamera_Obj) pCamera_Obj->Release();
    if (pReader) pReader->Release();
    DeleteDC(memDC);
    ReleaseDC(NULL, screenDC);
    if (hBitmap) DeleteObject(hBitmap);
    MFShutdown();
    CoUninitialize();
    delete[] image;
    delete[] imageCopy;
    delete[] BufferImage;
}


/// <summary>
/// Initialize Camera , opean port to transfer data
/// </summary>
/// <returns>HRESULT to succeeded funcation or faild</returns>
HRESULT Camera::InitializeCamera() {

    // initialize COM with thread "COM work parallel"
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (SUCCEEDED(hr)) {
        MFStartup(MF_VERSION);

        // create object for camera connected with usb with name 
        hr = CreateCameraObject(&pCamera_Obj);
        if (SUCCEEDED(hr)) {

            // create object "Reader" that read data from buffer "pReader"
            hr = MFCreateSourceReaderFromMediaSource(pCamera_Obj, NULL, &pReader);
            if (SUCCEEDED(hr)) {
                hr = ConfigureMediaType();
                if (SUCCEEDED(hr))
                {
                    Sleep(1000);
                }
                else {
                    MessageBoxW(NULL, L"Failed to configure media type", L"Camera Error", MB_OK | MB_ICONERROR);
                }
            }
            else {
                MessageBoxW(NULL, L"Failed to create source reader", L"Camera Error", MB_OK | MB_ICONERROR);
            }
        }
        else {
            MessageBoxW(NULL, L"Failed to create camera object", L"Camera Error", MB_OK | MB_ICONERROR);
        }
    }
    else {
        MessageBoxW(NULL, L"Failed to initialize COM", L"Camera Error", MB_OK | MB_ICONERROR);
    }

    BITMAPINFO bmi = { 0 };
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = ImageWidth;
    bmi.bmiHeader.biHeight = -ImageHeight;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 24;
    bmi.bmiHeader.biCompression = BI_RGB;

    hBitmap = CreateDIBSection(
        memDC,              // DC
        &bmi,               // Bitmap info
        DIB_RGB_COLORS,     // Color usage
        (void**)&ArrayRGB,  // Pointer to bit values
        NULL,               // File mapping object
        0                   // Offset to bitmap bits
    );

   /* thread ReadU(ThreadHandler::ReadFile, dataU);
    thread ReadV(ThreadHandler::ReadFile, dataV);
    ReadU.join();
    ReadV.join();*/
    return hr;
}


/// <summary>
/// Create object for camera 
/// </summary>
/// <param name="ppCamera"></param>
/// <returns>HRESULT to succeeded funcation or faild</returns>
HRESULT Camera::CreateCameraObject(IMFMediaSource** ppCamera) {
    //Defines features of object
    IMFAttributes* pAttributes = NULL;
    // array of pointer to all devices connected have the same Attributes 
    IMFActivate** ppDevices = NULL;
    // count of devices connected 
    UINT32 count = 0;
    //make Attributes with one features "type of device that search about it"
    HRESULT hr = MFCreateAttributes(&pAttributes, 1);
    if (SUCCEEDED(hr)) {
        // set GUID of Attributes as video
        hr = pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
        if (SUCCEEDED(hr)) {
            // get all devices that have this Attributes "major type video"
            hr = MFEnumDeviceSources(pAttributes, &ppDevices, &count);

            // search about specific camera "with name"  
            if (SUCCEEDED(hr) && (count != 0))
            {
                bool found = false;
                for (UINT32 i = 0; i < count; i++) {
                    WCHAR* deviceName = NULL;
                    UINT32 nameLength = 0;
                    //get length of name device number i 
                    hr = ppDevices[i]->GetStringLength(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &nameLength);
                    if (FAILED(hr)) continue;
                    // make string with size of name ddevice length 
                    deviceName = new WCHAR[nameLength + 1];
                    //get device name number i
                    hr = ppDevices[i]->GetString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                        deviceName, nameLength + 1, &nameLength);
                    if (SUCCEEDED(hr)) {
                        if (wcscmp(deviceName, CameraName) == 0) {
                            // if this is the camera that i needed then active it and breake loop
                            hr = ppDevices[i]->ActivateObject(IID_PPV_ARGS(ppCamera));
                            found = true;
                            delete[] deviceName;
                            break;
                        }
                    }
                    delete[] deviceName;
                }
                if (!found) {
                    MessageBoxW(NULL, L"Specified camera not found", L"Camera Error", MB_OK | MB_ICONERROR);
                }
            }
            else {
                MessageBoxW(NULL, L"No camera devices found", L"Camera Error", MB_OK | MB_ICONERROR);
            }
        }
        else {
            MessageBoxW(NULL, L"Failed to set device source attribute", L"Camera Error", MB_OK | MB_ICONERROR);
        }
    }
    else {
        MessageBoxW(NULL, L"Failed to create attributes", L"Camera Error", MB_OK | MB_ICONERROR);
    }

    //free memory
    if (ppDevices) {
        for (UINT32 i = 0; i < count; i++) {
            if (ppDevices[i]) ppDevices[i]->Release();
        }
        CoTaskMemFree(ppDevices);
    }
    if (pAttributes) pAttributes->Release();
    return hr;
}


/// <summary>
/// Configures the media type for capturing
/// set features of camera frame , resolution and image size
/// </summary>
/// <returns>HRESULT to succeeded funcation or faild</returns>
HRESULT Camera::ConfigureMediaType() {
    IMFMediaType* pMediaType = NULL;
    HRESULT hr = MFCreateMediaType(&pMediaType);
    if (SUCCEEDED(hr)) {
        hr = pMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        if (SUCCEEDED(hr)) {
            hr = pMediaType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_YUY2);
            if (SUCCEEDED(hr)) {
                hr = pMediaType->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
                if (SUCCEEDED(hr))
                {
                    hr = MFSetAttributeSize(pMediaType, MF_MT_FRAME_SIZE, ImageWidth, ImageHeight);
                    if (SUCCEEDED(hr)) {
                        // Calculate the image size in bytes
                        hr = MFCalculateImageSize(MFVideoFormat_YUY2, ImageWidth, ImageHeight, &cbImage);
                        if (SUCCEEDED(hr)) {
                            hr = pMediaType->SetUINT32(MF_MT_SAMPLE_SIZE, cbImage);
                            if (SUCCEEDED(hr)) {
                                hr = pMediaType->SetUINT32(MF_MT_FIXED_SIZE_SAMPLES, TRUE);
                                if (SUCCEEDED(hr)) {
                                    pMediaType->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE);
                                }
                                else {
                                    MessageBoxW(NULL, L"Failed to sample size fixed", L"Camera Error", MB_OK | MB_ICONERROR);
                                }
                            }
                            else {
                                MessageBoxW(NULL, L"Failed to set sample size", L"Camera Error", MB_OK | MB_ICONERROR);
                            }
                        }
                        else {
                            MessageBoxW(NULL, L"Failed to calculate image size", L"Camera Error", MB_OK | MB_ICONERROR);
                        }
                        hr = MFSetAttributeRatio(pMediaType, MF_MT_FRAME_RATE, CameraFrame, 1);
                        if (SUCCEEDED(hr)) {
                            hr = pReader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, NULL, pMediaType);
                            if (FAILED(hr))
                                MessageBoxW(NULL, L"Failed to set media type for reader", L"Camera Error", MB_OK | MB_ICONERROR);
                        }
                        else {
                            MessageBoxW(NULL, L"Failed to set frame rate", L"Camera Error", MB_OK | MB_ICONERROR);
                        }
                    }
                    else
                    {
                        MessageBoxW(NULL, L"Failed to set frame size", L"Camera Error", MB_OK | MB_ICONERROR);
                    }
                }
                else {
                    MessageBoxW(NULL, L"Failed to set interlace mode", L"Camera Error", MB_OK | MB_ICONERROR);
                }
            }
            else {
                MessageBoxW(NULL, L"Failed to set video format", L"Camera Error", MB_OK | MB_ICONERROR);
            }
        }
        else {
            MessageBoxW(NULL, L"Failed to set major type", L"Camera Error", MB_OK | MB_ICONERROR);
        }
    }
    else {
        MessageBoxW(NULL, L"pMediaType", L"Camera Error", MB_OK | MB_ICONERROR);
    }
    if (pMediaType) pMediaType->Release();
    return hr;
}

/// <summary>
/// return HBITMAP for image
/// </summary>
/// <returns>HBITMAP to image capture</returns>
HBITMAP Camera::GetHbitmap() {
    return hBitmap;
}


/// <summary>
/// Capture image from camera and convert image to
/// 1 - Bitmap in RGB
/// 2 - HSI as value 
/// </summary>
/// <param name="image"></param>
/// <returns>HBITMAP to image in RGB</returns>
void Camera::CaptureImage(BYTE*& imagebuffer , mutex& m , bool& newImage) {
    this->ReadBuffer();
    m.lock();
    swap(imagebuffer, image);
    newImage = true;
    m.unlock();
}



void Camera::ReadBuffer() {
    HRESULT hr = S_OK;
    if (pReader) {
        DWORD  flags;
        IMFSample* pSample = NULL;
        IMFMediaBuffer* pBuffer = NULL;
        DWORD actualLength = 0;

        // Try to read sample multiple times
        for (unsigned char attempts = 0; attempts < 10; attempts++) {
            hr = pReader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, NULL, &flags, NULL, &pSample);
            if (SUCCEEDED(hr) && pSample) {
                break;
            }
        }

        if (SUCCEEDED(hr) && (pSample)) {
            hr = pSample->ConvertToContiguousBuffer(&pBuffer);
            if (SUCCEEDED(hr)) {
                pBuffer->Lock(&pData, NULL, &actualLength);
                if (actualLength >= cbImage)
                {
                    memcpy(image, pData, actualLength);
                    memcpy(BufferImage, pData, actualLength);
                }
                else
                {
                    MessageBoxW(NULL, L"Insufficient buffer size", L"Camera Error", MB_OK | MB_ICONERROR);
                }
            }
            if (pBuffer) pBuffer->Unlock();
            if (pBuffer) pBuffer->Release();
            if (pSample) pSample->Release();
        }
        else {
            MessageBoxW(NULL, L"Fail to ReadSample", L"Camera Error", MB_OK | MB_ICONERROR);
        }
    }
    else {
        MessageBoxW(NULL, L"PReader is NULL", L"Camera Error", MB_OK | MB_ICONERROR);
    }
}



void Camera::UpdateBitmap() {
    memcpy(imageCopy, BufferImage, cbImage);
    size_t index = { 0 }, counter = { 0 };
    for (int y = 0; y < ImageHeight; y++) {
        index = y * (ImageWidth * 2);
        for (int x = 0; x < ImageWidth; x += 2) {

            int y1 = imageCopy[index];
            int u = imageCopy[index + 1];
            int y2 = imageCopy[index + 2];
            int v = imageCopy[index + 3];

            // Convert first pixel from YUV to RGB
            int c1 = y1 - 16;
            int d = u - 128;
            int e = v - 128;

            ArrayRGB[counter++] = Clamp((298 * c1 + 516 * d + 128) >> 8);                //BLUE
            ArrayRGB[counter++] = Clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8);      //GREEN
            ArrayRGB[counter++] = Clamp((298 * c1 + 409 * e + 128) >> 8);                //RED

            if (x + 1 < ImageWidth) {
                c1 = y2 - 16;

                ArrayRGB[counter++] = Clamp((298 * c1 + 516 * d + 128) >> 8);                //BLUE
                ArrayRGB[counter++] = Clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8);      //GREEN
                ArrayRGB[counter++] = Clamp((298 * c1 + 409 * e + 128) >> 8);                //RED
            }
            index += 4;
        }
    }
}