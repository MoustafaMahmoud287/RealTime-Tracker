#ifndef IMAGEWINDOW_H
#define IMAGEWINDOW_H
#ifndef UNICODE
#define UNICODE
#endif


#include "HelperFuncs.h"
#include<stdlib.h>
#include<ctime>
#include <d2d1.h>
#include <wincodec.h> 
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "windowscodecs.lib") 


class ImageWindow : public BaseChildWindow<ImageWindow> {
private:
    /******************** CLASS VARIABLES *****************/
    //image class for Generating regions and binary image

    // image handle for the image to be shown
    HBITMAP hImage;
    BITMAP bm;
    //Mouse position Buffer
    point mousePos;
    // Num Of iterations
    int iter = 100;
    //points array for drawing
    point* pa;
    int numOVert;

    int ind = 0;

    ID2D1Factory* pD2DFactory;          // Direct2D factory
    ID2D1HwndRenderTarget* pRenderTarget; // Direct2D render target
    ID2D1Bitmap* pBitmap;               // Direct2D bitmap
    IWICImagingFactory* pWICFactory;
    IWICFormatConverter* m_pConvertedSourceBitmap;
    ID2D1Bitmap* m_pD2DBitmap;


public:
    bool Drawn = false;
    bool StartedDef = false;
    Image* im;
    ImageWindow();

    // Destructor
    ~ImageWindow();
    /****************** Func **********************/
    int initiateImage(HBITMAP him);

    /****************** ACCESSORS *****************/
    bool IsStarted();
    int getBMwidth();
    int getBMheight();
    /****************** Setters *******************/
    void SetStartModel(point* MArr);
    void SetImageModel(Image i);
    void SetBitmap();
    PCWSTR  ClassName() const;
    /******************* Message Handler *********************/
    LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);
    void GetCurrentMask(const model& mod);
    HRESULT InitializeFactories();
    HRESULT CreateGraphicsResources();
    HRESULT LoadBitmapFromHBITMAP(HBITMAP hBitmap);
    void DiscardGraphicsResources();
};


#endif // IMAGEWINDOW_H
