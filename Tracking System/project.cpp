#ifndef UNICODE
#define UNICODE
#endif

#include "include/MainWindow.h"
#include "include/camera.h"


MainWindow win;
mutex m;
bool newImage = false;

BYTE* Buffer = NULL;

//Global Handles
Camera camera;
HBITMAP HBM;
MSG msg = {};



void FrameUpdate() {
    while (1) {
        camera.CaptureImage(Buffer, m, newImage);
    }
}

void TrackUpdate() {
    while (1)
    {
        while (!newImage);
        m.lock();
        newImage = false;
        BYTE* temp = Buffer;
        win.Imwin->im->OriginalImage = temp;
        m.unlock();
        if (win.Imwin->StartedDef)
        {
            win.Imwin->im->AccGetAverageColor1();
            win.Imwin->Drawn = true;
            win.Imwin->GetCurrentMask(win.Imwin->im->Mask);
        }
    }
}

void ScreenUpdate() {
    while (1) {
        camera.UpdateBitmap();
        win.Imwin->SetBitmap();
    }
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszArgument, int nCmdShow)
{
    Buffer = new BYTE [Imagelegth * 2];

    camera.CaptureImage(Buffer, m, newImage);
    camera.UpdateBitmap();
    HBM = camera.GetHbitmap();
    win.CreateWindows(HBM);
    ShowWindow(win.Window(), nCmdShow);

    thread t4(FrameUpdate);
    t4.detach();

    thread t5(TrackUpdate);
    t5.detach();

    thread t6(ScreenUpdate);
    t6.detach();

    while (GetMessage(&msg, NULL, 0, 0))
    {

        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

