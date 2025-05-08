#include "../include/ImageWindow.h"
ImageWindow::ImageWindow() : numOVert(4), im(nullptr), hImage(nullptr)
{
    pa = new point[numOVert];
}
ImageWindow::~ImageWindow()
{
    if (hImage)
    {
        DeleteObject(hImage);
        hImage = NULL;
    }
}

bool ImageWindow::IsStarted()
{
    return StartedDef;
}
int ImageWindow::getBMwidth()
{
    return bm.bmWidth;
}
int ImageWindow::getBMheight()
{
    return bm.bmHeight;
}
int ImageWindow::initiateImage(HBITMAP him)
{
    hImage = him;
    im = new Image;
    /*hImage = (HBITMAP) LoadImageW(NULL,
                                L"test1.bmp",
                                IMAGE_BITMAP,
                                0, 0,
                                LR_LOADFROMFILE);*/
    if (!hImage)
    {
        MessageBox(NULL, L"Image Didn't Load", L"Error", MB_OK);
        return 0;
    }

    GetObject(hImage, sizeof(BITMAP), &bm);
    //im->OriginalImage = GetBitMapColorArray(bm, hImage);

    return 1;
}

void ImageWindow::SetImageModel(Image i)
{
    im = &i;
}

void ImageWindow::SetBitmap()
{

    RECT clientRect;
    GetClientRect(m_hwnd, &clientRect);
    RedrawWindow(m_hwnd, &clientRect, NULL, RDW_INVALIDATE | RDW_UPDATENOW);
}

PCWSTR  ImageWindow::ClassName() const
{
    return L"Window Image Class";
}

LRESULT ImageWindow::HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {

    case WM_DESTROY:
    {
        PostQuitMessage(0);
    }

    return 0;
    case WM_LBUTTONDOWN:
    {
        if (!StartedDef)
        {
            clock_t start = clock();
            StartedDef = true;
            im->Mask.insert(point(GET_X_LPARAM(lParam) - 10, GET_Y_LPARAM(lParam) - 10), 0);
            im->Mask.insert(point(GET_X_LPARAM(lParam) + 10, GET_Y_LPARAM(lParam) - 10), 1);
            im->Mask.insert(point(GET_X_LPARAM(lParam) + 10, GET_Y_LPARAM(lParam) + 10), 2);
            im->Mask.insert(point(GET_X_LPARAM(lParam) - 10, GET_Y_LPARAM(lParam) + 10), 3);
            for (int i = 0; i < 4; i++)
            {
                pa[i].x = im->Mask[i].x;
                pa[i].y = im->Mask[i].y;
            }
            ind++;
            //im->GetAverageColor();
            //im->GetBinaryImage();

            /*for(int i = 0; i < 3; i++)
            {
                im->GetRegions();
                im->Mask.deform(im->Histogram);
                for (int i = 0; i < 4; i++)
                {
                        pa[i].x = im->Mask[i].x;
                        pa[i].y = im->Mask[i].y;
                }
                ind++;
            }*/
            clock_t end = clock();
            double delta = double(end - start) / CLOCKS_PER_SEC;
            std::cout << "Operation took: " << delta << std::endl;
            RECT clientRect;
            GetClientRect(m_hwnd, &clientRect);
            RedrawWindow(m_hwnd, &clientRect, NULL, RDW_INVALIDATE | RDW_UPDATENOW);
        }
    }
    return 0;
    case WM_PAINT:
    {
        int RegionsC = 2;
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(m_hwnd, &ps);
        RECT clientRect;
        GetClientRect(m_hwnd, &clientRect);

        if (hImage)
        {
            HDC hdcMem = CreateCompatibleDC(hdc);
            SelectObject(hdcMem, hImage);
            BITMAP bm;
            GetObject(hImage, sizeof(BITMAP), &bm);
            BitBlt(hdc, 0, 0, bm.bmWidth, bm.bmHeight, hdcMem, 0, 0, SRCCOPY);
            DeleteDC(hdcMem);
        }
        else
        {
            FillRect(hdc, &ps.rcPaint, (HBRUSH)COLOR_WINDOW + 1);
        }

        if (StartedDef)
        {
            DrawVertA(hdc, pa, numOVert, RegionsC, 20);
        }
        EndPaint(m_hwnd, &ps);

    }
    return 0;
    default:
        return DefWindowProc(m_hwnd, uMsg, wParam, lParam);
    }
    return TRUE;
}
void ImageWindow::GetCurrentMask(const model& mod)
{
    if (mod.get_size() > 2)
    {
        if (mod.get_size() != numOVert)
        {
            delete[] pa;
            numOVert = mod.get_size();
            pa = new point[mod.get_size()];
        }
        for (int i = 0; i < numOVert; i++)
        {
            pa[i] = mod[i];
        }
        RECT clientRect;
        GetClientRect(m_hwnd, &clientRect);
        RedrawWindow(m_hwnd, &clientRect, NULL, RDW_INVALIDATE | RDW_UPDATENOW);
    }

}
