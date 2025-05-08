#include "../include/MainWindow.h"

MainWindow::MainWindow()
{
    Imwin = new ImageWindow();
}
PCWSTR  MainWindow::ClassName() const
{
    return L"Sample Class Name";
}
int MainWindow::CreateWindows(HBITMAP Hbm)
{
    this->Imwin->initiateImage(Hbm);
    if (!this->Create(L"Show BITMAP", WS_OVERLAPPEDWINDOW))
    {
        return 0;
    }

    if(! Imwin->Create(L"ImageWindow1", this->GetHWND(), 10 ,10 , Imwin->getBMwidth(), Imwin->getBMheight()))
    {
        return 0;
    }

    return 1;
}

LRESULT MainWindow::HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch(uMsg)
    {
    case WM_DESTROY:
        {
            PostQuitMessage(0);
        }

        return 0;
    case WM_PAINT:
        {

            PAINTSTRUCT ps ;
            HDC hdc = BeginPaint(m_hwnd, &ps);
            FillRect(hdc, &ps.rcPaint, (HBRUSH) COLOR_WINDOW);

            EndPaint(m_hwnd, &ps);
        }
        return 0;
    default:
        return DefWindowProc(m_hwnd,uMsg,wParam,lParam);
    }
    return TRUE;
}
