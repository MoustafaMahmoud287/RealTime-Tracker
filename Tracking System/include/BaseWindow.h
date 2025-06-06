#ifndef BASEWINDOW_H
#define BASEWINDOW_H

#include<windows.h>
#include<windowsx.h>

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

template <class DERIVED_TYPE>
class BaseWindow{
protected:
    virtual PCWSTR  ClassName() const = 0;
    virtual LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam) = 0;

    HWND m_hwnd;

public:
    ~BaseWindow()
    {
        if(m_hwnd != NULL)
        {
            DestroyWindow(m_hwnd);
            m_hwnd = NULL;
        }
    }
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
    {
        DERIVED_TYPE* pThis = NULL;

        if (uMsg == WM_CREATE)
        {
            CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
            pThis = (DERIVED_TYPE*)(pCreate->lpCreateParams);
            SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)pThis);

            pThis->m_hwnd = hwnd;
        }
        else
        {
            pThis = (DERIVED_TYPE*) GetWindowLongPtr(hwnd, GWLP_USERDATA);
        }

        if(pThis)
        {
            return pThis->HandleMessage(uMsg, wParam, lParam);
        }
        else
        {
            return DefWindowProc(hwnd,uMsg,wParam,lParam);
        }

    }
    HWND GetHWND() {return m_hwnd;}
    BaseWindow() : m_hwnd(NULL){}
    BOOL Create (PCWSTR lpWindowName,
        DWORD dwStyle,
        DWORD dwExStyle = 0,
        int x = CW_USEDEFAULT,
        int y = CW_USEDEFAULT,
        int nWidth = CW_USEDEFAULT,
        int nHeight = CW_USEDEFAULT,
        HWND hWndParent = 0,
        HMENU hMenu = 0
        )
        {
            WNDCLASS wc = {0};
            wc.lpfnWndProc = DERIVED_TYPE::WindowProc;
            wc.hInstance = GetModuleHandle(NULL);
            wc.lpszClassName = ClassName();
            RegisterClass(&wc);
            m_hwnd = CreateWindowEx(
                    dwExStyle, ClassName(), lpWindowName, dwStyle, x, y,
                    nWidth , nHeight, hWndParent, hMenu, GetModuleHandle(NULL), this);
            return (m_hwnd ? TRUE : FALSE);
        }
    HWND Window() const
    {
        return m_hwnd;
    }
};

#endif // BASEWINDOW_H
