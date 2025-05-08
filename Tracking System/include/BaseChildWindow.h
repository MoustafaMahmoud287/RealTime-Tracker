#ifndef BASECHILDWINDOW_H
#define BASECHILDWINDOW_H

#include<windows.h>
#include<windowsx.h>


LRESULT CALLBACK ChildWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

template <class DERIVED_TYPE>
class BaseChildWindow{
protected:
    virtual PCWSTR  ClassName() const = 0;
    virtual LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam) = 0;

    HWND m_hwnd;

public:
    ~BaseChildWindow()
    {
        if(m_hwnd != NULL)
        {
            DestroyWindow(m_hwnd);
            m_hwnd = NULL;
        }
    }
    static LRESULT CALLBACK ChildWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
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
    BaseChildWindow() : m_hwnd(NULL){}
    BOOL Create (PCWSTR lpWindowName,
                 HWND hWndParent,
                 int x = CW_USEDEFAULT,
                 int y = CW_USEDEFAULT,
                 int nWidth = CW_USEDEFAULT,
                 int nHeight = CW_USEDEFAULT,
                 DWORD dwExStyle = 0,
                 DWORD dwStyle = WS_CHILDWINDOW | WS_VISIBLE,
                 HMENU hMenu = 0)
        {
            WNDCLASS wc = {0};
            if (!GetClassInfo(GetModuleHandle(NULL), ClassName(), &wc))
            {
                wc.lpfnWndProc = DERIVED_TYPE::ChildWindowProc;
                wc.hInstance = GetModuleHandle(NULL);
                wc.lpszClassName = ClassName();
                RegisterClass(&wc);
            }
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
