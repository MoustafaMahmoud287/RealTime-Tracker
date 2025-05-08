#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#ifndef UNICODE
#define UNICODE
#endif
#include "ImageWindow.h"
#include "BaseWindow.h"
#include<iostream>

class MainWindow : public BaseWindow<MainWindow>
{
public :
    ImageWindow* Imwin;
    MainWindow();
    PCWSTR  ClassName() const;
    LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);
    int CreateWindows(HBITMAP Hbm);

};
#endif // MAINWINDOW_H
