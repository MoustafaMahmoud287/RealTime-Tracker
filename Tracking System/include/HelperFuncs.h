#ifndef HELPERFUNCS_H
#define HELPERFUNCS_H
#ifndef UNICODE
#define UNICODE
#endif
#include "BaseChildWindow.h"
#include "ImageEdit.h"
void DrawVertA(HDC _hdc,point* MArr, int l, int &C, int _w);
//void DrawVertA(HDC _hdc,pointm* pArr, int l, int &C, int _w);
Color** GetBitMapColorArray(BITMAP bm , HBITMAP hImage);
#endif // HELPERFUNCS_H
