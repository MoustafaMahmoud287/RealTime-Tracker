#include "../include/HelperFuncs.h"



void DrawVertA(HDC _hdc,point* pArr, int l, int &C, int _w)
{
                C = 2;
                int w = _w;
                int radius = 5;

    //Drawing 1 line and dividing the regions
                //1st the 2 verticeies

                for(int i = 0 ; i < l ; i++)
                {

                    point Vertix1(pArr[i]);
                    point Vertix2 = i + 1 >= l   ? point(pArr[0]) : point(pArr[i+1]);
                    Ellipse(_hdc, Vertix1.x - radius, Vertix1.y - radius , Vertix1.x + radius , Vertix1.y + radius );
                    Ellipse(_hdc, Vertix2.x - radius, Vertix2.y - radius , Vertix2.x + radius , Vertix2.y + radius );
                    MoveToEx(_hdc, Vertix1.x, Vertix1.y, NULL);
                    LineTo(_hdc, Vertix2.x, Vertix2.y);

                    // drawing the regions around the edge
                }


}

Color** GetBitMapColorArray(BITMAP bm, HBITMAP hImage) {
    // Allocate 2D array for the colors
    Color** ImToShow = new Color*[bm.bmHeight];
    for (int i = 0; i < bm.bmHeight; i++) {
        ImToShow[i] = new Color[bm.bmWidth];
    }

    // Allocate buffer for pixel data
    BYTE* pixelData = new BYTE[bm.bmHeight * bm.bmWidth * 4];

    // Create DCs
    HDC hdc = GetDC(NULL);
    HDC memDC = CreateCompatibleDC(hdc);

    // Configure BITMAPINFO
    BITMAPINFO bmi = {0};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = bm.bmWidth;
    bmi.bmiHeader.biHeight = -bm.bmHeight; // Top-down DIB
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    // Select bitmap into memory DC
    SelectObject(memDC, hImage);

    // Retrieve bitmap data
    if (GetDIBits(memDC, hImage, 0, bm.bmHeight, pixelData, &bmi, DIB_RGB_COLORS)) {
        for (int y = 0; y < bm.bmHeight; y++) {
            for (int x = 0; x < bm.bmWidth; x++) {
                int index = (y * bm.bmWidth + x) * 4;
                ImToShow[y][x] = Color(pixelData[index], pixelData[index + 1], pixelData[index + 2]); // RGB
            }
        }
    }

    // Clean up
    delete[] pixelData;
    DeleteDC(memDC);
    ReleaseDC(NULL, hdc);

    return ImToShow;
}

/*void DrawVertAReg(HDC _hdc,point* pArr, int l, int &C, int _w)
{
                C = 2;
                int w = _w;
                int radius = 5;

    //Drawing 1 line and dividing the regions
                //1st the 2 verticeies

                for(int i = 0 ; i < l ; i++)
                {

                    point Vertix1(pArr[i]);
                    point Vertix2 = i + 1 >= l   ? point(pArr[0]) : point(pArr[i+1]);
                    Ellipse(_hdc, Vertix1.x - radius, Vertix1.y - radius , Vertix1.x + radius , Vertix1.y + radius );
                    Ellipse(_hdc, Vertix2.x - radius, Vertix2.y - radius , Vertix2.x + radius , Vertix2.y + radius );
                    MoveToEx(_hdc, Vertix1.x, Vertix1.y, NULL);
                    LineTo(_hdc, Vertix2.x, Vertix2.y);

                    // drawing the regions around the edge
                    float Mx, My;
                    if(Vertix2.x < Vertix1.x)
                        Mx = Vertix1.x-Vertix2.x;
                    else
                        Mx = Vertix2.x-Vertix1.x;
                    if(Vertix2.y < Vertix1.y)
                        My = Vertix1.y-Vertix2.y;
                    else
                        My = Vertix2.y-Vertix1.y;
                    point MLine(Mx,My);
                    point NLineVec1 = MLine.Normal1unit() * w;
                    point NLineVec2 = MLine.Normal2unit() * w;
                    point Npoint1 = NLineVec1 + Vertix1;
                    point Npoint2 = NLineVec2 + Vertix1;
                    Ellipse(_hdc, Npoint1.x - radius, Npoint1.y - radius , Npoint1.x + radius , Npoint1.y + radius );
                    Ellipse(_hdc, Npoint2.x - radius, Npoint2.y - radius , Npoint2.x + radius , Npoint2.y + radius );
                    MoveToEx(_hdc, Vertix1.x, Vertix1.y, NULL);
                    LineTo(_hdc, Npoint1.x, Npoint1.y);
                    LineTo(_hdc, Npoint2.x, Npoint2.y);
                    point Npoint3 = NLineVec1 + Vertix2;
                    point Npoint4 = NLineVec2 + Vertix2;
                    Ellipse(_hdc, Npoint3.x - radius, Npoint3.y - radius , Npoint3.x + radius , Npoint3.y + radius );
                    Ellipse(_hdc, Npoint4.x - radius, Npoint4.y - radius , Npoint4.x + radius , Npoint4.y + radius );
                    MoveToEx(_hdc, Vertix2.x, Vertix2.y, NULL);
                    LineTo(_hdc, Npoint3.x, Npoint3.y);
                    LineTo(_hdc, Npoint4.x, Npoint4.y);
                    MoveToEx(_hdc, Npoint1.x, Npoint1.y, NULL);
                    LineTo(_hdc, Npoint3.x, Npoint3.y);
                    MoveToEx(_hdc, Npoint2.x, Npoint2.y, NULL);
                    LineTo(_hdc, Npoint4.x, Npoint4.y);

                    point Nline1Mid = (Npoint1.MidPoint(Npoint3)) ;
                    point Nline2Mid = (Npoint2.MidPoint(Npoint4)) ;
                    Ellipse(_hdc, Nline1Mid.x - radius, Nline1Mid.y - radius , Nline1Mid.x + radius , Nline1Mid.y + radius );
                    Ellipse(_hdc, Nline2Mid.x - radius, Nline2Mid.y - radius , Nline2Mid.x + radius , Nline2Mid.y + radius );
                    MoveToEx(_hdc, Nline1Mid.x, Nline1Mid.y, NULL);
                    LineTo(_hdc, Nline2Mid.x, Nline2Mid.y);


                    point RN1 = Vertix1.MidPoint(Nline1Mid);
                    point RN2 = Vertix2.MidPoint(Nline1Mid);
                    point RN3 = Vertix2.MidPoint(Nline2Mid);
                    point RN4 = Vertix1.MidPoint(Nline2Mid);
                    wchar_t Buffer[20];
                    wsprintf(Buffer, L"%d", C);
                    TextOut(_hdc, RN1.x , RN1.y ,Buffer, lstrlen(Buffer));
                    C += 2;
                    wsprintf(Buffer, L"%d", C);
                    TextOut(_hdc, RN2.x , RN2.y ,Buffer, lstrlen(Buffer));
                    C += 2;
                    wsprintf(Buffer, L"%d", C);
                    TextOut(_hdc, RN3.x , RN3.y ,Buffer, lstrlen(Buffer));
                    C += 2;
                    wsprintf(Buffer, L"%d", C);
                    TextOut(_hdc, RN4.x , RN4.y ,Buffer, lstrlen(Buffer));
                }


}*/


