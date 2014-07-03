#include <stdlib.h>
#include <math.h>
#include "imageLib/imageLib.h"

typedef unsigned char uchar;

// colorcode.cpp
//
// Color encoding of flow vectors
// adapted from the color circle idea described at
//   http://members.shaw.ca/quadibloc/other/colint.htm
//
// Daniel Scharstein, 4/2007
// added tick marks and out-of-range coding 6/05/07

#include <opencv2/highgui/highgui.hpp>

int ncols = 0;
#define MAXCOLS 60
#define M_PI   3.1415926535897932384626433832795
int colorwheel[MAXCOLS][3];

using namespace cv;

void setcols(int r, int g, int b, int k)
{
	colorwheel[k][0] = r;
	colorwheel[k][1] = g;
	colorwheel[k][2] = b;
}

void makecolorwheel()
{
	// relative lengths of color transitions:
	// these are chosen based on perceptual similarity
	// (e.g. one can distinguish more shades between red and yellow 
	//  than between yellow and green)
	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;
	ncols = RY + YG + GC + CB + BM + MR;
	//printf("ncols = %d\n", ncols);
	if (ncols > MAXCOLS)
		exit(1);
	int i;
	int k = 0;
	for (i = 0; i < RY; i++) setcols(255,	   255*i/RY,	 0,	       k++);
	for (i = 0; i < YG; i++) setcols(255-255*i/YG, 255,		 0,	       k++);
	for (i = 0; i < GC; i++) setcols(0,		   255,		 255*i/GC,     k++);
	for (i = 0; i < CB; i++) setcols(0,		   255-255*i/CB, 255,	       k++);
	for (i = 0; i < BM; i++) setcols(255*i/BM,	   0,		 255,	       k++);
	for (i = 0; i < MR; i++) setcols(255,	   0,		 255-255*i/MR, k++);
}

void computeColor(float fx, float fy, uchar *pix)
{
	if (ncols == 0)
		makecolorwheel();

	float rad = sqrt(fx * fx + fy * fy);
	float a = atan2(-fy, -fx) / M_PI;
	float fk = (a + 1.0) / 2.0 * (ncols-1);
	int k0 = (int)fk;
	int k1 = (k0 + 1) % ncols;
	float f = fk - k0;
	//f = 0; // uncomment to see original color wheel
	for (int b = 0; b < 3; b++) {
		float col0 = colorwheel[k0][b] / 255.0;
		float col1 = colorwheel[k1][b] / 255.0;
		float col = (1 - f) * col0 + f * col1;
		if (rad <= 1)
			col = 1 - rad * (1 - col); // increase saturation with radius
		else
			col *= .75; // out of range
		pix[2 - b] = (int)(255.0 * col);
	}
}

//void MotionToColor(CFloatImage motim, CByteImage &colim, float maxmotion)
void MotionToColor(Mat matu, Mat matv, Mat &colim, float maxmotion)
{
	//CShape sh = motim.Shape();
	//int width = sh.width, height = sh.height;
	int width = matu.cols, height = matu.rows;
	colim = Mat::zeros(matu.size(),CV_8UC3);
	uchar* pixelPtr = (uchar*)colim.data;
	int cn = colim.channels();
	int x, y;
	// determine motion range:
	float maxx = -999, maxy = -999;
	float minx =  999, miny =  999;
	float maxrad = -1;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			// 			float fx = motim.Pixel(x, y, 0);
			// 			float fy = motim.Pixel(x, y, 1);
			float fx = matu.at<float>(y,x);
			float fy = matv.at<float>(y,x);
			maxx = __max(maxx, fx);
			maxy = __max(maxy, fy);
			minx = __min(minx, fx);
			miny = __min(miny, fy);
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = __max(maxrad, rad);
		}
	}
// 	printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n",
// 		maxrad, minx, maxx, miny, maxy);


	if (maxmotion > 0) // i.e., specified on commandline
		maxrad = maxmotion;

	if (maxrad == 0) // if flow == 0 everywhere
		maxrad = 1;

	// 	if (verbose)
	// 		fprintf(stderr, "normalizing by %g\n", maxrad);

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			// 			float fx = motim.Pixel(x, y, 0);
			// 			float fy = motim.Pixel(x, y, 1);
			float fx = matu.at<float>(y,x);
			float fy = matv.at<float>(y,x);
			uchar pix[3];
			computeColor(fx/maxrad, fy/maxrad, pix);
			pixelPtr[y*colim.cols*cn + x*cn + 0] = pix[0];
			pixelPtr[y*colim.cols*cn + x*cn + 1] = pix[1];
			pixelPtr[y*colim.cols*cn + x*cn + 2] = pix[2];
		}
	}
}