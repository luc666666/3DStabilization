// matcher_simple.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cuda_runtime.h>

#include <iu/iucore.h>
#include <iu/iuio.h>
#include <iu/iumath.h>
#include <rof/rof.h>
#include <fl/flowlib.h>

#include <SimpleSIFT.h>
#include "colorcode.h"

//#define HOMO
#define ROCHESTER
#define MYMETHOD 0
#define SCALE_FACTOR 0.3
#define dis_thres 1.2
// #define BU 0
// #define BV 0
#define LU 1404
#define LV 936
using namespace cv;
using namespace std;

double avSubMatValue32F( const CvPoint2D64f* pt, const cv::Mat* mat );
double interpolatePoint( Point2f pt, const cv::Mat mat );

// combine flow and save flow in first flow
void combineFlow(Mat matu1, Mat matv1, Mat matu2, Mat matv2, Mat &resultu, Mat &resultv)
{
	resultu = matu1.clone();
	resultv = matv1.clone();
	for (int y = 0; y < matu1.rows; y++)
	{
		for (int x = 0; x < matu1.cols; x++)
		{			
			double u1 = x + matu1.at<float>(y,x);
			double v1 = y + matv1.at<float>(y,x);
			if ((int)u1 >= 0 && (int)v1 >= 0 && (int)u1 < matu1.cols-1 && (int)v1 < matu1.rows-1 )
			{
				double u2 = interpolatePoint( Point2f(u1, v1), matu2 );
				double v2 = interpolatePoint( Point2f(u1, v1), matv2 );
				resultu.at<float>(y,x) = matu1.at<float>(y,x) + u2;
				resultv.at<float>(y,x) = matv1.at<float>(y,x) + v2;
			}
		}
	}
}
void writeMatToFile(cv::Mat& m, const char* filename)
{
	ofstream fout(filename);

	if(!fout)
	{
		cout<<"File Not Opened"<<endl;  return;
	}

	for(int i=0; i<m.rows; i++)
	{
		for(int j=0; j<m.cols; j++)
		{
			fout<<m.at<float>(i,j)<<"\t";
		}
		fout<<endl;
	}

	fout.close();
}

double sigmoid(double x)
{
	return 1/(1+ exp(-x));
}
void runKmeans(Mat img, Mat &structure_mask)
{
	// 	
	//step 1 : map the img to the samples
	int orgRows = img.rows;
	int orgCols = img.cols;
	Mat colVec = img.reshape(1,img.rows*img.cols);
	Mat colVecD, centers, labels;
	int attempts = 5;
	int clusts = 2;
	double eps = 0.01;
	colVec.convertTo(colVecD, CV_32FC3); // convert to floating point

	//step 2 : apply kmeans to find labels and centers
	double compactness = kmeans(colVecD, clusts, labels, 
		TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, attempts, eps), 
		attempts, KMEANS_PP_CENTERS, centers);

	//step 3 : map the centers to the output
	float temp1 = 0;
	float temp2 = 0;
	Mat new_image(img.size(), img.type());
	for( int row = 0; row != img.rows; ++row){
		auto new_image_begin = new_image.ptr<uchar>(row);
		auto new_image_end = new_image_begin + new_image.cols * 3;
		auto labels_ptr = labels.ptr<int>(row * img.cols);

		while(new_image_begin != new_image_end){
			int const cluster_idx = *labels_ptr;
			auto centers_ptr = centers.ptr<float>(cluster_idx);
			// 			new_image_begin[0] = centers_ptr[0];
			// 			new_image_begin[1] = centers_ptr[1];
			// 			new_image_begin[2] = centers_ptr[2];
			if (cluster_idx == 0)
			{
				temp1 += (centers_ptr[0] + centers_ptr[1] + centers_ptr[2])/3;
			} else {
				temp2 += (centers_ptr[0] + centers_ptr[1] + centers_ptr[2])/3;
			}
			// 			new_image_begin[0] = (cluster_idx^labels.at<int>(0) == 1)?255:0;
			// 			new_image_begin[1] = (cluster_idx^labels.at<int>(0) == 1)?255:0;
			// 			new_image_begin[2] = (cluster_idx^labels.at<int>(0) == 1)?255:0;
			new_image_begin[0] = (cluster_idx == 1)?255:0;
			new_image_begin[1] = (cluster_idx == 1)?255:0;
			new_image_begin[2] = (cluster_idx == 1)?255:0;
			new_image_begin += 3; ++labels_ptr;
		}
	}
	cvtColor(new_image,structure_mask,CV_RGB2GRAY);
	if (temp1 < temp2)
		structure_mask = 255 - structure_mask;
}

static void floodFillPostprocess( Mat& img, Mat &mask, const Scalar& colorDiff=Scalar::all(1) )
{
	CV_Assert( !img.empty() );
	RNG rng = theRNG();
	mask = Mat::zeros( img.rows+2, img.cols+2, CV_8UC1 );
	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			if( mask.at<uchar>(y+1, x+1) == 0 )
			{
				Scalar newVal( rng(256), rng(256), rng(256) );
				floodFill( img, mask, Point(x,y), newVal, 0, colorDiff, colorDiff );				
			}
		}
	}
	threshold(mask, mask, 1, 128, CV_THRESH_BINARY);
	floodFill( img, mask, Point(0,0), 255, 0, colorDiff, colorDiff,  4 + (255 << 8) + CV_FLOODFILL_FIXED_RANGE );
	floodFill( img, mask, Point(img.rows,0), 255, 0, colorDiff, colorDiff,  4 + (255 << 8) + CV_FLOODFILL_FIXED_RANGE );
	mask = 255 - mask;
}

void getTVL1Flow(Mat img1, Mat img2, Mat &matU, Mat &matV, char* filename, Mat &colorFlow);
//void getTVL1Flow(Mat img1, Mat img2, Mat &flowColor);

void refineFlow(Mat &mat_u1, Mat &mat_v1, Mat mat_u2, Mat mat_v2)
{
	int h = mat_u1.rows;
	int w = mat_u1.cols;
	float epsilon = 0.01;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			// perfect case
			if (mat_u1.at<float>(i,j)+mat_u2.at<float>(i,j) < epsilon && mat_v1.at<float>(i,j)+mat_v2.at<float>(i,j) < epsilon)
				continue;
			if (i > 0 && j > 0 && i < h-1 && j < w-1)
			{
				mat_u1.at<float>(i,j) = (mat_u1.at<float>(i+1,j) + mat_u1.at<float>(i,j+1) + mat_u1.at<float>(i-1,j) + mat_u1.at<float>(i,j-1)
					+ mat_u1.at<float>(i+1,j+1) + mat_u1.at<float>(i-1,j+1) + mat_u1.at<float>(i-1,j-1) + mat_u1.at<float>(i+1,j-1))/8;
				mat_v1.at<float>(i,j) = (mat_v1.at<float>(i+1,j) + mat_v1.at<float>(i,j+1) + mat_v1.at<float>(i-1,j) + mat_v1.at<float>(i,j-1)
					+ mat_v1.at<float>(i+1,j+1) + mat_v1.at<float>(i-1,j+1) + mat_v1.at<float>(i-1,j-1) + mat_v1.at<float>(i+1,j-1))/8;
			}
		}
	}
}

void find_line_eq(float x1, float y1, float x2, float y2, float &A, float &B, float &C)
{
	float dX = x2 - x1;
	float dY = y2 - y1;
	A = dY;
	B = -dX;
	C = dX*y1 - dY*x1;
}
void drawEpLine(float a, float b, float c, Mat& img, Scalar sc)
{
	float y1 = -c/b;
	float x = img.cols;
	float y2 = -(a*x+c)/b;
	line(img,Point(0,y1),Point(x,y2),sc,1);
}

void drawLine(float a, float b, float c, Mat& img)
{
	float y1 = -c/b;
	float x = img.cols;	
	float y2 = -(a*x+c)/b;
	line(img,Point(0,y1),Point(x,y2),Scalar::all(0));
}

double interpolatePoint( Point2f pt, const cv::Mat mat )
{
	//cout << pt.x << "\t" << pt.y << endl;
	int floorx = (int)( pt.x );
	int floory = (int)( pt.y );

	// 	cout << floorx << "\t" << floory << endl;
	// 	cout << mat.cols << "\t" << mat.rows << endl;
	if( floorx < 0 || floorx >= mat.cols - 1 || 
		floory < 0 || floory >= mat.rows - 1 )
		return 0;

	double px = pt.x - floorx;
	double py = pt.y - floory;

	double tl = mat.at<float>(floory,   floorx  );
	double tr = mat.at<float>(floory,   floorx+1);
	double bl = mat.at<float>(floory+1, floorx);
	double br = mat.at<float>(floory+1, floorx+1);

	return tl*(1-px)*(1-py) + tr*px*(1-py) + bl*(1-px)*py + br*px*py;
}

double avSubMatValue32F( const CvPoint2D64f* pt, const cv::Mat* mat )
{
	int floorx = (int)floor( pt->x );
	int floory = (int)floor( pt->y );

	if( floorx < 0 || floorx >= (*mat).cols - 1 || 
		floory < 0 || floory >= (*mat).rows - 1 )
		return 0;

	double px = pt->x - floorx;
	double py = pt->y - floory;

	double tl = (*mat).at<float>(floory,   floorx  );
	double tr = (*mat).at<float>(floory,   floorx+1);
	double bl = (*mat).at<float>(floory+1, floorx);
	double br = (*mat).at<float>(floory+1, floorx+1);

	return tl*(1-px)*(1-py) + tr*px*(1-py) + bl*(1-px)*py + br*px*py;
}

void chooseSamples(Mat matu, Mat matv, vector<Point2f> &left, vector<Point2f> &right)
{
	int* ptu = new int[10000];
	int* ptv = new int[10000];
	double* flowu = new double[10000];
	double* flowv = new double[10000];
	left.resize(10000);
	right.resize(10000);

	for(int i = 0; i < 10000; i++)
	{
		ptu[i] = rand()%matu.cols;
		ptv[i] = rand()%matu.rows;
		double val_u=avSubMatValue32F( &cvPoint2D64f((double)ptu[i], (double)ptv[i]), &matu );
		double val_v=avSubMatValue32F( &cvPoint2D64f((double)ptu[i], (double)ptv[i]), &matv );
		while(abs(val_u)<0.001 && abs(val_v)<0.001)
		{
			ptu[i] = rand()%matu.cols;
			ptv[i] = rand()%matu.rows;
			val_u=avSubMatValue32F( &cvPoint2D64f((double)ptu[i], (double)ptv[i]), &matu );
			val_v=avSubMatValue32F( &cvPoint2D64f((double)ptu[i], (double)ptv[i]), &matv );
		}		
	}

	for(int i = 0; i < 10000; i++)
	{
		double fu, fv, dfu, dfv;

		left[i].x = (float)ptu[i]; 
		left[i].y = (float)ptv[i];

		fu = 0; fv = 0;

		for(int j = 0; j < 1; j++)
		{
			dfu = fu + avSubMatValue32F( &cvPoint2D64f((double)ptu[i] + fu, (double)ptv[i] + fv), &matu );
			dfv = fv + avSubMatValue32F( &cvPoint2D64f((double)ptu[i] + fu, (double)ptv[i] + fv), &matv );
			fu = dfu;
			fv = dfv;
		}

		flowu[i] = fu;
		flowv[i] = fv;

		left[i].x = (float)ptu[i]; 
		left[i].y = (float)ptv[i];

		right[i].x = left[i].x + (float)flowu[i];
		right[i].y = left[i].y + (float)flowv[i];
	}
	delete [] ptu;
	delete [] ptv;
	delete [] flowu;
	delete [] flowv;
}

int _tmain(int argc, _TCHAR* argv[])
{
	// 	Mat gray = imread("gray1.jpg",0);
	// 	Mat moving_mask;
	// 	threshold(gray,moving_mask,22,255,THRESH_BINARY);
	// 	dilate(moving_mask,moving_mask,Mat(),Point(-1,-1),5);
	// 	imwrite("g_moving.jpg",moving_mask);
	// 	Mat static_mask = 255 - moving_mask;
	// 	bitwise_and(gray,255-moving_mask,gray);
	// 	threshold(gray,gray,3,255,THRESH_BINARY);
	// 	imwrite("gray2.jpg",gray);
	// 
	// 	RNG rng(12345);
	// 	vector<vector<Point>> contours;
	// 	vector<vector<Point>> contours_structure;
	// 	vector<Vec4i> hierarchy;
	// 	findContours( gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	// 	/// Draw contours
	// 	Mat drawing = Mat::zeros( gray.size(), CV_8UC3 );
	// 	for(int i = 0; i< contours.size(); i++ )
	// 		if (contourArea(contours[i]) > 1200)
	// 			contours_structure.push_back(contours[i]);

	clock_t start, finish;// cost time evaluation	
	double duration;
	HANDLE hFind;
	WIN32_FIND_DATA data;
	char filename[260];
	char fullname[260];
	char DefChar = ' ';

	hFind = FindFirstFile(L"data\\*", &data);
	if (hFind == INVALID_HANDLE_VALUE) {
		cout << "Error reading directory\n";
		return -1;
	}

	// ignore the first 2 files
	do {
		WideCharToMultiByte(CP_ACP,0,data.cFileName,-1,filename,260,&DefChar,NULL);
		if (strcmp("..",filename) == 0)
			break;
	} while (FindNextFile(hFind, &data));

#ifdef SIFTGPU_DLL_RUNTIME
	#ifdef _WIN32
		HMODULE  hsiftgpu = LoadLibrary(L"bin\\SiftGPU64.dll");
	#else
		void * hsiftgpu = dlopen("libsiftgpu.so", RTLD_LAZY);
	#endif

	if(hsiftgpu == NULL) return 0;

#ifdef REMOTE_SIFTGPU
	ComboSiftGPU* (*pCreateRemoteSiftGPU) (int, char*) = NULL;
	pCreateRemoteSiftGPU = (ComboSiftGPU* (*) (int, char*)) GET_MYPROC(hsiftgpu, "CreateRemoteSiftGPU");
	ComboSiftGPU * combo = pCreateRemoteSiftGPU(REMOTE_SERVER_PORT, REMOTE_SERVER);
	SiftGPU* sift = combo;
	SiftMatchGPU* matcher = combo;
#else
	SiftGPU* (*pCreateNewSiftGPU)(int);
	SiftMatchGPU* (*pCreateNewSiftMatchGPU)(int);
	// 	SiftGPU* (*pCreateNewSiftGPU)(int) = NULL;
	// 	SiftMatchGPU* (*pCreateNewSiftMatchGPU)(int) = NULL;
	pCreateNewSiftGPU = (SiftGPU* (*) (int)) GET_MYPROC(hsiftgpu, "CreateNewSiftGPU");
	pCreateNewSiftMatchGPU = (SiftMatchGPU* (*)(int)) GET_MYPROC(hsiftgpu, "CreateNewSiftMatchGPU");
	SiftGPU* sift = pCreateNewSiftGPU(1);
	SiftMatchGPU* matcher = pCreateNewSiftMatchGPU(4096);
#endif

#elif defined(REMOTE_SIFTGPU)
	ComboSiftGPU * combo = CreateRemoteSiftGPU(REMOTE_SERVER_PORT, REMOTE_SERVER);
	SiftGPU* sift = combo;
	SiftMatchGPU* matcher = combo;
#else
	//this will use overloaded new operators
	SiftGPU  *sift = new SiftGPU;
	SiftMatchGPU *matcher = new SiftMatchGPU(4096);
#endif
	// 	char * argvv[] = {"-fo", "-1",  "-v", "1"};//
	// 	//-fo -1    staring from -1 octave 
	// 	//-v 1      only print out # feature and overall time
	// 	//-loweo    add a (.5, .5) offset
	// 	//-tc <num> set a soft limit to number of detected features
	// 	int argcc = sizeof(argvv)/sizeof(char*);
	// 	sift->ParseParam(argcc, argvv);

	if(sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return 0;

	Mat img1;
	Mat img2;
	//Mat prev_img; // previous image
	Mat pref_img; // preference image
	//Mat mat_u_cum_org, mat_v_cum_org;
	Mat moving_car_mask_p;
	//Mat H_cum = Mat::ones(3,3,CV_64FC1); // combine all prev Homography
	//Mat H_cum_p = Mat::ones(3,3,CV_64FC1);
	//Mat H_compensate = Mat::ones(3,3,CV_64FC1);
	//setIdentity(H_cum,1);
	//setIdentity(H_cum_p,1);
	//setIdentity(H_compensate,1);
	int fr = 0;
	Mat training_structure_org;
	Mat trained_structure_org;
	// 	Mat training_structure = Mat::zeros(LV,LU,CV_32F);
	// 	Mat trained_structure = Mat::zeros(LV,LU,CV_8UC1);
	char ff[100];
	if (FindNextFile(hFind, &data))
	{
		WideCharToMultiByte(CP_ACP,0,data.cFileName,-1,filename,260,&DefChar,NULL);
		strcpy(fullname, "data/");
		strcat(fullname,filename);
		img1 = imread(fullname, CV_LOAD_IMAGE_GRAYSCALE);
		if (img1.empty())
		{
			printf("Cannot read the first image");
			return -1;
		}
		// just for testing
		//resize(img1,img1,Size(),SCALE_FACTOR,SCALE_FACTOR);

		sprintf(fullname,"result//%s",filename);
		imwrite(fullname,img1);
// 		mat_u_cum_org = Mat::zeros(img1.size(),CV_32FC1);
// 		mat_v_cum_org = Mat::zeros(img1.size(),CV_32FC1);
		pref_img = img1.clone();
		//pref_img = img1.clone();
		//resize(img1,img1,Size(),SCALE_FACTOR,SCALE_FACTOR);
		training_structure_org = Mat::zeros(img1.rows,img1.cols,CV_32F);
		trained_structure_org = Mat::zeros(img1.rows,img1.cols,CV_8UC1);
	}
	while (FindNextFile(hFind, &data))
	{
		WideCharToMultiByte(CP_ACP,0,data.cFileName,-1,filename,260,&DefChar,NULL);
		strcpy(fullname, "data/");
		strcat(fullname,filename);
		// 		if (strcmp("IMG_2213.JPG",filename) == 0)
		// 		{
		// 			int a = 0;
		// 		}
		img1 = pref_img.clone();
		img2 = imread(fullname, CV_LOAD_IMAGE_GRAYSCALE);
		
		// just for testing
		//resize(img2,img2,Size(),SCALE_FACTOR,SCALE_FACTOR);
		//prev_img = img2.clone();

		if (img2.empty())
		{
			printf("Cannot read one of the images");
			return -1;
		}
		//resize(img2,img2,Size(),SCALE_FACTOR,SCALE_FACTOR);
		fr++;

		// intensity correction
// 		Scalar a = sum(img1);
// 		Scalar b = sum(img2);
// 		float gainFactor = a.val[0] / b.val[0];
// 		img2 = img2*gainFactor;

		Mat confidentMap;
		confidentMap = Mat(img1.rows,img1.cols,CV_8UC1,Scalar(255));
		vector<Point2f> obj;
		vector<Point2f> scene;
		detectSIFT(img1, img2, obj, scene, sift, matcher);

		// find H
		Mat H = findHomography(obj,scene,CV_RANSAC);
		// 		H_cum_p = H_cum.clone();
		//H_cum = H_cum*H;

		// 		// change pref_img
		// 		if (fr++ % 100 == 0)
		// 		{
		// 			cout << fr << fr%3 << endl;
		// 			warpPerspective(img2,pref_img,H_cum.inv(),Size());
		// 			imwrite(filename,pref_img(Rect(0,0,LU,LV)));
		// 			continue;
		// 		}

		Mat img3_org;
		//warpPerspective(img2,img3_org,H_cum.inv(),Size());
		warpPerspective(img2,img3_org,H.inv(),Size());
		sprintf(ff,"%s_homo.jpg",filename);
		imwrite(ff,img3_org);

		Mat img1_org = img1.clone();
		//img1 = pref_img.clone();
		// update accummulate H
		// 		if (fr % 5 == 0)
		// 		{
		// 			cout << "Updating accumulate H." << endl;
		// 			obj.clear();
		// 			scene.clear();
		// 			detectSIFT(img1, img3_org, obj, scene, sift, matcher);
		// 			Mat H = findHomography(obj,scene,CV_RANSAC);
		// 			H_cum = H*H_cum;
		// 			warpPerspective(img2,img3_org,H_cum.inv(),Size());
		// 		}

		Mat mask_org(img1.rows,img1.cols,CV_8UC1, Scalar::all(255));
		//warpPerspective(mask,mask,H_cum.inv(),Size(LU,LV));
		warpPerspective(mask_org,mask_org,H.inv(),Size());

		int noHorizontalSegments = img1.cols / LU;
		int noVerticalSegments = img1.rows / LV;
		//Mat myflow(noVerticalSegments*LV,noHorizontalSegments*LU,CV_8UC3, Scalar::all(0));
		Mat result(noVerticalSegments*LV,noHorizontalSegments*LU,CV_8UC1, Scalar::all(255));
		Mat u_global(noVerticalSegments*LV,noHorizontalSegments*LU,CV_32FC1, Scalar::all(255));
		Mat v_global(noVerticalSegments*LV,noHorizontalSegments*LU,CV_32FC1, Scalar::all(255));
		Mat u_global2(noVerticalSegments*LV,noHorizontalSegments*LU,CV_32FC1, Scalar::all(255));
		Mat v_global2(noVerticalSegments*LV,noHorizontalSegments*LU,CV_32FC1, Scalar::all(255));

		// crop image to compute optical flow using Flowlib
		start=clock();
		for (int vs = 0; vs < noVerticalSegments; vs++)
		{
			for (int hs = 0; hs < noHorizontalSegments; hs++)
			{
				int BU = hs * LU;
				int BV = vs * LV;
				cout << "BU,BV:" << BU << "," << BV << endl;
				Mat mask = mask_org(Rect(BU,BV,LU,LV)).clone();

				// ROI
				// 				Mat mat_u_cum(mat_u_cum_org,Rect(BU,BV,LU,LV)); 
				// 				Mat mat_v_cum(mat_v_cum_org,Rect(BU,BV,LU,LV));
				Mat u3(u_global,Rect(BU,BV,LU,LV));
				Mat v3(v_global,Rect(BU,BV,LU,LV));
				Mat u3_2(u_global2,Rect(BU,BV,LU,LV));
				Mat v3_2(v_global2,Rect(BU,BV,LU,LV));
				Mat cmap(confidentMap,Rect(BU,BV,LU,LV));

				// find flow -> warp back to get correct flow
				img1 = img1_org(Rect(BU,BV,LU,LV)).clone();
				Mat img3 = img3_org(Rect(BU,BV,LU,LV)).clone();

				vector<Point2f> left;
				vector<Point2f> right;
				Mat mat_u, mat_v, mat_u2, mat_v2, colorFlow;

				bitwise_and(img1,mask,img1);

				// intensity correction
				Scalar a = sum(img1);
				Scalar b = sum(img3);
				float gainFactor = a.val[0] / b.val[0];
				img3 = img3*gainFactor;

// 				sprintf(ff,"%s_%d_%d_img1.jpg",filename,vs,hs);
// 				imwrite(ff,img1);
// 				sprintf(ff,"%s_%d_%d_img3.jpg",filename,vs,hs);
// 				imwrite(ff,img3);
				//start=clock();
				getTVL1Flow(img3, img1, mat_u2, mat_v2, filename, colorFlow);
				getTVL1Flow(img1, img3, mat_u, mat_v, filename, colorFlow);

// 				Mat flowColor;
// 				MotionToColor(mat_u,mat_v,flowColor,40);
// 				sprintf(ff,"%s_%d_%d_flow.jpg",filename,vs,hs);
// 				imwrite(ff,flowColor);
// 				Mat ttt(myflow,Rect(BU,BV,LU,LV)); flowColor.copyTo(ttt);
				mat_u.copyTo(u3);
				mat_v.copyTo(v3);
				mat_u2.copyTo(u3_2);
				mat_v2.copyTo(v3_2);

				//refineFlow(mat_u,mat_v,mat_u2,mat_v2);
				// 				sprintf(ff,"%s_%d_%d_flow.jpg",filename,vs,hs);
				// 				imwrite(ff,colorFlow);

				//refineFlow(mat_u,mat_v,mat_u2,mat_v2);
				// 				finish=clock();
				// 				duration=(double)(finish-start)/CLOCKS_PER_SEC;
				// 				cout << "TVL1: " << duration << endl;

				//start=clock();
// 				u3 = Mat::zeros(mat_u.rows,mat_u.cols,CV_32FC1);
// 				v3 = Mat::zeros(mat_u.rows,mat_u.cols,CV_32FC1);
// 
// 				//Point3f bin1(0,0,0), bin2(0,0,0), bin3(0,0,0), bin4(0,0,0);
// 				for (int y = 0; y < mat_u.rows; y++)
// 				{
// 					for (int x = 0; x < mat_u.cols; x++)
// 					{
// 						float i3 = BU + x + mat_u.at<float>(y,x);
// 						float j3 = BV + y + mat_v.at<float>(y,x);
// 
// 						Mat x3_homo = (Mat_<double>(1,3) << i3, j3, 1);
// 
// 						//Mat x2_H = H_cum*x3_homo.t();
// 						Mat x2_H = H*x3_homo.t();
// 						float proj_y = x2_H.at<double>(1,0) / x2_H.at<double>(2,0);
// 						float proj_x = x2_H.at<double>(0,0) / x2_H.at<double>(2,0);
// 						u3.at<float>(y,x) = proj_x - x - BU;
// 						v3.at<float>(y,x) = proj_y - y - BV;
// 					}
// 				}
// 				Mat flowColor;
// 				MotionToColor(u3,v3,flowColor,-1);
// 				sprintf(ff,"%s_%d_%d_flow.jpg",filename,vs,hs);
// 				imwrite(ff,flowColor);
			}
		}
		finish=clock();
		duration=(double)(finish-start)/CLOCKS_PER_SEC;
		cout << "Compute flow: " << duration << endl;

		// find confident area
		start=clock();
		cout << "Find inconsistent flow: ";
		for (int y = 0; y < u_global.rows; y++)
		{
			for (int x = 0; x < u_global.cols; x++)
			{
				float u1 = x + u_global.at<float>(y,x);
				float v1 = y + v_global.at<float>(y,x);
				if ((int)u1 >= 0 && (int)v1 >= 0 && (int)u1 < u_global.cols-1 && (int)v1 < u_global.rows-1 )
				{
					double u2 = interpolatePoint( Point2f(u1, v1), u_global2 );
					double v2 = interpolatePoint( Point2f(u1, v1), v_global2 );
					u1 = u1 - x; v1 = v1 - y;
					if (abs(u1+u2) < 0.1 || abs(v1+v2) < 0.1)
						confidentMap.at<uchar>(y,x) = 0;						
				}		
			}
		}
		sprintf(ff,"%s_inconsistentMap.jpg",filename);
		imwrite(ff,confidentMap);
		finish=clock();
		duration=(double)(finish-start)/CLOCKS_PER_SEC;
		cout << duration << "s" << endl;

 		Mat flowColor;
// 		sprintf(ff,"%s_flow_crop.jpg",filename);
// 		imwrite(ff,myflow);

		start=clock();
		cout << "Interpolate inconsistent flow: ";	 
		// fill hole using linear heat equation		
		int numInterations = 30;
		int avgPrecisionSize = 8; // smaller is better, but takes longer
		Mat kernel = Mat::ones( avgPrecisionSize, avgPrecisionSize, CV_32F )/ (avgPrecisionSize*avgPrecisionSize);
		Mat tempU, tempV;

		GaussianBlur( u_global, tempU, Size( 99, 99 ), 50, 50 );
		GaussianBlur( v_global, tempV, Size( 99, 99 ), 50, 50 );

		bitwise_not(confidentMap,confidentMap);
		u_global.copyTo(tempU,confidentMap);
		v_global.copyTo(tempV,confidentMap);

		for (int i = 0; i < numInterations; i++)
		{
			filter2D(tempU, tempU, -1 , kernel, Point( -1, -1 ), 0, BORDER_DEFAULT );
			filter2D(tempV, tempV, -1 , kernel, Point( -1, -1 ), 0, BORDER_DEFAULT );
			u_global.copyTo(tempU,confidentMap);
			v_global.copyTo(tempV,confidentMap);
		}		
		u_global = tempU.clone();
		v_global = tempV.clone();
		finish=clock();
		duration=(double)(finish-start)/CLOCKS_PER_SEC;
		cout << duration << "s" << endl;

		MotionToColor(u_global,v_global,flowColor,40);
		sprintf(ff,"%s_flow.jpg",filename);
		imwrite(ff,flowColor);

		start=clock();
		cout << "Warp back the flow: ";
		for (int y = 0; y < u_global.rows; y++)
		{
			for (int x = 0; x < u_global.cols; x++)
			{
				float i3 = x + u_global.at<float>(y,x);
				float j3 = y + v_global.at<float>(y,x);
				Mat x3_homo = (Mat_<double>(3,1) << i3, j3, 1);

				Mat x2_H = H*x3_homo;
				float proj_y = x2_H.at<double>(1,0) / x2_H.at<double>(2,0);
				float proj_x = x2_H.at<double>(0,0) / x2_H.at<double>(2,0);
				u_global.at<float>(y,x) = proj_x - x;
				v_global.at<float>(y,x) = proj_y - y;					
			}
		}
		finish=clock();
		duration=(double)(finish-start)/CLOCKS_PER_SEC;
		cout << duration << "s" << endl;

		// find F of consecutive frames
		start=clock();
		vector<Point2f> left;
		vector<Point2f> right;
		chooseSamples(u_global,v_global,left,right);
		vector<uchar> status;
		Mat F = findFundamentalMat(left,right,FM_LMEDS,1,0.999,status);
		//Mat F = findFundamentalMat(left,right,FM_RANSAC);	
		finish=clock();
		duration=(double)(finish-start)/CLOCKS_PER_SEC;
		cout << "Find F for consecutive frames: " << duration << endl;

		// refine flow using epipolar constraints
		Mat moving_cars_mask = Mat::zeros(result.rows,result.cols,CV_8UC1);
		start=clock();
		cout << "Refine flow for consecutive frames: ";
		for (int y = 0; y < u_global.rows; y++)
		{
			for (int x = 0; x < u_global.cols; x++)
			{
				float x2_flow = x + u_global.at<float>(y,x);
				float y2_flow = y + v_global.at<float>(y,x);
				float proj_x = x2_flow;
				float proj_y = y2_flow;

				// find the epipolar line in 2nd image using F
				Mat x1_homo = (Mat_<double>(1,3) << x, y, 1);
				Mat line = x1_homo*F.t();
				//Mat line = x1_homo*F;
				float a = line.at<double>(0,0);
				float b = line.at<double>(0,1);
				float c = line.at<double>(0,2);

				float d = abs(a*x2_flow + b*y2_flow + c) / sqrt(a*a + b*b);
				if (d > dis_thres && (abs(u_global.at<float>(y,x))>0.001 || abs(v_global.at<float>(y,x))>0.001))
					moving_cars_mask.at<uchar>(y,x) = 255;
// 				else
// 				{
					// project to epipolar line
					proj_x = (b*b*x2_flow - a*b*y2_flow - c*a) / (a*a + b*b);
					//proj_y = -a*proj_x/b - c/b;
					proj_y = (a*a*y2_flow - a*b*x2_flow - b*c) / (a*a + b*b);

					u_global.at<float>(y,x) = proj_x - x;
					v_global.at<float>(y,x) = proj_y - y;
				//}				
			}
		}
		finish=clock();
		duration=(double)(finish-start)/CLOCKS_PER_SEC;
		cout << duration << "s" << endl;
		MotionToColor(u_global,v_global,flowColor,40);
		sprintf(ff,"%s_flow_corrected.jpg",filename);
		imwrite(ff,flowColor);

		start=clock();
		left.clear();
		right.clear();
		chooseSamples(u_global,v_global,left,right);
		Mat H_cum = findHomography(left,right,CV_RANSAC);
		Mat warpedHomo;
		warpPerspective(img2,warpedHomo,H_cum.inv(),Size());

		// refine moving cars mask
		vector<vector<Point> > contours;
		vector<vector<Point> > contours2;
		vector<Vec4i> hierarchy;
		findContours( moving_cars_mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		/// Find the rotated rectangles for each contour
		for( int i = 0; i < contours.size(); i++ )
		{ 
			RotatedRect minRect = minAreaRect( Mat(contours[i]) );
			float w = minRect.size.width;
			float h = minRect.size.height;
			if ((abs(w - h) > 0.5*w || abs(w - h) > 0.5*h) && contourArea(contours[i]) < 2000)
			//if (contourArea(contours[i]) < 2000)
				contours2.push_back(contours[i]);
		}
		moving_cars_mask.setTo(Scalar::all(0));
		for(int i = 0; i< contours2.size(); i++ )
			drawContours( moving_cars_mask, contours2, i, Scalar::all(255), CV_FILLED, 8, hierarchy, 0, Point() );
		sprintf(ff,"%s_moving.jpg",filename);
		imwrite(ff,moving_cars_mask);

		// warping
		start=clock();
		
		for (int y = 0; y < result.rows; y++)
		{
			for (int x = 0; x < result.cols; x++)
			{
				if (mask_org.at<uchar>(y,x) == 0)
					result.at<uchar>(y,x) = 0;
// 				else if (moving_cars_mask.at<uchar>(y,x) != 0)
// 				{
// 					result.at<uchar>(y,x) = warpedHomo.at<uchar>(y,x);
// 				}
				else
				{
					float x2_flow = x + u_global.at<float>(y,x);
					float y2_flow = y + v_global.at<float>(y,x);
					float proj_x = x2_flow;
					float proj_y = y2_flow;

					if (proj_x < 0 || proj_y < 0 || proj_x >= img1_org.cols || proj_y >= img1_org.rows)
					{
						// boundary
						result.at<uchar>(y,x) = 0;
						continue;
					}
					result.at<uchar>(y,x) = img2.at<uchar>(proj_y,proj_x);
				}				
			}
		}
		finish=clock();
		duration=(double)(finish-start)/CLOCKS_PER_SEC;
		cout << "Write result: " << duration << endl;
		sprintf(ff,"result//%s.jpg",filename);
		imwrite(ff,result);
	}
	FindClose(hFind);
#ifdef REMOTE_SIFTGPU
	delete combo;
#else
	delete sift;
	delete matcher;
#endif

#ifdef SIFTGPU_DLL_RUNTIME
	FREE_MYLIB(hsiftgpu);
#endif
	return 0;
}

void getTVL1Flow(Mat img1, Mat img2, Mat &matU, Mat &matV, char* filename, Mat &colorFlow)
	//void getTVL1Flow(Mat img1, Mat img2, Mat &flowColor)
{
	// convert to 32-bit float images
	IuSize sz(img1.cols, img2.rows);
	iu::ImageCpu_32f_C1 *im_cpu1 = new iu::ImageCpu_32f_C1(sz);
	iu::ImageCpu_32f_C1 *im_cpu2 = new iu::ImageCpu_32f_C1(sz);
	Mat img1_mat(sz.height,sz.width,CV_32FC1,im_cpu1->data(),im_cpu1->pitch());
	Mat img2_mat(sz.height,sz.width,CV_32FC1,im_cpu2->data(),im_cpu2->pitch());
	img1.convertTo(img1_mat,img1_mat.type(),1.0f/255.0f,0);
	img2.convertTo(img2_mat,img2_mat.type(),1.0f/255.0f,0);	

	// copy CPU images to GPU images
	iu::ImageGpu_32f_C1 *im_gpu1 = new iu::ImageGpu_32f_C1(sz);
	iu::ImageGpu_32f_C1 *im_gpu2 = new iu::ImageGpu_32f_C1(sz);
	iu::copy(im_cpu1,im_gpu1);
	iu::copy(im_cpu2,im_gpu2);

	// parametrization
	fl::FlowLib flow(0);
	//flow.parameters().model = fl::HL1;
	flow.parameters().model = fl::HL1_COMPENSATION;
	//flow.parameters().model = fl::FAST_HL1_TENSOR; //fl::HL1;
	//flow.parameters().model = fl::FAST_HL1;
	//flow.parameters().model = fl::HL1_TENSOR_COMPENSATION_PRECOND;
	flow.parameters().verbose = 0;
	flow.parameters().iters = 50;	//50
	flow.parameters().warps = 4;	// 4
	flow.parameters().scale_factor = 0.95f;	//0.5f
	flow.parameters().interpolation_method = IU_INTERPOLATE_LINEAR;
	flow.parameters().lambda = 40.0f; //25f	//35f
	flow.parameters().gamma_c = 0.01f;
	flow.parameters().epsilon_u = 0.01f;
	flow.parameters().epsilon_c = 0.01f;
	flow.parameters().str_tex_decomp_method = fl::STR_TEX_DECOMP_OFF;

	// 	flow.parameters().model = fl::FAST_HL1_TENSOR;
	// 	flow.parameters().iters = 200; // Rochester
	// 	flow.parameters().warps = 1; // Rochester
	// 	flow.parameters().scale_factor = (float)0.5f;// Rochester
	// 	flow.parameters().lambda = (float)20; // Rochester
	// 	flow.parameters().gamma_c = 0.01f;

	flow.setInputImages(im_gpu1,im_gpu2);
	if (!flow.calculate())
	{
		cout << "Error getting the optical flow\n";
		return;
	}

	// results and display
	IuSize result_sz = flow.getSize(flow.parameters().stop_level);
	// 	iu::ImageGpu_8u_C4 cu_flow(result_sz);
	// 	iu::ImageCpu_8u_C4 cpu_flow(result_sz);
	// 	//flow.getColorFlow_8u_C4(flow.parameters().stop_level,&cu_flow,5.0f);
	// 	flow.getColorFlow_8u_C4(flow.parameters().stop_level,&cu_flow);
	// 	iu::copy(&cu_flow,&cpu_flow);
	// 	Mat outFLow(result_sz.height,result_sz.width,CV_8UC4,cpu_flow.data(),cpu_flow.pitch());
	// 	cvtColor(outFLow,colorFlow,CV_RGBA2RGB);
	// 	//cvtColor(outFLow,grayFlow,CV_RGBA2GRAY);
	// 	colorFlow = outFLow.clone();
	// 
	// 	Mat bgr(result_sz.height,result_sz.width,CV_8UC3);
	// 	char ff[100]; strcpy(ff,filename);
	// 	strcat(ff, "_flow.jpg");
	// 	//iu::imsave(&cu_flow,ff);
	// 	iu::imsave(&cu_flow,"flow2.jpg");

	iu::ImageGpu_32f_C1 cu_matU(result_sz);
	iu::ImageGpu_32f_C1 cu_matV(result_sz);
	iu::ImageCpu_32f_C1 cpu_matU(result_sz);
	iu::ImageCpu_32f_C1 cpu_matV(result_sz);
	flow.getU_32f_C1(flow.parameters().stop_level, &cu_matU);
	flow.getV_32f_C1(flow.parameters().stop_level, &cu_matV);
	iu::copy(&cu_matU,&cpu_matU);
	iu::copy(&cu_matV,&cpu_matV);

	Mat outU(result_sz.height,result_sz.width,CV_32FC1,cpu_matU.data(),cpu_matU.pitch());
	Mat outV(result_sz.height,result_sz.width,CV_32FC1,cpu_matV.data(),cpu_matV.pitch());
	matU = outU.clone();
	matV = outV.clone();

	delete im_gpu1;
	delete im_gpu2;
	delete im_cpu1;
	delete im_cpu2;
}

