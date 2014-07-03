#include "stdafx.h"
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <gl/GL.h>
using std::vector;
using std::iostream;
using namespace cv;

////////////////////////////////////////////////////////////////////////////

#if !defined(SIFTGPU_STATIC) && !defined(SIFTGPU_DLL_RUNTIME) 
// SIFTGPU_STATIC comes from compiler
//#define SIFTGPU_DLL_RUNTIME
// Load at runtime if the above macro defined
// comment the macro above to use static linkingx`
#endif

#ifdef _WIN32
	#ifdef SIFTGPU_DLL_RUNTIME
		#define WIN32_LEAN_AND_MEAN
		#include <windows.h>
		#define FREE_MYLIB FreeLibrary
		#define GET_MYPROC GetProcAddress
	#else
//define this to get dll import definition for win32
		#define SIFTGPU_DLL
		#ifdef _DEBUG 
			#pragma comment(lib, "lib/siftgpu_d.lib")
		#else
			#pragma comment(lib, "lib/siftgpu.lib")
		#endif
	#endif
#else
	#ifdef SIFTGPU_DLL_RUNTIME
		#include <dlfcn.h>
		#define FREE_MYLIB dlclose
		#define GET_MYPROC dlsym
	#endif
#endif

#include <../SiftGPU/SiftGPU.h>

int detectSIFT(Mat img1, Mat img2, vector<Point2f> &obj, vector<Point2f> &scene,SiftGPU* sift, SiftMatchGPU* matcher)
{
    vector<float > descriptors1(1), descriptors2(1);
    vector<SiftGPU::SiftKeypoint> keys1(1), keys2(1);    
    int num1 = 0, num2 = 0;
    
	if(sift->RunSIFT(img1.cols,img1.rows,img1.data,GL_LUMINANCE,GL_UNSIGNED_BYTE));
    {
        //Call SaveSIFT to save result to file, the format is the same as Lowe's
        //sift->SaveSIFT("../data/800-1.sift"); //Note that saving ASCII format is slow

        //get feature count
        num1 = sift->GetFeatureNum();

        //allocate memory
        keys1.resize(num1);    descriptors1.resize(128*num1);

        //reading back feature vectors is faster than writing files
        //if you dont need keys or descriptors, just put NULLs here
        sift->GetFeatureVector(&keys1[0], &descriptors1[0]);
        //this can be used to write your own sift file.            
    }

	if(sift->RunSIFT(img2.cols,img2.rows,img2.data,GL_LUMINANCE,GL_UNSIGNED_BYTE));
    {
        num2 = sift->GetFeatureNum();
        keys2.resize(num2);    descriptors2.resize(128*num2);
        sift->GetFeatureVector(&keys2[0], &descriptors2[0]);
    }

    matcher->VerifyContextGL(); //must call once
    matcher->SetDescriptors(0, num1, &descriptors1[0]); //image 1
    matcher->SetDescriptors(1, num2, &descriptors2[0]); //image 2

    //match and get result.    
    int (*match_buf)[2] = new int[num1][2];
    //use the default thresholds. Check the declaration in SiftGPU.h
    int num_match = matcher->GetSiftMatch(num1, match_buf);
    std::cout << num_match << " sift matches were found;\n";
    
    //enumerate all the feature matches
	obj.clear();
	scene.clear();
	vector<DMatch> matches;
    for(int i  = 0; i < num_match; ++i)
    {
        //How to get the feature matches: 
        SiftGPU::SiftKeypoint & key1 = keys1[match_buf[i][0]];
        SiftGPU::SiftKeypoint & key2 = keys2[match_buf[i][1]];
        //key1 in the first image matches with key2 in the second image
		obj.push_back(Point2f(key1.x,key1.y));
		scene.push_back(Point2f(key2.x,key2.y));
		DMatch temp;
		temp.trainIdx = i;
		temp.queryIdx = i;
		matches.push_back(temp);
	}
	vector<KeyPoint> newKey1;
	vector<KeyPoint> newKey2;
	KeyPoint::convert(obj,newKey1);
	KeyPoint::convert(scene,newKey2);
	newKey1.resize(obj.size());
	newKey2.resize(obj.size());
	Mat img_matches;
 	drawMatches(img1,newKey1, img2, newKey2, matches, img_matches,
 		Scalar::all(-1),Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
 	imwrite("matches1.jpg", img_matches);

    // clean up..
    delete[] match_buf;


#ifdef SIFTGPU_DLL_RUNTIME
    FREE_MYLIB(hsiftgpu);
#endif
}