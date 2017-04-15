#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

#ifndef _UTIL_HPP
#define _UTIL_HPP

using namespace cv;


/******************************************************************************
 *********************** Cascade Utility Functions ****************************
 ******************************************************************************/

struct cascadeClassifier {
    CvHaarClassifierCascade* cascade;
    int numStages;
};
typedef struct cascadeClassifier* cascadeClassifier_t;

cascadeClassifier_t newCascadeClassifier(CvHaarClassifierCascade*);

//@TODO: free struct

CvHaarClassifierCascade* loadCVHaarCascade(const char*);

void printStats(CvHaarClassifierCascade*);


/******************************************************************************
 ************************* Image Utility Functions ****************************
 ******************************************************************************/

struct imageData {
    Mat* image;
    int height;
    int width;
    Mat* normInt;
    Mat* intImage;
    Mat* sqImage;
};
typedef struct imageData* imageData_t;

imageData_t newImageData(const char *);

IplImage* loadGrayImage(const char *);

#endif /*_UTIL_HPP */