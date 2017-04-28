#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <pthread.h>

#ifndef _UTIL_HPP
#define _UTIL_HPP

using namespace cv;


class threadVector {
public:
    std::vector<CvRect> faces_vec;
private:
    pthread_mutex_t lock;
public:
    void combineWith(std::vector<CvRect> v)
    {
        pthread_mutex_lock(&lock);
        faces_vec.insert(faces_vec.end(), v.begin(), v.end());
        pthread_mutex_unlock(&lock);
    }
};




/******************************************************************************
 ************************* Cascade Utility Functions **************************
 ******************************************************************************/

struct cascadeClassifier {
    CvHaarClassifierCascade* cascade;
    int numStages;
};
typedef struct cascadeClassifier* cascadeClassifier_t;

cascadeClassifier_t newCascadeClassifier(const char*);

void freeCascadeClassifier(cascadeClassifier_t);

CvHaarClassifierCascade* loadCVHaarCascade(const char*);

int printStats(CvHaarClassifierCascade*);

void groupRectangles(std::vector<CvRect> &, int, float);



/******************************************************************************
 ************************** Image Utility Functions ***************************
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