#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <time.h>

#include "util.hpp"
#include "cpuDetection.hpp"
#include "cpuThreadDetection.hpp"
#include "cudaDetection.cuh"

#define NTHREADS 16

using namespace cv;
using namespace std;

// Threshold for grouping rectangles in post-processing stage on CPU
static const float GROUP_EPS = 0.4f;

void displayResult(IplImage* im, std::vector<CvRect> faces, const char * image_path)
{
    groupRectangles(faces, 5, GROUP_EPS);

    for (size_t i = 0; i < faces.size(); i++)
    {
        CvRect face = faces[i];
        CvPoint topLeft = cvPoint(face.x, face.y);
        CvPoint botRight = cvPoint(face.x + face.width, face.y + face.height);
        //@TODO: update this ish to the newer version
        cvRectangle(im, topLeft, botRight, CV_RGB(255, 255, 255), 3);
    }

    cvShowImage(image_path, im);
    cvWaitKey(0);
}

int main()
{
    const char * cascade_path = "data/haarcascade_frontalface_alt.xml";
    //const char * other_cascade_path = "data/haarcascade_frontalface_default.xml";
    const char * image_path = "images/soccer.jpg";
    const char * thread_image_path = "images/soccer.jpg";
    std::vector<CvRect> faces;
    threadVector threadFaces;
    struct timespec initStart, initEnd, cpuStart, cpuEnd, threadStart, threadEnd;
    double elapsed;



    clock_gettime(CLOCK_MONOTONIC, &initStart);

    IplImage* im = loadGrayImage(image_path);
    IplImage* im_thread = loadGrayImage(thread_image_path);


    imageData_t i = newImageData(image_path);
    imageData_t i_thread = newImageData(thread_image_path);

    cascadeClassifier_t c = newCascadeClassifier(cascade_path);


    clock_gettime(CLOCK_MONOTONIC, &initEnd);


    elapsed = (initEnd.tv_sec - initStart.tv_sec);
    elapsed += (initEnd.tv_nsec - initStart.tv_nsec) / 1000000000.0;
    cout << elapsed << endl;  
    

    /******************
     * Run detections *
     ******************/

    // CPU
    clock_gettime(CLOCK_MONOTONIC, &cpuStart);
    faces = runCPUdetect(c, i);
    clock_gettime(CLOCK_MONOTONIC, &cpuEnd);
    //cout << faces.size() << endl;
    elapsed = (cpuEnd.tv_sec - cpuStart.tv_sec);
    elapsed += (cpuEnd.tv_nsec - cpuStart.tv_nsec) / 1000000000.0;
    cout << elapsed << endl;

    //displayResult(im, faces, image_path);


    // Threads
    clock_gettime(CLOCK_MONOTONIC, &threadStart);
    threadFaces = runThreadDetect(c, i_thread, NTHREADS);
    clock_gettime(CLOCK_MONOTONIC, &threadEnd);
    //cout << threadFaces.faces_vec.size() << endl;
    elapsed = (threadEnd.tv_sec - threadStart.tv_sec);
    elapsed += (threadEnd.tv_nsec - threadStart.tv_nsec) / 1000000000.0;
    cout << elapsed << endl;

    //displayResult(im_thread, threadFaces.faces_vec, thread_image_path);

    printf("%p, %p\n",c, i );
    runCudaDetection(c, i);


    freeCascadeClassifier(c);
    cvDestroyAllWindows();
    cvReleaseImage(&im);
    cvReleaseImage(&im_thread);

    return 0;
}