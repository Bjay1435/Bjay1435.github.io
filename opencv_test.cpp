
#include <stdio.h>
#include <iostream>
#include <time.h>

#include "util.hpp"
#include "cpuDetection.hpp"
#include "cpuThreadDetection.hpp"
#include "cudaDetection.cuh"
#include "opencvCuda.cuh"

#define NTHREADS 16
#define RUNS 5

using namespace cv;
using namespace std;

// Threshold for grouping rectangles in post-processing stage on CPU
static const float GROUP_EPS = 0.4f;

void displayResult(imageData_t imData, std::vector<CvRect> faces, const char * image_path)
{
    groupRectangles(faces, 5, GROUP_EPS);

    for (size_t i = 0; i < faces.size(); i++)
    {
        CvRect face = faces[i];
        CvPoint topLeft = Point(face.x, face.y);
        CvPoint botRight = Point(face.x + face.width, face.y + face.height);
        //@TODO: update this ish to the newer version
        rectangle(*(imData->image), topLeft, botRight, CV_RGB(255, 255, 255), 3);
    }

    //cvShowImage(image_path, im);
    imshow(image_path, *(imData->image));
    waitKey(0);
}



int main()
{
    const char * cascade_path = "data/haarcascade_frontalface_alt.xml";
    //const char * other_cascade_path = "data/haarcascade_frontalface_default.xml";
    const char * image_path = "images/lena_256.jpg";
    const char * thread_image_path = "images/lena_256.jpg";
    //const char * image_path = "images/lena_256.jpg";
    //const char * thread_image_path = "images/lena_256.jpg";
    std::vector<CvRect> faces, gpuFaces;
    threadVector threadFaces;
    struct timespec initStart, initEnd, cpuStart, cpuEnd, threadStart, threadEnd, gpuStart, gpuEnd;
    double elapsed;

    printf("starting\n");

    clock_gettime(CLOCK_MONOTONIC, &initStart);

    IplImage* im = loadGrayImage(image_path);
    IplImage* im_thread = loadGrayImage(thread_image_path);

    printf("here\n");

    imageData_t i = newImageData(image_path);
    imageData_t i_thread = newImageData(thread_image_path);
    imageData_t i_gpu = newImageData(image_path);

    cudaImageData* gpuImageData = newCudaImageData(image_path);

    cascadeClassifier_t c = newCascadeClassifier(cascade_path);

    printf("here too\n");

    GPUHaarClassifierCascade* devCascade;
    GPUHaarStageClassifier* devStageClassifier;
    GPUHaarClassifier* devClassifier;
    GPUHaarFeature* devFeature;
    allocateGPUCascade(c, 
                   &devCascade, 
                   &devStageClassifier, 
                   &devClassifier, 
                   &devFeature);

    clock_gettime(CLOCK_MONOTONIC, &initEnd);


    elapsed = (initEnd.tv_sec - initStart.tv_sec);
    elapsed += (initEnd.tv_nsec - initStart.tv_nsec) / 1000000000.0;
    cout << elapsed << endl;  
    

    /******************
     * Run detections *
     ******************/

    /*******
     * CPU *
     *******/

    double cpuAvg = 0.0f;
    for (int n = 0; n < RUNS; n++) 
    {
        clock_gettime(CLOCK_MONOTONIC, &cpuStart);
        faces = runCPUdetect(c, i);
        clock_gettime(CLOCK_MONOTONIC, &cpuEnd);
        elapsed = (cpuEnd.tv_sec - cpuStart.tv_sec);
        elapsed += (cpuEnd.tv_nsec - cpuStart.tv_nsec) / 1000000000.0;
        //cout << elapsed << endl;
        cpuAvg += elapsed;
        printf(".");
    }
    cpuAvg /= RUNS;
    printf("\nCPU Average: %f\n", cpuAvg);
    
    //displayResult(i, faces, image_path);

    double ompAvg = 0.0f;
    for (int n = 0; n < RUNS; n++) 
    {
        clock_gettime(CLOCK_MONOTONIC, &cpuStart);
        faces = runCPUdetectOMP(c, i);
        clock_gettime(CLOCK_MONOTONIC, &cpuEnd);
        elapsed = (cpuEnd.tv_sec - cpuStart.tv_sec);
        elapsed += (cpuEnd.tv_nsec - cpuStart.tv_nsec) / 1000000000.0;
        //cout << elapsed << endl;
        ompAvg += elapsed;
        printf(".");
    }
    ompAvg /= RUNS;
    printf("\nOMP Average: %f\n", ompAvg);

    /***********
     * Threads *
     ***********/

    double threadAvg = 0.0f;
    for (int n = 0; n < RUNS; n++)
    {
        clock_gettime(CLOCK_MONOTONIC, &threadStart);
        threadFaces = runThreadDetect(c, i_thread, NTHREADS);
        clock_gettime(CLOCK_MONOTONIC, &threadEnd);
        elapsed = (threadEnd.tv_sec - threadStart.tv_sec);
        elapsed += (threadEnd.tv_nsec - threadStart.tv_nsec) / 1000000000.0;
        //cout << elapsed << endl;
        threadAvg += elapsed;
        printf(".");
    }
    threadAvg /= RUNS;
    printf("\nPthread Average: %f (%d threads)\n", threadAvg, NTHREADS);
    displayResult(i_thread, threadFaces.faces_vec, thread_image_path);

    /*******
     * GPU *
     *******/

    //double gpuElapsed[RUNS];
    double gpuAvg = 0.0f;
    for (int n = 0; n < RUNS; n++)
    {
        clock_gettime(CLOCK_MONOTONIC, &gpuStart);
        faces = runCudaDetection(devCascade, i_gpu, gpuImageData);
        clock_gettime(CLOCK_MONOTONIC, &gpuEnd);
        elapsed = (gpuEnd.tv_sec - gpuStart.tv_sec);
        elapsed += (gpuEnd.tv_nsec - gpuStart.tv_nsec) / 1000000000.0;
        //cout << elapsed << endl;
        gpuAvg += elapsed;
        printf(".");
    }
    gpuAvg /= RUNS;
    printf("\nGPU Average: %f\n", gpuAvg);
    
    //displayResult(i_gpu, faces, image_path);
    double gpuCvAvg = 0.0f;
    Mat gpuFacesMat;
    for (int n = 0; n < RUNS; n++)
    {
        clock_gettime(CLOCK_MONOTONIC, &gpuStart);
        gpuFacesMat = tryIt();
        clock_gettime(CLOCK_MONOTONIC, &gpuEnd);
        elapsed = (gpuEnd.tv_sec - gpuStart.tv_sec);
        elapsed += (gpuEnd.tv_nsec - gpuStart.tv_nsec) / 1000000000.0;
        //cout << elapsed << endl;
        gpuCvAvg += elapsed;
        printf(".");
    }
    gpuCvAvg /= RUNS;
    printf("\nGPU Average: %f\n", gpuCvAvg);

    imshow(image_path, gpuFacesMat);
    waitKey(0);



    /************
     * Clean Up *
     ************/

    freeImData(i);
    freeImData(i_thread);
    freeImData(i_gpu);
    freeGPUCascade(devCascade);
    freeCascadeClassifier(c);
    cvDestroyAllWindows();
    cvReleaseImage(&im);
    cvReleaseImage(&im_thread);

    return 0;
}