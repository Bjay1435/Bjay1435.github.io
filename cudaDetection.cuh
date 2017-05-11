
#include "util.hpp"
#include "rect.hpp"

#include <opencv2/core/cuda.hpp>

#ifndef _CPU_DETECTION_CUH
#define _CPU_DETECTION_CUH

using namespace cv;
using namespace cv::cuda;

struct cudaImageData {
    GpuMat* image;
    int height;
    int width;
    GpuMat* normInt;
    GpuMat* intImage;
    GpuMat* sqImage;
};
typedef struct cudaImageData cudaImageData;

typedef struct GPURect 
{
    int x;
    int y;
    int width;
    int height;
} GPURect;

typedef struct rect
{
    GPURect r;
    float weight;
} myRect;

typedef struct GPUHaarFeature
{
    myRect rect[3];
} GPUHaarFeature;

typedef struct GPUHaarClassifier 
{
    //int count;
    GPUHaarFeature haar_feature; // no longer a pointer
    float threshold;
    float alpha0, alpha1; //unpacked
} GPUHaarClassifier;

typedef struct GPUHaarStageClassifier
{
    int  count;
    float threshold;
    GPUHaarClassifier* classifier;

    int next;
    int child;
    int parent;
} GPUHaarStageClassifier;


typedef struct GPUHaarClassifierCascade
{
    int  flags;
    int  count;
    CvSize orig_window_size;
    CvSize real_window_size;
    double scale;
    GPUHaarStageClassifier* stage_classifier;
} GPUHaarClassifierCascade;

cudaImageData* newCudaImageData(const char*);

std::vector<CvRect> runCudaDetection(GPUHaarClassifierCascade*, imageData_t, cudaImageData*);

void allocateGPUCascade(cascadeClassifier_t,
                        GPUHaarClassifierCascade**,
                        GPUHaarStageClassifier**,
                        GPUHaarClassifier**,
                        GPUHaarFeature**);



#endif /* _CPU_DETECTION_CUH */