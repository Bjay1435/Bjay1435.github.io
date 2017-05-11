#include "cudaUtil.cuh"
#include <cuda.h>
#include <cuda_runtime.h>


imageData_t newImageDataCuda(const char * image_path)
{
    imageData_t i = (imageData_t) cudaMalloc(sizeof(struct imageData));
    Mat* src = new GpuMat;
    Mat* sum = new GpuMat;
    Mat* sqsum = new GpuMat;
    Mat* norm = new GpuMat;

    (*src) = imread(image_path, IMREAD_GRAYSCALE);

    int height = src->rows;
    int width = src->cols;

    integral(*src, *sum, *sqsum);

    double max;
    minMaxIdx(*sum, 0, &max);

    (*sum).convertTo(*norm, CV_8UC1, 255/max);

    i->image = src;
    i->normInt = norm;
    i->intImage = sum;
    i->sqImage = sqsum;
    i->height = height;
    i->width = width;

    return i;
}
