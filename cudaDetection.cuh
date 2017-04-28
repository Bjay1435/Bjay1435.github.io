
//#include <cuda.h>
//#include <cuda_runtime.h>
#include "util.hpp"
#include "rect.hpp"
//#include <opencv2/opencv.hpp>

#ifndef _CPU_DETECTION_CUH
#define _CPU_DETECTION_CUH


std::vector<CvRect> runCudaDetection(cascadeClassifier_t, imageData_t);




#endif /* _CPU_DETECTION_CUH */