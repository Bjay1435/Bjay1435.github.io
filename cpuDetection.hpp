
#include "util.hpp"
#include "rect.hpp"
#include <opencv2/opencv.hpp>

#ifndef _CPU_DETECTION_HPP
#define _CPU_DETECTION_HPP


std::vector<CvRect> runCPUdetect(cascadeClassifier_t, imageData_t);
std::vector<CvRect> cpuDetectAtScale(cascadeClassifier_t, imageData_t, double);


#endif /* _CPU_DETECTION_HPP */