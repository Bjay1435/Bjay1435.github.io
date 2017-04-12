//cpuDetection.cpp

#ifndef _CPU_DETECTION_HPP
#define _CPU_DETECTION_HPP

#include "util.hpp"
#include <opencv2/opencv.hpp>

std::vector<CvRect> runCPUdetect(cascadeClassifier_t, imageData_t);
std::vector<CvRect> cpuDetectAtScale(cascadeClassifier_t, imageData_t, double);
float calculateRectange(Mat&, CvRect, CvRect);
float windowMean(Mat&, CvRect);

#endif /* _CPU_DETECTION_HPP */