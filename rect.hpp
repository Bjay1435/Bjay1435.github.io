
#include <opencv2/opencv.hpp>

#ifndef _RECT_HPP
#define _RECT_HPP

using namespace cv;

double calcRect(Mat&, CvRect, CvRect);
double findWindowMean(Mat&, CvRect);

#endif /* _RECT_HPP */