
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <objdetect_c.h>
#include <opencv2/imgcodecs.hpp>
#include <imgcodecs_c.h>
#include <opencv2/core/cuda.hpp>

#ifndef _RECT_HPP
#define _RECT_HPP

using namespace cv;

double calcRect(Mat&, CvRect, CvRect);
double findWindowMean(Mat&, CvRect);

#endif /* _RECT_HPP */