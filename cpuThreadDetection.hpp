
#include "util.hpp"
#include "rect.hpp"
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <stdlib.h>

#ifndef _CPU_THREAD_HPP
#define _CPU_THREAD_HPP


threadVector runThreadDetect(cascadeClassifier_t, imageData_t);
void* threadDetectMultiScale(void*);


#endif /* _CPU_DETECTION_HPP */