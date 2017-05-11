
#include "util.hpp"
#include "rect.hpp"
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <objdetect_c.h>
#include <opencv2/imgcodecs.hpp>
#include <imgcodecs_c.h>
#include <pthread.h>
#include <stdlib.h>

#ifndef _CPU_THREAD_HPP
#define _CPU_THREAD_HPP


threadVector runThreadDetect(cascadeClassifier_t, imageData_t, int);
void* threadDetectMultiScale(void*);


#endif /* _CPU_DETECTION_HPP */