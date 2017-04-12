#include "util.hpp"
#include <stdio.h>
#include <iostream>

using namespace cv;

/******************************************************************************
 *********************** Cascade Utility Functions ****************************
 ******************************************************************************/

cascadeClassifier_t newCascadeClassifier(CvHaarClassifierCascade* cascade)
{
    cascadeClassifier_t c = (cascadeClassifier_t) malloc(sizeof(struct cascadeClassifier));
    c->cascade = cascade;
    c->numStages = cascade->count;
    return c;
}

//@TODO: free struct


CvHaarClassifierCascade* loadCVHaarCascade(const char* cascade_path)
{
    // not sure why we have to do it this way first
    CascadeClassifier cc;
    if (!cc.load(cascade_path)) {
        std::cout << "CASCADE FAIL" << std::endl;
        exit(1);
    }

    return (CvHaarClassifierCascade*)cvLoad( cascade_path );
}

void printStats(CvHaarClassifierCascade* ch)
{
    printf("numStages: %d\n", ch->count);
    printf("Original Window Size: %dx%d\n", ch->orig_window_size.width, ch->orig_window_size.height);
    printf("Real Window Size: %dx%d\n", ch->real_window_size.width, ch->real_window_size.height);
    printf("Scale: %f\n", ch->scale);
    for (int i = 0; i < ch->count; i++) {
        //iterate through stage classifiers
        CvHaarStageClassifier stage = ch->stage_classifier[i];
        printf("%d, %d, %d\n", stage.next, stage.child, stage.parent);
        printf("%d: numClassifiers: %d\n", i, stage.count);
        for (int j = 0; j < stage.count; j++) {
            CvHaarClassifier classifier = stage.classifier[j];
            printf("t: %f, l: %d, r: %d, a0: %f, a1: %f\n", *classifier.threshold, 
                                        *classifier.left,
                                        *classifier.right, classifier.alpha[0],
                                        classifier.alpha[1]);
            CvHaarFeature feature = *(classifier.haar_feature);
            for (int k = 0; k < CV_HAAR_FEATURE_MAX; k++)
                printf("count: %f ", feature.rect[k].weight);
            printf("\n");
        }
        printf("\n");
    }
}

/******************************************************************************
 ************************* Image Utility Functions ****************************
 ******************************************************************************/

 imageData_t newImageData(IplImage* image)
{
    imageData_t i = (imageData_t) malloc(sizeof(struct imageData));

    int height = image->height;
    int width = image->width;

    //calculate integral image
    CvMat* sum = cvCreateMat(height + 1, width + 1, CV_32SC1);
    CvMat* sqsum = cvCreateMat(height + 1, width + 1, CV_64FC1);

    cvIntegral(image, sum, sqsum);

    i->image = image;
    i->sum = sum;
    i->sqsum = sqsum;
    i->height = height;
    i->width = width;

    return i;
}

IplImage* loadGrayImage(const char * image_path)
{
    IplImage* image = cvLoadImage(image_path, CV_LOAD_IMAGE_GRAYSCALE);
    if (!image) {
        std::cout << "IMAGE FAIL" << std::endl;
        exit(1);
    }

    return image;
}