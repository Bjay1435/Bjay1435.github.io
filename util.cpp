#include "util.hpp"
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

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

 imageData_t newImageData(const char * image_path)
{
    imageData_t i = (imageData_t) malloc(sizeof(struct imageData));
    Mat* src = new Mat;
    Mat* sum = new Mat;
    Mat* sqsum = new Mat;
    Mat* norm = new Mat;

    (*src) = imread(image_path, IMREAD_GRAYSCALE);

    int height = src->rows;
    int width = src->cols;

    integral(*src, *sum, *sqsum);
    //cout << "src = "<< endl << " " << *src << endl << endl;
    //cout << "sum = "<< endl << " " << *sum << endl << endl;


    //(*sum).convertTo(*floatSum, CV_8UC1);

    double max;
    minMaxIdx(*sum, 0, &max);

    (*sum).convertTo(*norm, CV_8UC1, 255/max);

    //imshow("simple output", *sum);//shows normally
    //imshow("normalized output", *norm);

    //waitKey(0);


    i->image = src;
    i->normInt = norm;
    i->intImage = sum;
    i->sqImage = sqsum;
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