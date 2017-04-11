// OpenCV library
#include <opencv2/opencv.hpp>
//#include <opencv2/core/persistence.hpp>
#include <fstream>
#include <stdio.h>
#include <iostream>


using namespace cv;


static CvHaarClassifierCascade* loadCVHaarCascade(const char* cascade_path)
{
    // not sure why we have to do it this way first
    CascadeClassifier cc;
    if (!cc.load(cascade_path)) {
        std::cout << "CASCADE FAIL" << std::endl;
        exit(1);
    }

    return (CvHaarClassifierCascade*)cvLoad( cascade_path );
}

static IplImage* loadIplImage(const char * image_path)
{
    IplImage* image = cvLoadImage(image_path, CV_LOAD_IMAGE_GRAYSCALE);
    if (!image) {
        std::cout << "IMAGE FAIL" << std::endl;
        exit(1);
    }

    return image;
}

static void printStats(CvHaarClassifierCascade* ch)
{
    printf("numStages: %d\n", ch->count);
    for (int i = 0; i < ch->count; i++) {
        //iterate through stage classifiers
        CvHaarStageClassifier stage = ch->stage_classifier[i];
        printf("numClassifiers: %d\n", stage.count);
        for (int j = 0; j < stage.count; j++) {
            CvHaarClassifier classifier = stage.classifier[j];
            printf("count: %d ", classifier.count);
        }
        printf("\n");
    }
}

int main()
{
    const char * cascade_path = "data/haarcascade_frontalface_alt.xml";
    const char * image_path = "images/lena_256.jpg";

    CvHaarClassifierCascade* ch = loadCVHaarCascade(cascade_path);

    printStats(ch);

    IplImage* image = loadIplImage(image_path);

    cvShowImage(image_path, image);
    cvWaitKey(0);

    cvDestroyAllWindows();
    cvReleaseImage(&image);

    return 0;
}