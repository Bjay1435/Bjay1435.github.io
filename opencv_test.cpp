#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

#include "util.hpp"
#include "cpuDetection.hpp"


using namespace cv;
using namespace std;


int main()
{
    const char * cascade_path = "data/haarcascade_frontalface_alt.xml";
    const char * image_path = "images/lena_256.jpg";

    CvHaarClassifierCascade* ch = loadCVHaarCascade(cascade_path);

    //printStats(ch);

    IplImage* image = loadGrayImage(image_path);

    cascadeClassifier_t c = newCascadeClassifier(ch);
    imageData_t i = newImageData(image);

    std::vector<CvRect> faces;

    faces = runCPUdetect(c, i);
    cout << faces.size() << endl;

    for (size_t i = 0; i < faces.size(); i++)
    {
        CvRect face = faces[i];
        CvPoint topLeft = cvPoint(face.x, face.y);
        CvPoint botRight = cvPoint(face.x + face.width, face.y + face.height);
        //@TODO: update this ish to the newer version
        cvRectangle(image, topLeft, botRight, CV_RGB(255, 255, 255), 3);
    }


    cvShowImage(image_path, image);
    cvWaitKey(0);

    cvDestroyAllWindows();
    cvReleaseImage(&image);

    return 0;
}