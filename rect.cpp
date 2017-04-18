#include "rect.hpp"

using namespace cv;


double calcRect(Mat& intImage, CvRect rect, CvRect window)
{
    int topX = window.x + rect.x;
    int topY = window.y + rect.y;
    int botX = topX + rect.width;
    int botY = topY + rect.height;

    int topLeft, topRight, botLeft, botRight;
    topLeft  = intImage.at<int>(topY, topX);
    topRight = intImage.at<int>(topY, botX); 
    botLeft  = intImage.at<int>(botY, topX);
    botRight = intImage.at<int>(botY, botX);

    int ret = topLeft - topRight - botLeft + botRight;

    return (double) ret;
}

double findWindowMean(Mat& sqImage, CvRect window)
{
    int topX = window.x;
    int topY = window.y;
    int botX = topX + window.width;
    int botY = topY + window.height;

    int topLeft, topRight, botLeft, botRight;
    topLeft  = sqImage.at<int>(topY, topX);
    topRight = sqImage.at<int>(topY, botX); 
    botLeft  = sqImage.at<int>(botY, topX);
    botRight = sqImage.at<int>(botY, botX);

    int ret = topLeft - topRight - botLeft + botRight;

    return (double) ret;
}
