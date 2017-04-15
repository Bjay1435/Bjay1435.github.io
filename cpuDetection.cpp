#include "cpuDetection.hpp"
#include "cascade.hpp"

#define BIAS 0.5f

using namespace std;


float calculateRectangle(Mat& intImage, CvRect rect, CvRect window)
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

    return ret;
}

float windowMean(Mat& sqImage, CvRect window)
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

    return ret;
}

std::vector<CvRect> runCPUdetect(cascadeClassifier_t classifier, imageData_t imData)
{
    std::vector<CvRect> faces;
    std::vector<CvRect> scaleFaces;
    std::vector<double> scales;

    /***********************
     * Get scaling factors *
     ***********************/
    int imageWidth = imData->width;
    int imageHeight = imData->height;

    int startWidth = classifier->cascade->orig_window_size.width;
    int startHeight = classifier->cascade->orig_window_size.height;

    double scale = 1.0;
    double factor = 2;

    while((scale * startWidth < imageWidth) && (scale * startHeight < imageHeight))
    {
        scales.push_back(scale);
        printf("%f, ", scale);
        scale *= factor;
    }
    printf("\n");

    /*****************************************
     * For each window size, run the cascade *
     *****************************************/
    for (size_t i = 0; i < scales.size(); i++) 
    {
        double thisScale = scales[i];
        scaleFaces = cpuDetectAtScale(classifier, imData, thisScale);

        //TODO: append scaleFaces to faces
        faces.insert(faces.end(), scaleFaces.begin(), scaleFaces.end());
    }
    return faces;
}

std::vector<CvRect> cpuDetectAtScale(cascadeClassifier_t cascade, imageData_t imData, double scale)
{
    std::vector<CvRect> faces;

    Mat intImage = *(imData->intImage);
    Mat sqImage  = *(imData->sqImage);

    // Debug
    cout << imData->width << imData->height << endl;
    int windowWidth = cascade->cascade->orig_window_size.width * scale + BIAS;
    int windowHeight = cascade->cascade->orig_window_size.height * scale + BIAS;


    // For each window
    for (int i = 0; i < imData->width - windowWidth; i+=1)
    {
        for (int j = 0; j < imData->height - windowHeight; j+=1)
        {

            CvRect detectionWindow;
            detectionWindow.x = i;
            detectionWindow.y = j;
            detectionWindow.width = windowHeight;
            detectionWindow.height = windowWidth;

            // Normalization calculations here
            // sd^2 = m^2 - 1/N*SUM(x^2)
            double invArea = 1.0f /(detectionWindow.width * detectionWindow.height);

            double winMean = windowMean(intImage, detectionWindow) * invArea;
            double sqSum = windowMean(sqImage, detectionWindow);

            //@TODO: maybe flipped?
            double normalization = winMean * winMean - sqSum * invArea;
            //double normalization = invArea * sqSum - winMean * winMean;

            if (normalization > 1.0f) normalization = sqrt(normalization);
            else normalization = 1.0f;


            bool windowPassed = true;

            //for each stage on this window
            for (int k = 0; k < cascade->numStages; k++) 
            {
                // accumulate stage sum
                double stageSum = 0.f;

                CvHaarStageClassifier stage = cascade->cascade->stage_classifier[k];
                //for each feature in the stage
                for (int m = 0; m < stage.count; m++)
                {
                    double featureSum = 0.f;
                    CvHaarClassifier classifier = stage.classifier[m];
                    CvHaarFeature feature = *(classifier.haar_feature);
                    double threshold = *(classifier.threshold) * normalization;
                    
                    //scale feature
                    CvRect newRect0;
                    CvRect newRect1;
                    CvRect newRect2;

                    newRect0.x = feature.rect[0].r.x * scale + BIAS;
                    newRect0.y = feature.rect[0].r.y * scale + BIAS;
                    newRect0.width = feature.rect[0].r.width * scale + BIAS;
                    newRect0.height = feature.rect[0].r.height * scale + BIAS;

                    newRect1.x = feature.rect[1].r.x * scale + BIAS;
                    newRect1.y = feature.rect[1].r.y * scale + BIAS;
                    newRect1.width = feature.rect[1].r.width * scale + BIAS;
                    newRect1.height = feature.rect[1].r.height * scale + BIAS;

                    if (feature.rect[2].weight) {
                        newRect2.x = feature.rect[2].r.x * scale + BIAS;
                        newRect2.y = feature.rect[2].r.y * scale + BIAS;
                        newRect2.width = feature.rect[2].r.width * scale + BIAS;
                        newRect2.height = feature.rect[2].r.height * scale + BIAS;
                        
                        featureSum += calculateRectangle(intImage, newRect2, detectionWindow)
                                     * feature.rect[2].weight * invArea;
                    }

                    featureSum += calculateRectangle(intImage, newRect0, detectionWindow)
                                 * feature.rect[0].weight * invArea;
                    featureSum += calculateRectangle(intImage, newRect1, detectionWindow)
                                 * feature.rect[1].weight * invArea;


                    if (featureSum >= threshold)
                        stageSum += classifier.alpha[1];
                    else 
                        stageSum += classifier.alpha[0];
                }
                if (stageSum < stage.threshold) {
                    windowPassed = false;
                    break;
                }
            }
            // if we get to here and we haven't broken, we have a face!!
            if (windowPassed) faces.push_back(detectionWindow);
        }
    }
    return faces;
}