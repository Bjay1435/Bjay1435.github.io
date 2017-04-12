#include "cpuDetection.hpp"
#include "cascade.hpp"


float calculateRectangle(Mat& intImage, CvRect rect, CvRect window)
{
    float tx = window.x + rect.x;
    float ty = window.y + rect.y;

    return intImage.at<float>(ty,tx) 
            - intImage.at<float>(ty, tx + rect.width) 
            - intImage.at<float>(ty + rect.height, tx) 
            + intImage.at<float>(ty + rect.height, tx +  rect.width);
}

float windowMean(Mat& intImage, CvRect window)
{
    float topX = window.x;
    float topY = window.y;
    float botX = topX + window.width;
    float botY = topY + window.height;

    return intImage.at<int>(topY,topX) 
            - intImage.at<int>(topY, botX) 
            - intImage.at<int>(botY, topX) 
            + intImage.at<int>(botY + botX);
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
    double factor = 2.0;

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

    Mat intImage = cvarrToMat(imData->sum);

    // For each window
    for (int i = 0; i < imData->width; i++)
    {
        for (int j = 0; j < imData->height; j++)
        {
            CvRect detectionWindow;
            detectionWindow.x = i;
            detectionWindow.y = j;
            detectionWindow.width = cascade->cascade->orig_window_size.width;
            detectionWindow.height = cascade->cascade->orig_window_size.height;

            //@TODO: do normalization calculations here
            // sd^2 = m^2 - 1/N*SUM(x^2)
            float winMean = windowMean(intImage, detectionWindow);
            float sqSum = windowMean(intImage, detectionWindow);
            float invN = 1.0f /(detectionWindow.width * detectionWindow.height);

            //@TODO: maybe flipped?
            float normalization = winMean * winMean - sqSum * sqSum;

            if (normalization >= 0.0f) normalization = sqrt(normalization);
            else normalization = 1.0f;


            bool windowPassed = true;

            //for each stage
            for (int k = 0; k < cascade->numStages; k++) 
            {
                // accumulate stage sum
                float stageSum = 0.f;

                CvHaarStageClassifier stage = cascade->cascade->stage_classifier[k];
                //for each feature in the stage
                for (int m = 0; m < stage.count; m++)
                {
                    float featureSum = 0.f;
                    CvHaarClassifier classifier = stage.classifier[m];
                    CvHaarFeature feature = *(classifier.haar_feature);
                    
                    //scale feature
                    CvRect newRect0;
                    CvRect newRect1;
                    CvRect newRect2;

                    newRect0.x = feature.rect[0].r.x * scale;
                    newRect0.y = feature.rect[0].r.y * scale;
                    newRect0.width = feature.rect[0].r.width * scale;
                    newRect0.height = feature.rect[0].r.height * scale;

                    newRect1.x = feature.rect[1].r.x * scale;
                    newRect1.y = feature.rect[1].r.y * scale;
                    newRect1.width = feature.rect[1].r.width * scale;
                    newRect1.height = feature.rect[1].r.height * scale;

                    if (feature.rect[2].weight) {
                        
                        newRect2.x = feature.rect[2].r.x * scale;
                        newRect2.y = feature.rect[2].r.y * scale;
                        newRect2.width = feature.rect[2].r.width * scale;
                        newRect2.height = feature.rect[2].r.height * scale;
                    }


                    featureSum += calculateRectangle(intImage, newRect0, detectionWindow)
                                 * feature.rect[0].weight * invN;
                    featureSum += calculateRectangle(intImage, newRect1, detectionWindow)
                                 * feature.rect[1].weight * invN;
                    if (feature.rect[2].weight) 
                        featureSum += calculateRectangle(intImage, newRect2, detectionWindow)
                                     * feature.rect[2].weight * invN;

                    if (featureSum >= *(classifier.threshold) * normalization)
                        stageSum += classifier.alpha[1];
                    else 
                        stageSum += classifier.alpha[0];
                }
                if (stageSum >= stage.threshold) {
                    windowPassed = false;
                    break;
                }
            }
            // if we get to here and we haven't broke, we have a face!!
            if (windowPassed) faces.push_back(detectionWindow);
        }
    }
    return faces;
}