#include "cpuThreadDetection.hpp"
#include "rect.hpp"

#define BIAS 0.5f
#define NTHREADS 32

using namespace std;

struct args {
    int tid;
    std::vector<double>* scales;
    cascadeClassifier_t cascade;
    imageData_t imData;
    std::vector<CvRect>* retFaces;
} arguments;
typedef struct args* args_t;

static pthread_t threads[NTHREADS];


threadVector runThreadDetect(cascadeClassifier_t classifier, imageData_t imData)
{
    threadVector faces = threadVector();
    std::vector<CvRect>* scaleFaces[NTHREADS];
    args_t args[NTHREADS];
    std::vector<double>* scales = new std::vector<double>;


    /**************************
     * Create scaling factors *
     **************************/

    int imageWidth = imData->width;
    int imageHeight = imData->height;

    int startWidth = classifier->cascade->orig_window_size.width;
    int startHeight = classifier->cascade->orig_window_size.height;

    double scale = 1.0;
    double factor = 2;

    while((scale * startWidth < imageWidth) && (scale * startHeight < imageHeight))
    {
        (*scales).push_back(scale);
        scale *= factor;
    }

    /*******************
     * Spin up threads *
     *******************/

    for (int i = 0; i < NTHREADS; i++)
    {

        args[i] = (args_t) malloc(sizeof(struct args));
        args[i]->tid = i;
        args[i]->scales = scales;
        args[i]->cascade = classifier;
        args[i]->imData = imData;
        scaleFaces[i] = new std::vector<CvRect>;
        args[i]->retFaces = scaleFaces[i];

        pthread_create(&threads[i], NULL, &threadDetectMultiScale, (void*) args[i]);
    }

    /****************
     * Reap threads *
     ****************/

    void* ret;
    for (int i = 0; i < NTHREADS; i++)
    {
        pthread_join(threads[i], &ret);
        //printf("%lu\n", (*(args[i]->retFaces)).size());
        faces.combineWith(*(args[i]->retFaces));
        free(args[i]);
    }

    return faces;
}


void* threadDetectMultiScale(void* arg)
{
    args_t args = (args_t) arg;

    int tid = args->tid;
    std::vector<double>* scales = (args->scales);
    cascadeClassifier_t cascade = args->cascade;
    imageData_t imData = args->imData;

    Mat intImage = *(imData->intImage);
    Mat sqImage  = *(imData->sqImage);

    for (size_t s = 0; s < (*scales).size(); s++) 
    {
        double scale = (*scales)[s];

        int windowWidth  = cascade->cascade->orig_window_size.width * scale + BIAS;
        int windowHeight = cascade->cascade->orig_window_size.height * scale + BIAS;

        // For each window
        for (int i = tid; i < imData->width - windowWidth; i+=NTHREADS)
        {
            for (int j = 0; j < imData->height - windowHeight; j+=1)
            {

                CvRect detectionWindow;
                detectionWindow.x = i;
                detectionWindow.y = j;
                detectionWindow.width  = windowHeight;
                detectionWindow.height = windowWidth;

                // Normalization calculations here
                // sd^2 = m^2 - 1/N*SUM(x^2)
                double invArea = 1.0f /(detectionWindow.width * detectionWindow.height);

                double winMean = findWindowMean(intImage, detectionWindow) * invArea;
                double sqSum   = findWindowMean(sqImage, detectionWindow);

                //@TODO: maybe flipped?
                double normalization = winMean * winMean - sqSum * invArea;
                //double normalization = invArea * sqSum - winMean * winMean;

                if (normalization > 1.0f) 
                    normalization = sqrt(normalization);
                else 
                    normalization = 1.0f;

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
                        newRect0.width  = feature.rect[0].r.width * scale + BIAS;
                        newRect0.height = feature.rect[0].r.height * scale + BIAS;

                        newRect1.x = feature.rect[1].r.x * scale + BIAS;
                        newRect1.y = feature.rect[1].r.y * scale + BIAS;
                        newRect1.width  = feature.rect[1].r.width * scale + BIAS;
                        newRect1.height = feature.rect[1].r.height * scale + BIAS;

                        if (feature.rect[2].weight) {
                            newRect2.x = feature.rect[2].r.x * scale + BIAS;
                            newRect2.y = feature.rect[2].r.y * scale + BIAS;
                            newRect2.width  = feature.rect[2].r.width * scale + BIAS;
                            newRect2.height = feature.rect[2].r.height * scale + BIAS;
                            
                            featureSum += calcRect(intImage, newRect2, detectionWindow)
                                         * feature.rect[2].weight;
                        }

                        featureSum += calcRect(intImage, newRect0, detectionWindow)
                                     * feature.rect[0].weight;
                        featureSum += calcRect(intImage, newRect1, detectionWindow)
                                     * feature.rect[1].weight;


                        if (featureSum * invArea >= threshold)
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
                if (windowPassed) 
                    (args->retFaces)->push_back(detectionWindow);
            }
        }
    }
    return NULL;
}