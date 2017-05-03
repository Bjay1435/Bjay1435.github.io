
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cudaDetection.cuh"
#include <opencv2/opencv.hpp>

#define THREADS_PER_BLOCK_X     20
#define THREADS_PER_BLOCK       (THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_X)

#define PIXELS_PER_THREAD_X     1
#define PIXELS_PER_THREAD       (PIXELS_PER_BLOCK_X * PIXELS_PER_BLOCK_X)

#define PIXELS_PER_BLOCK_X      (THREADS_PER_BLOCK_X * PIXELS_PER_THREAD_X)

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", 
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif

void setup();

texture<int, cudaTextureType2D, cudaReadModeElementType> devIntTex;


void releaseTextures()
{
    cudaUnbindTexture(devIntTex);
}

void initGPUcascade(cascadeClassifier_t cascade, CvHaarClassifierCascade* devCascade)
{
    int numClassifiers = cascade->numClassifiers;
    int numStages = cascade->numStages;
    // cudaMalloc space for the cascade
    devCascade = (CvHaarClassifierCascade*) malloc(sizeof(CvHaarClassifier));
    devCascade->count = numStages;

    //cudaCheckError(cudaMalloc((void**)&(devCascade), sizeof(CvHaarClassifierCascade)));
    printf("cascade allocated\n");
    cudaCheckError(cudaMalloc((void**)&(devCascade->stage_classifier), 
                    numClassifiers * sizeof(CvHaarStageClassifier)));

    printf("stages allocated\n");

    cudaCheckError(cudaMemcpy(devCascade->stage_classifier, 
                cascade->cascade->stage_classifier, 
                numClassifiers * sizeof(CvHaarStageClassifier),
                cudaMemcpyHostToDevice));
}

void initGPUimages(Mat& intImage, cudaArray* devIntImage)
{
    printf("init\n");
    cudaChannelFormatDesc intChan = cudaCreateChannelDesc(32, 0, 0, 0, 
                                                          cudaChannelFormatKindSigned);
    cudaCheckError(cudaMallocArray(&devIntImage, &intChan, 
                    intImage.cols, intImage.rows));
    cudaCheckError(cudaMemcpy2DToArray(devIntImage, 0, 0, intImage.data, 
                        intImage.step, intImage.cols * sizeof(int), intImage.rows, cudaMemcpyHostToDevice));

    devIntTex.addressMode[0] = cudaAddressModeWrap;
    devIntTex.addressMode[1] = cudaAddressModeWrap;
    devIntTex.filterMode = cudaFilterModePoint;
    devIntTex.normalized = false;

    cudaCheckError(cudaBindTextureToArray(devIntTex, devIntImage, intChan)); 

}


__global__
void printTest()
{
    printf("In here %d\n", threadIdx.x);
}


/*
 * dbgPrintCascade
 *
 * Test cuda kernel make sure the casacde data structure is properly loaded
 */
__global__
void dbgPrintCascade(CvHaarStageClassifier* ch, int count)
{
    printf("IN\n");
    printf("numStages: %d\n", count);
/*    printf("Original Window Size: %dx%d\n", ch->orig_window_size.width, ch->orig_window_size.height);
    printf("Real Window Size: %dx%d\n", ch->real_window_size.width, ch->real_window_size.height);
    printf("Scale: %f\n", ch->scale);*/
    int numClassifiers = 0;
    __syncthreads();
    for (int i = 0; i < count; i++) {
        //iterate through stage classifiers
        CvHaarStageClassifier stage = ch[i];
        printf("%d, %d, %d\n", stage.next, stage.child, stage.parent);
        printf("%d: numClassifiers: %d\n", i, stage.count);
        numClassifiers += stage.count;
        __syncthreads();
        for (int j = 0; j < stage.count; j++) {
            printf("%p, %p\n", stage, stage.classifier);
            CvHaarClassifier classifier = stage.classifier[j];
            printf("t: %f, l: %d, r: %d, a0: %f, a1: %f\n", *classifier.threshold, 
                                        *classifier.left,
                                        *classifier.right, classifier.alpha[0],
                                        classifier.alpha[1]);
            
            //CvHaarFeature feature = *(classifier.haar_feature);
            //for (int k = 0; k < CV_HAAR_FEATURE_MAX; k++) 
                //printf("count: %f ", feature.rect[k].weight);
            
            //printf("\n");
            //numClassifiers++;
        }
    }
    printf("total: %d\n", numClassifiers);
}

__global__
void setFeatures(CvHaarClassifier*& devClassifier, CvHaarFeature*& devFeature)
{
    (devClassifier)->haar_feature = devFeature;
}


std::vector<CvRect> runCudaDetection(cascadeClassifier_t cascade, imageData_t imData) 
{
    //setup();
    printf("in Cuda, %p, %p\n", cascade, imData);
    std::vector<CvRect> faces;

    int rows = (imData->intImage)->rows;
    int cols = (imData->intImage)->cols;
    dim3 blockDim(THREADS_PER_BLOCK, 1);
    dim3 gridDim((cols + PIXELS_PER_BLOCK_X - 1) / PIXELS_PER_BLOCK_X,
                 (rows + PIXELS_PER_BLOCK_X - 1) / PIXELS_PER_BLOCK_X);
    
    printf("%d, %d, %d, %d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);

    int numClassifiers = cascade->numClassifiers;
    int numStages = cascade->cascade->count;



    printf("initGPUcascade\n");

    CvHaarClassifierCascade* devCascade;
    CvHaarStageClassifier* devStageClassifier;
    CvHaarClassifier* devClassifier;
    CvHaarFeature* devFeature;


    cudaCheckError(cudaMalloc(&devCascade, sizeof(CvHaarClassifierCascade)));
    cudaCheckError(cudaMemcpy(devCascade,
                    cascade->cascade,
                    sizeof(CvHaarClassifierCascade),
                    cudaMemcpyHostToDevice));

    // devStageClassifier will be an array of stages
    cudaCheckError(cudaMalloc(&devStageClassifier, numStages * sizeof(CvHaarStageClassifier)));
    // for each stage in the cascade, allocate and copy over that stage's data
    for (int i = 0; i < numStages; i++)
    {
        CvHaarStageClassifier stage = cascade->cascade->stage_classifier[i];

        // copy stage info into device stage
        cudaCheckError(cudaMemcpy(&devStageClassifier[i],
                        &stage,
                        sizeof(CvHaarStageClassifier),
                        cudaMemcpyHostToDevice));

        // devClassifier will be an array 
        cudaCheckError(cudaMalloc(&devClassifier, stage.count * sizeof(CvHaarClassifier)));

        // for each classifier in the stage
        for (int j = 0; j < stage.count; j++)
        {
            //printf("%d ", j);
            //cudaCheckError(cudaDeviceSynchronize());
            CvHaarClassifier classifier = stage.classifier[j];

            //copy over details
            int *left, *right;
            float *threshold, *alpha;
            cudaCheckError(cudaMalloc(&left, sizeof(int)));
            cudaCheckError(cudaMalloc(&right, sizeof(int)));
            cudaCheckError(cudaMalloc(&threshold, sizeof(float)));
            cudaCheckError(cudaMalloc(&alpha, 2 * sizeof(float)));
            
            cudaCheckError(cudaMemcpy(left,
                            classifier.left,
                            sizeof(int),
                            cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(right,
                            classifier.right,
                            sizeof(int),
                            cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(threshold,
                            classifier.threshold,
                            sizeof(float),
                            cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(alpha,
                            classifier.alpha,
                            2 * sizeof(float),
                            cudaMemcpyHostToDevice));


            cudaCheckError(cudaMalloc(&devFeature, sizeof(CvHaarFeature)));

            // copy classifier info from host to device
            cudaCheckError(cudaMemcpy(&devClassifier[j],
                            &stage.classifier[j],
                            sizeof(CvHaarClassifier),
                            cudaMemcpyHostToDevice));

            cudaCheckError(cudaMemcpy(&(devClassifier[j].left),
                            &left,
                            sizeof(int*),
                            cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(&(devClassifier[j].right),
                            &right,
                            sizeof(int*),
                            cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(&(devClassifier[j].threshold),
                            &threshold,
                            sizeof(float*),
                            cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(&(devClassifier[j].alpha),
                            &alpha,
                            sizeof(float*),
                            cudaMemcpyHostToDevice));

            // copy feature info from host to device
            // only handles count here
            cudaCheckError(cudaMemcpy(devFeature,
                            classifier.haar_feature,
                            sizeof(CvHaarFeature),
                            cudaMemcpyHostToDevice));


            // copy feature pointer to device mem into device classifier
            cudaCheckError(cudaMemcpy(&(devClassifier[j].haar_feature),
                            &devFeature,
                            sizeof(CvHaarFeature*),
                            cudaMemcpyHostToDevice)); //didn't expect hostToDevice here

        }
        // once each stage has all of its features, copy it into the Cascade

        // copy classifier "array" in to stage
        cudaCheckError(cudaMemcpy(&(devStageClassifier[i].classifier),
                        &devClassifier,
                        sizeof(CvHaarClassifier*),
                        cudaMemcpyHostToDevice));

    }
    // copy address of stage array to device Cascade
    cudaCheckError(cudaMemcpy(&(devCascade->stage_classifier),
                        &devStageClassifier,
                        sizeof(CvHaarStageClassifier*),
                        cudaMemcpyHostToDevice));


    // dbg
    CvHaarStageClassifier* cpyStage = (CvHaarStageClassifier*) malloc(sizeof(CvHaarStageClassifier));

    int * test;
    cudaCheckError(cudaMalloc((void**)&test, sizeof(int)));

/*    cudaCheckError(cudaMemcpy(cpyStage, 
                devCascade->stage_classifier, 
                numStages * sizeof(CvHaarStageClassifier),
                cudaMemcpyDeviceToHost));

    printf("cpyStage: %p, %f\n", cpyStage, cpyStage[1].classifier[0].haar_feature[0].rect[1].weight);
*/

    //printStats(devCascade);

    printf("Here %d\n", numStages * sizeof(CvHaarStageClassifier));
    dbgPrintCascade<<<1,1>>>(devStageClassifier, numStages);
    //printTest<<<1,1>>>();
    cudaCheckError(cudaDeviceSynchronize());

    
    // cudaMalloc space for integral image
    cudaArray* devIntImage;
    //initGPUimages(*(imData->intImage), devIntImage);
    cudaChannelFormatDesc intChan = cudaCreateChannelDesc(32, 0, 0, 0, 
                                                          cudaChannelFormatKindSigned);
    cudaCheckError(cudaMallocArray(&devIntImage, &intChan, 
                    imData->intImage->cols, imData->intImage->rows));
    cudaCheckError(cudaMemcpy2DToArray(devIntImage, 0, 0, imData->intImage->data, 
                        imData->intImage->step, imData->intImage->cols * sizeof(int), 
                        imData->intImage->rows, 
                        cudaMemcpyHostToDevice));
    

    devIntTex.addressMode[0] = cudaAddressModeWrap;
    devIntTex.addressMode[1] = cudaAddressModeWrap;
    devIntTex.filterMode = cudaFilterModePoint;
    devIntTex.normalized = false;

    cudaCheckError(cudaBindTextureToArray(devIntTex, devIntImage, intChan));

    //dummy code
    CvRect detectionWindow;
    detectionWindow.x = 1;
    detectionWindow.y = 1;
    detectionWindow.width  = 2;
    detectionWindow.height = 2;
    faces.push_back(detectionWindow);
    printf("Done\n");

    return faces;
}

__global__
void roiKernel(std::vector<double>* faces, std::vector<double>* scales)
{
    return;
}




void setup() 
{
    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

}