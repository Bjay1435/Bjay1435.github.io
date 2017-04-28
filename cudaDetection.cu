
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

texture<int, cudaTextureType2D, cudaReadModeElementType> devIntTex;


void releaseTextures()
{
    cudaUnbindTexture(devIntTex);
}

__global__
void printTest()
{
    printf("In here %d\n", threadIdx.x);
}


__global__
void dbgPrintCascade(CvHaarStageClassifier* ch, int count)
{
    printf("IN\n");
    printf("numStages: %d\n", count);
/*    printf("Original Window Size: %dx%d\n", ch->orig_window_size.width, ch->orig_window_size.height);
    printf("Real Window Size: %dx%d\n", ch->real_window_size.width, ch->real_window_size.height);
    printf("Scale: %f\n", ch->scale);*/
    int numClassifiers = 0;
    for (int i = 0; i < count; i++) {
        //iterate through stage classifiers
        CvHaarStageClassifier stage = ch[i];
        printf("%d, %d, %d\n", stage.next, stage.child, stage.parent);
        printf("%d: numClassifiers: %d\n", i, stage.count);
        for (int j = 0; j < stage.count; j++) {
            //CvHaarClassifier classifier = stage.classifier[j];
            /*printf("t: %f, l: %d, r: %d, a0: %f, a1: %f\n", *classifier.threshold, 
                                        *classifier.left,
                                        *classifier.right, classifier.alpha[0],
                                        classifier.alpha[1]);
            CvHaarFeature feature = *(classifier.haar_feature);*/
            //for (int k = 0; k < CV_HAAR_FEATURE_MAX; k++) 
                //printf("count: %f ", feature.rect[k].weight);
            
            //printf("\n");
            numClassifiers++;
        }
    }
    printf("total: %d\n", numClassifiers);

}

void initGPUcascade(cascadeClassifier_t cascade, CvHaarClassifierCascade* devCascade)
{
    // cudaMalloc space for the cascade
    //CvHaarClassifierCascade devCascade;
    int numClassifiers = printStats(cascade->cascade);
    cudaCheckError(cudaMalloc((void**)&(devCascade->stage_classifier), 
                numClassifiers * sizeof(CvHaarClassifier)));
    cudaCheckError(cudaMemcpy(devCascade->stage_classifier, 
                cascade->cascade->stage_classifier, 
                numClassifiers,
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

std::vector<CvRect> runCudaDetection(cascadeClassifier_t cascade, imageData_t imData) 
{
    //setup();
    printf("in Cuda, %p, %p\n", cascade, imData);
    std::vector<CvRect> faces;
    cudaError_t code;
    int rows = (imData->intImage)->rows;
    int cols = (imData->intImage)->cols;
    dim3 blockDim(THREADS_PER_BLOCK, 1);
    dim3 gridDim((cols + PIXELS_PER_BLOCK_X - 1) / PIXELS_PER_BLOCK_X,
                 (rows + PIXELS_PER_BLOCK_X - 1) / PIXELS_PER_BLOCK_X);
    
    printf("%d, %d, %d, %d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);

    int numClassifiers = printStats(cascade->cascade);
    int numStages = cascade->cascade->count;

    printf("initGPUcascade\n");


    // cudaMalloc space for the cascade
    CvHaarClassifierCascade* devCascade;
    CvHaarStageClassifier* devClassifier;
    //initGPUcascade(cascade, &devCascade);

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

    


    
    printf("Here\n");
    dbgPrintCascade<<<1,1>>>(devCascade->stage_classifier, numStages);
    cudaDeviceSynchronize();
    if (cudaSuccess != (code = cudaGetLastError())) 
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), __FILE__, __LINE__);
        exit(code);
    }
    
    // cudaMalloc space for integral image
    //cudaArray* devIntImage;
    //initGPUimages(*(imData->intImage), devIntImage);
    cudaChannelFormatDesc intChan = cudaCreateChannelDesc(32, 0, 0, 0, 
                                                          cudaChannelFormatKindSigned);
    /*cudaCheckError(cudaMallocArray(&devIntImage, &intChan, 
                    imData->intImage->cols, imData->intImage->rows));
    cudaCheckError(cudaMemcpy2DToArray(devIntImage, 0, 0, imData->intImage->data, 
                        imData->intImage->step, imData->intImage->cols * sizeof(int), 
                        imData->intImage->rows, 
                        cudaMemcpyHostToDevice));
    */

    devIntTex.addressMode[0] = cudaAddressModeWrap;
    devIntTex.addressMode[1] = cudaAddressModeWrap;
    devIntTex.filterMode = cudaFilterModePoint;
    devIntTex.normalized = false;

    //cudaCheckError(cudaBindTextureToArray(devIntTex, devIntImage, intChan));

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