
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <stdio.h>
#include "cudaDetection.cuh"
#include <opencv2/objdetect.hpp>
#include <objdetect_c.h>
#include <time.h>


#define THREADS_PER_BLOCK_X     32
#define THREADS_PER_BLOCK       (THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_X)

#define BIAS    0.5
#define SCALES  4

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

using namespace cv;
using namespace cv::cuda;


void setup();



texture<int, cudaTextureType2D, cudaReadModeElementType> devIntTex;
texture<int, cudaTextureType2D, cudaReadModeElementType> devSqIntTex;


void releaseTextures()
{
    cudaUnbindTexture(devIntTex);
    cudaUnbindTexture(devSqIntTex);
}



__device__
double calcRect(GPURect rect, GPURect window)
{
    int topX = window.x + rect.x;
    int topY = window.y + rect.y;
    int botX = topX + rect.width;
    int botY = topY + rect.height;

    int topLeft, topRight, botLeft, botRight;

    topLeft  = tex2D(devIntTex, topX, topY);
    topRight = tex2D(devIntTex, topX, botY); 
    botLeft  = tex2D(devIntTex, botX, topY);
    botRight = tex2D(devIntTex, botX, botY);

    int ret = topLeft - topRight - botLeft + botRight;

    return (double) ret;
}


__device__
double findWindowMean(GPURect window)
{
    int topX = window.x;
    int topY = window.y;
    int botX = topX + window.width;
    int botY = topY + window.height;

    int topLeft, topRight, botLeft, botRight;

    topLeft  = tex2D(devIntTex, topX, topY);
    topRight = tex2D(devIntTex, topX, botY); 
    botLeft  = tex2D(devIntTex, botX, topY);
    botRight = tex2D(devIntTex, botX, botY);

    int ret = topLeft - topRight - botLeft + botRight;

    return (double) ret;    
}

/*
 * findSqWindowMean - used for variance calculation
 */
__device__
double findSqWindowMean(GPURect window)
{
    int topX = window.x;
    int topY = window.y;
    int botX = topX + window.width;
    int botY = topY + window.height;

    int topLeft, topRight, botLeft, botRight;
    topLeft  = tex2D(devIntTex, topX, topY);
    topRight = tex2D(devIntTex, topX, botY); 
    botLeft  = tex2D(devIntTex, botX, topY);
    botRight = tex2D(devIntTex, botX, botY);

    int ret = topLeft - topRight - botLeft + botRight;

    return (double) ret;    
}

__device__
double processClassifier(GPUHaarFeature feature, GPURect detectionWindow, double scale)
{
    double featureSum = 0.f;

    GPURect newRect0;
    GPURect newRect1;
    GPURect newRect2;

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
        
        featureSum += calcRect(newRect2, detectionWindow)
                     * feature.rect[2].weight;
    }
    featureSum += calcRect(newRect0, detectionWindow)
                 * feature.rect[0].weight;
    featureSum += calcRect(newRect1, detectionWindow)
                 * feature.rect[1].weight;

    return featureSum;
}


__device__
double getVariance(GPURect detectionWindow)
{
    double invArea = 1.0f /(detectionWindow.width * detectionWindow.height);
    double winMean = findWindowMean(detectionWindow) * invArea;
    double sqSum = findSqWindowMean(detectionWindow);
    double normalization = winMean * winMean - sqSum * invArea;

    if (normalization > 1.0f) normalization = sqrt(normalization);
    else normalization = 1.0f; 

    return normalization;
}


/*
 * Make it like the pthread version
 */
__global__
void cudaDetectKernel_3(GPUHaarClassifierCascade* devCascade, cudaImageData* imData,
                        GPURect* devFaces)
{
    int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    int thisOffset;

    double scale = 4.0;

    //for (int s = 1; s < SCALES; s++)
    {
        int windowWidth = devCascade->orig_window_size.width * scale;;
        int windowHeight = devCascade->orig_window_size.height * scale;

        for (int a = tid; a < imData->width - windowWidth; a += 1024 * 64)
        {
            for (int b = 0; b < imData->height - windowHeight; b+=1)
            {
                thisOffset = (a + b * imData->width);

                GPURect detectionWindow;
                detectionWindow.x = a;
                detectionWindow.y = b;
                detectionWindow.width = windowHeight;
                detectionWindow.height = windowWidth;

                double invArea = 1.0f /(detectionWindow.width * detectionWindow.height);
                double normalization = getVariance(detectionWindow);
                bool windowPassed = true;

                //for each stage on this window

                for (int k = 0; k < devCascade->count; k++) 
                {
                    // accumulate stage sum
                    double stageSum = 0.f;
                    GPUHaarStageClassifier stage = devCascade->stage_classifier[k];
                    
                    //for each feature in the stage
                    for (int m = 0; m < stage.count; m++)
                    {
                        double featureSum = 0.f;
                        GPUHaarClassifier classifier = stage.classifier[m];
                        GPUHaarFeature feature = (classifier.haar_feature);
                        double threshold = (classifier.threshold) * normalization;

                        featureSum = processClassifier(feature, detectionWindow, scale);

                        if (featureSum * invArea >= threshold)
                            stageSum += classifier.alpha1;
                        else 
                            stageSum += classifier.alpha0;
                    }
                    if (stageSum < stage.threshold) {
                        windowPassed = false;
                        break;
                    }
                }
                // if we get to here and we haven't broken, we have a face!!
                if (windowPassed) {
                    devFaces[thisOffset] = detectionWindow;
                }

            }
        }  
        scale *= 2;
    }
}


/*
 * Consistently 7 msec slower than naive implementation...
 */
__global__
void cudaDetectKernel_2(GPUHaarClassifierCascade* devCascade, cudaImageData* imData,
                        GPURect* devFaces, double thisScale = 1.0)
{
    const int threadsPerBlock = 50;
    
    int xCoord = blockIdx.x * blockDim.x + threadIdx.x;
    int yCoord = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = xCoord + yCoord * blockDim.x * gridDim.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int thisOffset;

    double scale = 1.0;

    __shared__ GPUHaarClassifier sharedClassifiers[threadsPerBlock];

    for (int i = 1; i < SCALES; i++)
    {
        int windowWidth = devCascade->orig_window_size.width * scale;
        int windowHeight = devCascade->orig_window_size.height * scale;

        if (xCoord > imData->width - windowWidth || 
            yCoord > imData->height - windowHeight) {
            break;
        }

        thisOffset = offset*SCALES + i;

        GPURect detectionWindow;
        detectionWindow.x = xCoord;
        detectionWindow.y = yCoord;
        detectionWindow.width = windowHeight;
        detectionWindow.height = windowWidth;

        double invArea = 1.0f /(detectionWindow.width * detectionWindow.height);
        double normalization = getVariance(detectionWindow);
        bool windowPassed = true;

        for (int k = 0; k < devCascade->count; k++) 
        {
            GPUHaarStageClassifier stage = devCascade->stage_classifier[k];
            
            if (tid < stage.count && tid < threadsPerBlock)
            {
                sharedClassifiers[tid] = stage.classifier[tid];
            }
            __syncthreads();            
            
            double stageSum = 0.f;
            for (int m = 0; m < stage.count && windowPassed; m++)
            {
                GPUHaarClassifier classifier;
                double featureSum = 0.f;
                if (m < threadsPerBlock) classifier = sharedClassifiers[m];
                else classifier = stage.classifier[m];
                GPUHaarFeature feature = (classifier.haar_feature);
                double threshold = (classifier.threshold) * normalization;

                featureSum = processClassifier(feature, detectionWindow, scale);

                if (featureSum * invArea >= threshold)
                    stageSum += classifier.alpha1;
                else 
                    stageSum += classifier.alpha0;
            }
            if (stageSum < stage.threshold) {
                windowPassed = false;
                //printf("%d\n", k);
            }
        }
        // if we get to here and we haven't broken, we have a face!!
        if (windowPassed) {
            devFaces[thisOffset] = detectionWindow;
            //printf("%d %d %f, ", xCoord, yCoord, scale);
        }
        scale *= 2;
    }
}

/*
 * Kernel looks for faces and draws directly into GpuMat
 * Shared memory not effecient enough for access pattern
 */
__global__
void cudaDetectKernel_1(GPUHaarClassifierCascade* devCascade, cudaImageData* imData,
                        GPURect* devFaces)
{
    int xCoord = (blockIdx.x * blockDim.x + threadIdx.x);// * 3;
    int yCoord = (blockIdx.y * blockDim.y + threadIdx.y);// * 3;
    int offset = xCoord + yCoord * blockDim.x * gridDim.x;
    int thisOffset;

    double scale = 1.0;

    for (int i = 1; i < SCALES; i++)
    {
        //std::vector<CvRect> faces;
        int windowWidth = devCascade->orig_window_size.width * scale;// + BIAS;
        int windowHeight = devCascade->orig_window_size.height * scale;// + BIAS;

        if (xCoord > imData->width - windowWidth || 
            yCoord > imData->height - windowHeight)
            break;

        thisOffset = offset*SCALES + i;

        GPURect detectionWindow;
        detectionWindow.x = xCoord;
        detectionWindow.y = yCoord;
        detectionWindow.width = windowHeight;
        detectionWindow.height = windowWidth;

        double invArea = 1.0f /(detectionWindow.width * detectionWindow.height);
        double normalization = getVariance(detectionWindow);
        bool windowPassed = true;

        //for each stage on this window

        for (int k = 0; k < devCascade->count; k++) 
        {
            // accumulate stage sum
            double stageSum = 0.f;
            GPUHaarStageClassifier stage = devCascade->stage_classifier[k];
            
            //for each feature in the stage
            for (int m = 0; m < stage.count; m++)
            {
                double featureSum = 0.f;
                GPUHaarClassifier classifier = stage.classifier[m];
                GPUHaarFeature feature = (classifier.haar_feature);
                double threshold = (classifier.threshold) * normalization;

                featureSum = processClassifier(feature, detectionWindow, scale);

                if (featureSum * invArea >= threshold)
                    stageSum += classifier.alpha1;
                else 
                    stageSum += classifier.alpha0;
            }
            if (stageSum < stage.threshold) {
                windowPassed = false;
                break;
            }
        }
        // if we get to here and we haven't broken, we have a face!!
        if (windowPassed) {
            devFaces[thisOffset] = detectionWindow;
        }
        scale *= 2;
    }
}



std::vector<CvRect> runCudaDetection(GPUHaarClassifierCascade* devCascade, 
                                     imageData_t imData,
                                     cudaImageData* cudaImData) 
{
    std::vector<CvRect> faces;

    int rows = imData->height;
    int cols = imData->width;
    
    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_X, 1);
    dim3 blocks((cols - THREADS_PER_BLOCK_X + 1) / THREADS_PER_BLOCK_X + 1,
                (rows - THREADS_PER_BLOCK_X + 1) / THREADS_PER_BLOCK_X + 1);


    printf("%d, %d, %d, %d, %d %d\n", cols, rows, blocks.x, blocks.y,
                            blocks.x * threadsPerBlock.x,
                            blocks.y * threadsPerBlock.y);

    //setup();

    /***********************************
     * Set up textures for device code *
     ***********************************/

    cudaArray* devIntImage, *devSqIntImage;

    cudaChannelFormatDesc intChan = cudaCreateChannelDesc(32, 0, 0, 0, 
                                                          cudaChannelFormatKindSigned);
    cudaChannelFormatDesc sqIntChan = cudaCreateChannelDesc(32, 0, 0, 0, 
                                                          cudaChannelFormatKindSigned);
    // Integral Image texture
    cudaCheckError(cudaMallocArray(&devIntImage, &intChan, 
                    imData->intImage->cols, imData->intImage->rows));
    cudaCheckError(cudaMemcpy2DToArray(devIntImage, 0, 0, imData->intImage->data, 
                        imData->intImage->step, imData->intImage->cols * sizeof(int), 
                        imData->intImage->rows, 
                        cudaMemcpyHostToDevice));

    // Sq Integral Image texture
    cudaCheckError(cudaMallocArray(&devSqIntImage, &sqIntChan, 
                    imData->sqImage->cols, imData->sqImage->rows));
    cudaCheckError(cudaMemcpy2DToArray(devSqIntImage, 0, 0, imData->sqImage->data, 
                        imData->sqImage->step, imData->sqImage->cols * sizeof(int), 
                        imData->sqImage->rows, 
                        cudaMemcpyHostToDevice));
    

    devIntTex.addressMode[0] = cudaAddressModeWrap;
    devIntTex.addressMode[1] = cudaAddressModeWrap;
    devIntTex.filterMode = cudaFilterModePoint;
    devIntTex.normalized = false;

    devSqIntTex.addressMode[0] = cudaAddressModeWrap;
    devSqIntTex.addressMode[1] = cudaAddressModeWrap;
    devSqIntTex.filterMode = cudaFilterModePoint;
    devSqIntTex.normalized = false;

    cudaCheckError(cudaBindTextureToArray(devIntTex, devIntImage, intChan));
    cudaCheckError(cudaBindTextureToArray(devSqIntTex, devSqIntImage, sqIntChan));

    /**************************
     * Set up result handlers *
     **************************/

    struct timespec gpuStart, gpuEnd;
    double elapsed;
    GPURect* hostFaces, *devFaces;
    int threads = blocks.x * threadsPerBlock.x * blocks.y * threadsPerBlock.y;
    int resultSize = threads * SCALES * sizeof(GPURect); // 4 slots per thread

    // set up result buffer
    hostFaces = (GPURect*) malloc(resultSize);
    memset(hostFaces, 0, resultSize);
    cudaCheckError(cudaMalloc((void**) &devFaces, resultSize));
    cudaCheckError(cudaMemcpy(devFaces, hostFaces, resultSize, cudaMemcpyHostToDevice));

    /***************************
     * Launch detection kernel *
     ***************************/
    printf("Launch\n");
    clock_gettime(CLOCK_MONOTONIC, &gpuStart);
    //cudaDetectKernel_1<<<threadsPerBlock, blocks>>>(devCascade, cudaImData, devFaces);
    cudaCheckError(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &gpuEnd);
    elapsed = (gpuEnd.tv_sec - gpuStart.tv_sec);
    elapsed += (gpuEnd.tv_nsec - gpuStart.tv_nsec) / 1000000000.0;
    printf("kernel: %f\n", elapsed);

    printf("Launch\n");
    clock_gettime(CLOCK_MONOTONIC, &gpuStart);
    cudaDetectKernel_2<<<threadsPerBlock, blocks>>>(devCascade, cudaImData, devFaces);
    cudaCheckError(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &gpuEnd);
    elapsed = (gpuEnd.tv_sec - gpuStart.tv_sec);
    elapsed += (gpuEnd.tv_nsec - gpuStart.tv_nsec) / 1000000000.0;
    printf("kernel: %f\n", elapsed);

    // version 3 uses fewer thread blocks, with 2 windows per thread, to try
    // to avoid scheduling conflicts
    int threadsPerBlock_v3 = 1024;
    int blocks_v3 = 64;
    printf("Launch\n");
    clock_gettime(CLOCK_MONOTONIC, &gpuStart);
    //cudaDetectKernel_3<<<threadsPerBlock_v3, blocks_v3>>>(devCascade, cudaImData, devFaces);
    cudaCheckError(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &gpuEnd);
    elapsed = (gpuEnd.tv_sec - gpuStart.tv_sec);
    elapsed += (gpuEnd.tv_nsec - gpuStart.tv_nsec) / 1000000000.0;
    printf("kernel: %f\n", elapsed);


    /********************
     * Read return data *
     ********************/

    cudaCheckError(cudaMemcpy(hostFaces, devFaces, resultSize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < threads * SCALES; i++)
    {
        GPURect gpuFace = hostFaces[i];
        if (gpuFace.width != 0)
        {
            CvRect face;
            face.x = hostFaces[i].x;
            face.y = hostFaces[i].y;
            face.height = hostFaces[i].height;
            face.width = hostFaces[i].width;
            faces.push_back(face);
        }
    }

    printf("Done\n");
    releaseTextures();
    return faces;
}




















cudaImageData* newCudaImageData(const char * image_path)
{
    cudaImageData* i;
    cudaCheckError(cudaMalloc(&i, sizeof(struct imageData)));

    GpuMat* gpuSrc = new GpuMat;
    GpuMat* gpuSum = new GpuMat;
    GpuMat* gpuSqsum = new GpuMat;
    GpuMat* gpuNorm = new GpuMat;

    Mat* src = new Mat;
    Mat* sum = new Mat;
    Mat* sqsum = new Mat;
    Mat* norm = new Mat;

    (*src) = imread(image_path, IMREAD_GRAYSCALE);

    int height = src->rows;
    int width = src->cols;

    integral(*src, *sum, *sqsum);

    double max;
    minMaxIdx(*sum, 0, &max);

    (*sum).convertTo(*norm, CV_8UC1, 255/max);

    (*gpuSrc).upload(*src);
    (*gpuSum).upload(*sum);
    (*gpuSqsum).upload(*sum);
    (*gpuNorm).upload(*norm);

    cudaMemcpy(&(i->image), &gpuSrc, sizeof(GpuMat**), cudaMemcpyHostToDevice);
    cudaMemcpy(&(i->normInt), &gpuNorm, sizeof(GpuMat**), cudaMemcpyHostToDevice);
    cudaMemcpy(&(i->intImage), &gpuSum, sizeof(GpuMat**), cudaMemcpyHostToDevice);
    cudaMemcpy(&(i->sqImage), &gpuSqsum, sizeof(GpuMat**), cudaMemcpyHostToDevice);
    cudaMemcpy(&(i->height), &height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(i->width), &width, sizeof(int), cudaMemcpyHostToDevice);

    return i;
}

void allocateGPUCascade(cascadeClassifier_t cascade,
                        GPUHaarClassifierCascade** devCascade,
                        GPUHaarStageClassifier** devStageClassifier,
                        GPUHaarClassifier** devClassifier,
                        GPUHaarFeature** devFeature)
{
    int numStages = cascade->cascade->count;

    cudaCheckError(cudaMalloc(devCascade, sizeof(GPUHaarClassifierCascade)));
    cudaCheckError(cudaMemcpy(&(*devCascade)->count,
                    &numStages,
                    sizeof(int),
                    cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(&(*devCascade)->orig_window_size,
                    &cascade->cascade->orig_window_size,
                    sizeof(CvSize),
                    cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(&(*devCascade)->real_window_size,
                    &cascade->cascade->real_window_size,
                    sizeof(CvSize),
                    cudaMemcpyHostToDevice));

    // devStageClassifier will be an array of stages
    cudaCheckError(cudaMalloc(devStageClassifier, numStages * sizeof(GPUHaarStageClassifier)));
    // for each stage in the cascade, allocate and copy over that stage's data
    for (int i = 0; i < numStages; i++)
    {
        CvHaarStageClassifier stage = cascade->cascade->stage_classifier[i];
        //printf("%d, %f, \n", i, stage.threshold);

        // copy stage info into device stage
        cudaCheckError(cudaMemcpy(&(((*devStageClassifier)[i]).count),
                        &stage.count,
                        sizeof(int),
                        cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(&(((*devStageClassifier)[i]).threshold),
                        &stage.threshold,
                        sizeof(float),
                        cudaMemcpyHostToDevice));

        // devClassifier will be an array 
        cudaCheckError(cudaMalloc(devClassifier, stage.count * sizeof(GPUHaarClassifier)));

        // for each classifier in the stage
        for (int j = 0; j < stage.count; j++)
        {
            //cudaCheckError(cudaDeviceSynchronize());
            CvHaarClassifier classifier = stage.classifier[j];
            //printf("%f, \n", *stage.classifier[j].threshold);

            cudaCheckError(cudaMalloc(devFeature, sizeof(GPUHaarFeature)));

            // copy classifier info from host to device

            /*cudaCheckError(cudaMemcpy(&((*devClassifier)[j].left),
                            classifier.left,
                            sizeof(int),
                            cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(&((*devClassifier)[j].right),
                            classifier.right,
                            sizeof(int),
                            cudaMemcpyHostToDevice));*/
            cudaCheckError(cudaMemcpy(&((*devClassifier)[j].threshold),
                            classifier.threshold,
                            sizeof(float),
                            cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(&((*devClassifier)[j].alpha0),
                            &(classifier.alpha[0]),
                            sizeof(float),
                            cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(&((*devClassifier)[j].alpha1),
                            &(classifier.alpha[1]),
                            sizeof(float),
                            cudaMemcpyHostToDevice));

            // copy feature info from host to device
            for (int k = 0; k < 3; k++) 
            {
                //weight
                //printf("i: %d, j: %d, k: %d, %d ", i, j, k, (*(classifier.haar_feature)).rect[k].r.width);
                cudaCheckError(cudaMemcpy(&((*devFeature)->rect[k].weight),
                            &(classifier.haar_feature->rect[k].weight),
                            sizeof(float),
                            cudaMemcpyHostToDevice));
                // x
                cudaCheckError(cudaMemcpy(&((*devFeature)->rect[k].r.x),
                            &(classifier.haar_feature->rect[k].r.x),
                            sizeof(int),
                            cudaMemcpyHostToDevice));
                // y
                cudaCheckError(cudaMemcpy(&((*devFeature)->rect[k].r.y),
                            &(classifier.haar_feature->rect[k].r.y),
                            sizeof(int),
                            cudaMemcpyHostToDevice));
                // height
                cudaCheckError(cudaMemcpy(&((*devFeature)->rect[k].r.height),
                            &(classifier.haar_feature->rect[k].r.height),
                            sizeof(int),
                            cudaMemcpyHostToDevice));
                // width
                cudaCheckError(cudaMemcpy(&((*devFeature)->rect[k].r.width),
                            &(classifier.haar_feature->rect[k].r.width),
                            sizeof(int),
                            cudaMemcpyHostToDevice));
            }

            // copy feature pointer to device mem into device classifier
            cudaCheckError(cudaMemcpy(&((*devClassifier)[j].haar_feature),
                            *devFeature,
                            sizeof(GPUHaarFeature),
                            cudaMemcpyHostToDevice)); //didn't expect hostToDevice here

        }
        // once each stage has all of its features, copy it into the Cascade

        // copy classifier "array" in to stage

        cudaCheckError(cudaMemcpy(&(((*devStageClassifier)[i]).classifier),
                        devClassifier,
                        sizeof(GPUHaarClassifier*),
                        cudaMemcpyHostToDevice));

    }
    // copy address of stage array to device Cascade
    cudaCheckError(cudaMemcpy(&((*devCascade)->stage_classifier),
                        devStageClassifier,
                        sizeof(GPUHaarStageClassifier*),
                        cudaMemcpyHostToDevice));
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