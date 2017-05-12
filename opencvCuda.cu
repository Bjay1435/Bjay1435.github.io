#include <iostream>
#include <opencv2/core/core.hpp>
//#include <opencv2/contrib/contrib.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/cuda/cuda.hpp>
#include <vector>

#include "opencv2/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
 
using namespace std;
using namespace cv;
//using namespace cv::cuda;
 
Mat tryIt()
{
   string cascadeName = "data/haarcascade_frontalface_alt.xml";
 
   Ptr<cuda::CascadeClassifier> cascade_gpu = cuda::CascadeClassifier::create(cascadeName);
   CascadeClassifier cascade_cpu(cascadeName);
   //cascade_gpu = cascade_gpu;
   Mat image = imread("images/lena_256.jpg", IMREAD_GRAYSCALE);

   vector<Rect> cpuFaces; 

   int gpuCnt = cuda::getCudaEnabledDeviceCount();   // gpuCnt >0 if CUDA device detected
   if(gpuCnt==0) return image;  // no CUDA device found, quit
 
   if(image.empty() || !cascade_gpu)
     return image;  // failed to load cascade file, quit
 
   Mat frame;
   long frmCnt = 0;
   double totalT = 0.0;
   //printf("here\n");
   bool findLargestObject = false;
   bool filterRects = true;

   double t = (double) getTickCount();
/*
   std::vector<Rect> outputsCpu;
   cascade_cpu.detectMultiScale(image, outputsCpu, 1.2, 3, 0);

*/
   cuda::GpuMat faces;
   cuda::GpuMat gray_gpu(image);  // copy the gray image to GPU memory
   equalizeHist(image, image);

   cascade_gpu->setFindLargestObject(findLargestObject);
   cascade_gpu->setScaleFactor(1.2);
   cascade_gpu->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);

   cascade_gpu->detectMultiScale(gray_gpu, faces);  // retrieve results from GPU


   cascade_gpu->convert(faces, cpuFaces);

   t=((double) getTickCount()-t) / getTickFrequency();  // check how long did it take to detect face
   totalT += t;
   frmCnt++;

   //printf("here too tho\n");
   for(int i=0;i<cpuFaces.size();++i)
   {
      Point pt1 = cpuFaces[i].tl();
      Size sz = cpuFaces[i].size();
      Point pt2(pt1.x+sz.width, pt1.y+sz.height);
      rectangle(image, pt1, pt2, CV_RGB(255,255,255), 3);
   }  // retrieve all detected faces and draw rectangles for visualization
   
   //imshow("faces", image);
   //waitkey(0);
/*   for(int i=0;i<outputsCpu.size();++i)
   {
      Point pt1 = outputsCpu[i].tl();
      Size sz = outputsCpu[i].size();
      Point pt2(pt1.x+sz.width, pt1.y+sz.height);
      rectangle(image, pt1, pt2, CV_RGB(255,255,255), 3);
   }  // retrieve all detected faces and draw rectangles for visualization
*/
 
   //cout << "fps: " << 1.0/(totalT/(double)frmCnt) << endl;
   return image;
}