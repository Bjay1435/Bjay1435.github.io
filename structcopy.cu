

void allocateFeature(CvHaarFeature* d_f, CvHaarFeature* h_f)
{
    int size = sizeof(int) + 3*(sizeof(CvRect) + sizeof(float));
    //int size = sizeof(CvHaarFeature);
    cudaMalloc((void**) &d_f, size);
    cudaMemcpy(d_f, h_f, size, cudaMemcpyHostToDevice);
}

void allocateClassifier(CvHaarClassifier* d_c, CvHaarClassifier* h_c)
{
    int classifierSize = sizeof(CvHaarClassifier);
    cudaMalloc((void**) &d_c, classifierSize);
    int* left, right;
    float* threshold, alpha;

    cudaMalloc((void**) &threshold, sizeof(float));
    cudaMalloc((void**) &alpha, sizeof(float));
    cudaMemcpy(threshold, h_c->threshold, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(alpha, h_c->alpha, sizeof(float), cudaMemcpyHostToDevice);

}