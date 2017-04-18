#include "util.hpp"


using namespace cv;
using namespace std;


/******************************************************************************
 *********************** Cascade Utility Functions ****************************
 ******************************************************************************/

CvHaarClassifierCascade* loadCVHaarCascade(const char* cascade_path)
{
    // not sure why we have to do it this way first
    CascadeClassifier cc;
    if (!cc.load(cascade_path)) {
        std::cout << "CASCADE FAIL" << std::endl;
        exit(1);
    }

    return (CvHaarClassifierCascade*)cvLoad( cascade_path );
}


cascadeClassifier_t newCascadeClassifier(const char* cascade_path)
{
    CvHaarClassifierCascade* cascade = loadCVHaarCascade(cascade_path);
    cascadeClassifier_t c = (cascadeClassifier_t) malloc(sizeof(struct cascadeClassifier));
    c->cascade = cascade;
    c->numStages = cascade->count;
    return c;
}

//@TODO: free struct
void freeCascadeClassifier(cascadeClassifier_t cascade)
{
    cvReleaseHaarClassifierCascade(&(cascade->cascade));
    free(cascade);
    return;
}


void printStats(CvHaarClassifierCascade* ch)
{
    printf("numStages: %d\n", ch->count);
    printf("Original Window Size: %dx%d\n", ch->orig_window_size.width, ch->orig_window_size.height);
    printf("Real Window Size: %dx%d\n", ch->real_window_size.width, ch->real_window_size.height);
    printf("Scale: %f\n", ch->scale);
    for (int i = 0; i < ch->count; i++) {
        //iterate through stage classifiers
        CvHaarStageClassifier stage = ch->stage_classifier[i];
        printf("%d, %d, %d\n", stage.next, stage.child, stage.parent);
        printf("%d: numClassifiers: %d\n", i, stage.count);
        for (int j = 0; j < stage.count; j++) {
            CvHaarClassifier classifier = stage.classifier[j];
            printf("t: %f, l: %d, r: %d, a0: %f, a1: %f\n", *classifier.threshold, 
                                        *classifier.left,
                                        *classifier.right, classifier.alpha[0],
                                        classifier.alpha[1]);
            CvHaarFeature feature = *(classifier.haar_feature);
            for (int k = 0; k < CV_HAAR_FEATURE_MAX; k++)
                printf("count: %f ", feature.rect[k].weight);
            printf("\n");
        }
        printf("\n");
    }
}

/******************************************************************************
 ************************* Image Utility Functions ****************************
 ******************************************************************************/

imageData_t newImageData(const char * image_path)
{
    imageData_t i = (imageData_t) malloc(sizeof(struct imageData));
    Mat* src = new Mat;
    Mat* sum = new Mat;
    Mat* sqsum = new Mat;
    Mat* norm = new Mat;

    (*src) = imread(image_path, IMREAD_GRAYSCALE);

    int height = src->rows;
    int width = src->cols;

    integral(*src, *sum, *sqsum);
    //cout << "src = "<< endl << " " << *src << endl << endl;
    //cout << "sum = "<< endl << " " << *sum << endl << endl;


    //(*sum).convertTo(*floatSum, CV_8UC1);

    double max;
    minMaxIdx(*sum, 0, &max);

    (*sum).convertTo(*norm, CV_8UC1, 255/max);

    //imshow("simple output", *sum);//shows normally
    //imshow("normalized output", *norm);

    //waitKey(0);


    i->image = src;
    i->normInt = norm;
    i->intImage = sum;
    i->sqImage = sqsum;
    i->height = height;
    i->width = width;

    return i;
}

IplImage* loadGrayImage(const char * image_path)
{
    IplImage* image = cvLoadImage(image_path, CV_LOAD_IMAGE_GRAYSCALE);
    if (!image) {
        std::cout << "IMAGE FAIL" << std::endl;
        exit(1);
    }

    return image;
}

/******************************************************************************
 ******************************** Grouping ************************************
 ******************************************************************************/



int predicate(float eps, CvRect & r1, CvRect & r2)
{
  float delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;

  return abs(r1.x - r2.x) <= delta &&
            abs(r1.y - r2.y) <= delta &&
            abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
            abs(r1.y + r1.height - r2.y - r2.height) <= delta;
}


int partition(std::vector<CvRect>& _vec, std::vector<int>& labels, float eps)
{
    int i, j, N = (int)_vec.size();

    CvRect* vec = &_vec[0];

    const int PARENT=0;
    const int RANK=1;

    std::vector<int> _nodes(N*2);

    int (*nodes)[2] = (int(*)[2])&_nodes[0];

    /* The first O(N) pass: create N single-vertex trees */
    for(i = 0; i < N; i++)
    {
        nodes[i][PARENT]=-1;
        nodes[i][RANK] = 0;
    }

    /* The main O(N^2) pass: merge connected components */
    for( i = 0; i < N; i++ )
    {
        int root = i;

        /* find root */
        while( nodes[root][PARENT] >= 0 )
            root = nodes[root][PARENT];

        for( j = 0; j < N; j++ )
        {
            if( i == j || !predicate(eps, vec[i], vec[j]))
            continue;
            int root2 = j;

            while( nodes[root2][PARENT] >= 0 )
                root2 = nodes[root2][PARENT];

            if( root2 != root )
            {
                /* unite both trees */
                int rank = nodes[root][RANK], rank2 = nodes[root2][RANK];
                if( rank > rank2 )
                    nodes[root2][PARENT] = root;
                else
                {
                    nodes[root][PARENT] = root2;
                    nodes[root2][RANK] += rank == rank2;
                    root = root2;
                }

                int k = j, parent;

                /* compress the path from node2 to root */
                while( (parent = nodes[k][PARENT]) >= 0 )
                {
                    nodes[k][PARENT] = root;
                    k = parent;
                }

                /* compress the path from node to root */
                k = i;
                while( (parent = nodes[k][PARENT]) >= 0 )
                {
                    nodes[k][PARENT] = root;
                    k = parent;
                }
            }
        }
    }

    /* Final O(N) pass: enumerate classes */
    labels.resize(N);
    int nclasses = 0;

    for( i = 0; i < N; i++ )
    {
        int root = i;
        while( nodes[root][PARENT] >= 0 )
            root = nodes[root][PARENT];
      
        /* re-use the rank as the class label */
        if( nodes[root][RANK] >= 0 )
            nodes[root][RANK] = ~nclasses++;

        labels[i] = ~nodes[root][RANK];
    }

    return nclasses;
}


void groupRectangles(std::vector<CvRect> &rectList, int groupThreshold, float eps)
{
    if( groupThreshold <= 0 || rectList.empty() )
    return;

    std::vector<int> labels;

    int nclasses = partition(rectList, labels, eps);

    std::vector<CvRect> rrects(nclasses);
    std::vector<int> rweights(nclasses);

    int i, j, nlabels = (int)labels.size();

    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        rweights[cls]++;
    }
    for( i = 0; i < nclasses; i++ )
    {
        CvRect r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i].x = cvRound(r.x*s);
        rrects[i].y = cvRound(r.y*s);
        rrects[i].width = cvRound(r.width*s);
        rrects[i].height = cvRound(r.height*s);
    }

    rectList.clear();

    for( i = 0; i < nclasses; i++ )
    {
        CvRect r1 = rrects[i];
        int n1 = rweights[i];
        if( n1 <= groupThreshold )
        continue;
        /* filter out small face rectangles inside large rectangles */
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];

            /*********************************
            * if it is the same rectangle, 
            * or the number of rectangles in class j is < group threshold, 
            * do nothing 
            ********************************/
            if( j == i || n2 <= groupThreshold )
                continue;

            CvRect r2 = rrects[j];

            int dx = cvRound( r2.width * eps );
            int dy = cvRound( r2.height * eps );

            if( i != j &&
                r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
            break;
        }

        if( j == nclasses )
        {
        rectList.push_back(r1); // insert back r1
        }
    }
}