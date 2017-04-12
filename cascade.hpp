
typedef struct rect
{
    float x;
    float y;
    float width;
    float height;
    float weight;
} rect;

typedef struct feature
{
    rect rect0;
    rect rect1;
    rect rect2;
} feature;

typedef struct stage {
    int numClassifiers;
    float threshold;

} stage;

/*
 * myCascade's primary data structure is an array of classifiers made by flattening
 * OpenCV's tree structure. Additionally, the class holds a copy of the scaled
 * data structure currently being operated on
 */
class myCascade
{
public:
    int getCount();

private:
    int numStages;
    int numClassifiers;
    CvSize trained_size;
    double scale;

    stage* cascadeStage;

};