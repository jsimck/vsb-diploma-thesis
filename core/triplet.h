#ifndef VSB_SEMESTRAL_PROJECT_TRIPLET_H
#define VSB_SEMESTRAL_PROJECT_TRIPLET_H


#include <opencv2/core/types.hpp>

/*
 * struct Triplet
 * all points are stored in relative locations, we use 12x12 reference points, so every
 * point has x and y coordinates in interval <1, 12>. This allows to adapt reference point locations
 * to each template bounding box.
 */
struct Triplet {
private:
    cv::Point pc;
    cv::Point p1;
    cv::Point p2;
public:
    static const int TRIPLET_SIZE;
    static Triplet createRandomTriplet();

    Triplet(cv::Point pc, cv::Point p1, cv::Point p2) : pc(pc), p1(p1), p2(p2) {}

    cv::Point getCenterPoint(int offsetX, int offsetY, int stepX, int stepY);
    cv::Point getFirstPoint(int offsetX, int offsetY, int stepX, int stepY);
    cv::Point getSecondPoint(int offsetX, int offsetY, int stepX, int stepY);
};


#endif //VSB_SEMESTRAL_PROJECT_TRIPLET_H
