#ifndef VSB_SEMESTRAL_PROJECT_TRIPLET_H
#define VSB_SEMESTRAL_PROJECT_TRIPLET_H

#include <opencv2/core/types.hpp>
#include <ostream>

/*
 * struct Triplet
 * all points are stored in relative locations, we use 12x12 reference points, so every
 * point has x and y coordinates in interval <0, 11>. This allows to adapt reference point locations
 * to each template bounding box.
 */
struct Triplet {
private:
    cv::Point pc;
    cv::Point p1;
    cv::Point p2;

    inline static cv::Point randomPoint(const cv::Size referencePointsGrid);
public:
    static float random(const float rangeMin = 0.0f, const float rangeMax = 1.0f);
    static Triplet createRandomTriplet(const cv::Size referencePointsGrid);

    Triplet() {}
    Triplet(const cv::Point pc, const cv::Point p1, const cv::Point p2) : pc(pc), p1(p1), p2(p2) {}

    cv::Point getCenterCoords(float offsetX, float stepX, float offsetY, float stepY);
    cv::Point getP1Coords(float offsetX, float stepX, float offsetY, float stepY);
    cv::Point getP2Coords(float offsetX, float stepX, float offsetY, float stepY);

    friend std::ostream &operator<<(std::ostream &os, const Triplet &triplet);
};

#endif //VSB_SEMESTRAL_PROJECT_TRIPLET_H
