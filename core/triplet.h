#ifndef VSB_SEMESTRAL_PROJECT_TRIPLET_H
#define VSB_SEMESTRAL_PROJECT_TRIPLET_H

#include <opencv2/core/types.hpp>
#include <ostream>

/**
 * struct Triplet
 *
 * All generated points are stored in relative locations, we use 12x12 reference points, so every
 * point has x and y coordinates in interval <0, 11>. This allows to adapt reference point locations
 * to each template bounding box.
 */
struct Triplet {
private:
    inline static cv::Point randomPoint(const cv::Size referencePointsGrid);
public:
    cv::Point p1;
    cv::Point p2;
    cv::Point p3;

    // Statics
    static float random(const float rangeMin = 0.0f, const float rangeMax = 1.0f);
    static Triplet createRandomTriplet(const cv::Size &referencePointsGrid);
    static cv::Vec4f getCoordParams(const int width, const int height, const cv::Size &referencePointsGrid);

    // Constructors
    Triplet() {}
    Triplet(const cv::Point p1, const cv::Point p2, const cv::Point p3) : p1(p1), p2(p2), p3(p3) {}

    // Methods
    cv::Point getCoords(int index, float offsetX, float stepX, float offsetY, float stepY);
    cv::Point getCoords(int index, const cv::Vec4f &coordinateParams);
    cv::Point getP1Coords(float offsetX, float stepX, float offsetY, float stepY);
    cv::Point getP1Coords(const cv::Vec4f &coordinateParams);
    cv::Point getP2Coords(float offsetX, float stepX, float offsetY, float stepY);
    cv::Point getP2Coords(const cv::Vec4f &coordinateParams);
    cv::Point getP3Coords(float offsetX, float stepX, float offsetY, float stepY);
    cv::Point getP3Coords(const cv::Vec4f &coordinateParams);

    // Operators
    friend std::ostream &operator<<(std::ostream &os, const Triplet &triplet);
    bool operator==(const Triplet &rhs) const;
    bool operator!=(const Triplet &rhs) const;
};

#endif //VSB_SEMESTRAL_PROJECT_TRIPLET_H
