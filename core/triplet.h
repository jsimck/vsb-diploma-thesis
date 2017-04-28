#ifndef VSB_SEMESTRAL_PROJECT_TRIPLET_H
#define VSB_SEMESTRAL_PROJECT_TRIPLET_H

#include <opencv2/core/types.hpp>
#include <ostream>
#include "triplet_params.h"

/**
 * struct Triplet
 *
 * All generated points are stored in relative locations, we use 12x12 reference points, so every
 * point has x and y coordinates in interval <0, 11>. This allows to adapt reference point locations
 * to each template bounding box.
 */
struct Triplet {
private:
    inline static cv::Point randPoint(const cv::Size grid);
    inline static cv::Point randChildPoint(const int min = -4, const int max = 4);
public:
    cv::Point c;
    cv::Point p1;
    cv::Point p2;

    // Statics
    static float random(const float min = 0.0f, const float max = 1.0f);
    static Triplet create(const cv::Size &grid, const int distance = 3); // max distance from center triplet
    static TripletParams getParams(const int width, const int height, const cv::Size &grid, const int sOffsetX = 0, const int sOffsetY = 0);

    // Constructors
    Triplet() {}
    Triplet(const cv::Point c, const cv::Point p1, const cv::Point p2) : c(c), p1(p1), p2(p2) {}

    // Methods
    cv::Point getPoint(int index, const TripletParams &params);
    cv::Point getPoint(int x, int y, const TripletParams &params);
    cv::Point getCenter(const TripletParams &params);
    cv::Point getP1(const TripletParams &params);
    cv::Point getP2(const TripletParams &params);

    void visualize(const cv::Mat &src, const cv::Size &grid, bool showGrid = true);

    // Operators
    friend std::ostream &operator<<(std::ostream &os, const Triplet &triplet);
    bool operator==(const Triplet &rhs) const;
    bool operator!=(const Triplet &rhs) const;
};

#endif //VSB_SEMESTRAL_PROJECT_TRIPLET_H
