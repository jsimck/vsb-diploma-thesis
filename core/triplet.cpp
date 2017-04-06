#include <random>
#include <functional>
#include <iostream>
#include <cassert>
#include <opencv2/core/mat.hpp>
#include <opencv/cv.hpp>
#include "triplet.h"

typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<float> Distribution;
static auto uniformGenerator = std::bind(Distribution(0.0f, 1.0f), Engine(1));

cv::Point Triplet::randomPoint(const cv::Size referencePointsGrid) {
    return cv::Point(
        static_cast<int>(random(0, referencePointsGrid.width - 1)),
        static_cast<int>(random(0, referencePointsGrid.height - 1))
    );
}

cv::Point Triplet::randomPointBoundaries(int min, int max) {
    int y = static_cast<int>(random(min, max));
    int x = static_cast<int>(random(min, max));

    // Check if x || y equals zero, if yes, generate again
    while (x == 0 || y == 0) {
        y = static_cast<int>(random(min, max));
        x = static_cast<int>(random(min, max));
    }

    return cv::Point(x, y);
}

Triplet Triplet::createRandomTriplet(const cv::Size &referencePointsGrid, int maxNeighbourhood) {
    // Checks
    assert(referencePointsGrid.width > 0);
    assert(referencePointsGrid.height > 0);

    // Generate points
    cv::Point p1, p2;
    cv::Point c(randomPoint(referencePointsGrid));

    // Generate other 2 random points within boundaries to the center
    // and check for duplicates and valid coordinates
    do {
        p1 = cv::Point(c + randomPointBoundaries(-maxNeighbourhood, maxNeighbourhood));
        p2 = cv::Point(c + randomPointBoundaries(-maxNeighbourhood, maxNeighbourhood));

        // Check for negative values (simply multiply by -1 to get positive)
        if (p1.x < 0) p1.x *= -1;
        if (p1.y < 0) p1.y *= -1;
        if (p2.x < 0) p2.x *= -1;
        if (p2.y < 0) p2.y *= -1;
    } while (
        p1 == p2
        || p1.x >= referencePointsGrid.width
        || p2.x >= referencePointsGrid.width
        || p1.y >= referencePointsGrid.height
        || p2.y >= referencePointsGrid.height
    );

    return Triplet(c, p1, p2);
}

TripletCoords Triplet::getCoordParams(const int width, const int height, const cv::Size &referencePointsGrid, int sceneOffsetX, int sceneOffsetY) {
    // Calculate offsets and steps for relative grid
    float stepX = width / static_cast<float>(referencePointsGrid.width);
    float stepY = height / static_cast<float>(referencePointsGrid.height);
    float offsetX = stepX / 2.0f;
    float offsetY = stepY / 2.0f;

    return TripletCoords(offsetX, stepX, offsetY, stepY, sceneOffsetX, sceneOffsetY);
}

cv::Point Triplet::getPoint(int x, int y, const TripletCoords &coordinateParams) {
    return getPoint(x, y, coordinateParams.offsetX, coordinateParams.stepX, coordinateParams.offsetY,
                    coordinateParams.stepY, coordinateParams.sceneOffsetX, coordinateParams.sceneOffsetY);
}

cv::Point Triplet::getPoint(int x, int y, float offsetX, float stepX, float offsetY, float stepY, int sceneOffsetX, int sceneOffsetY) {
    return cv::Point(
        static_cast<int>(sceneOffsetX + offsetX + (x * stepX)),
        static_cast<int>(sceneOffsetY + offsetY + (y * stepY))
    );
}

cv::Point Triplet::getCoords(int pointNum, float offsetX, float stepX, float offsetY, float stepY, int sceneOffsetX, int sceneOffsetY) {
    cv::Point p;
    switch (pointNum) {
        case 1:
            p = c;
            break;

        case 2:
            p = p1;
            break;

        case 3:
            p = p2;
            break;

        default:
            break;
    }

    return getPoint(p.x, p.y ,offsetX, stepX, offsetY, stepY, sceneOffsetX, sceneOffsetY);
}

cv::Point Triplet::getCoords(int index, const TripletCoords &coordinateParams) {
    return getCoords(index, coordinateParams.offsetX, coordinateParams.stepX, coordinateParams.offsetY,
                     coordinateParams.stepY, coordinateParams.sceneOffsetX, coordinateParams.sceneOffsetY);
}

cv::Point Triplet::getCenterCoords(float offsetX, float stepX, float offsetY, float stepY, int sceneOffsetX,
                                   int sceneOffsetY) {
    return getCoords(1, offsetX, stepX, offsetY, stepY, sceneOffsetX, sceneOffsetY);
}

cv::Point Triplet::getCenterCoords(const TripletCoords &coordinateParams) {
    return getCoords(1, coordinateParams);
}

cv::Point Triplet::getP1Coords(float offsetX, float stepX, float offsetY, float stepY, int sceneOffsetX,
                               int sceneOffsetY) {
    return getCoords(2, offsetX, stepX, offsetY, stepY, sceneOffsetX, sceneOffsetY);
}

cv::Point Triplet::getP1Coords(const TripletCoords &coordinateParams) {
    return getCoords(2, coordinateParams);
}

cv::Point Triplet::getP2Coords(float offsetX, float stepX, float offsetY, float stepY, int sceneOffsetX,
                               int sceneOffsetY) {
    return getCoords(3, offsetX, stepX, offsetY, stepY, sceneOffsetX, sceneOffsetY);
}

cv::Point Triplet::getP2Coords(const TripletCoords &coordinateParams) {
    return getCoords(3, coordinateParams);
}


float Triplet::random(const float rangeMin, const float rangeMax) {
    float rnd;

    #pragma omp critical (random)
    {
        rnd = static_cast<float>(uniformGenerator());
    }

    return roundf(rnd * (rangeMax - rangeMin) + rangeMin);
}

void Triplet::visualize(const cv::Mat &src, const cv::Size &referencePointsGrid, bool grid) {
    // Checks
    assert(!src.empty());
    assert(src.rows >= referencePointsGrid.height);
    assert(src.cols >= referencePointsGrid.width);
    assert(src.type() == 21); // CV_32FC3

    // Get TripletCoord params
    TripletCoords coordParams = getCoordParams(src.cols, src.rows, referencePointsGrid);

    // Generate grid
    if (grid) {
        for (int y = 0; y < referencePointsGrid.height; ++y) {
            for (int x = 0; x < referencePointsGrid.width; ++x) {
                cv::circle(src, getPoint(x, y, coordParams), 1, cv::Scalar(1, 1, 1), -1);
            }
        }
    }

    // Draw triplets
    cv::line(src, getCenterCoords(coordParams), getP1Coords(coordParams), cv::Scalar(0, 0.5f, 0));
    cv::line(src, getCenterCoords(coordParams), getP2Coords(coordParams), cv::Scalar(0, 0.5f, 0));
    cv::circle(src, getCenterCoords(coordParams), 3, cv::Scalar(0, 0, 1), -1);
    cv::circle(src, getP1Coords(coordParams), 2, cv::Scalar(0, 1, 0), -1);
    cv::circle(src, getP2Coords(coordParams), 2, cv::Scalar(0, 1, 0), -1);
}

std::ostream &operator<<(std::ostream &os, const Triplet &triplet) {
    os << "c(" << triplet.c.x  << ", " << triplet.c.y << "), ";
    os << "p1(" << triplet.p1.x  << ", " << triplet.p1.y << "), ";
    os << "p2(" << triplet.p2.x  << ", " << triplet.p2.y << ") ";
    return os;
}

bool Triplet::operator==(const Triplet &rhs) const {
    // We ignore order of points
    return (c == rhs.c || c == rhs.p1 || c == rhs.p2) &&
        (p1 == rhs.c || p1 == rhs.p1 || p1 == rhs.p2) &&
        (p2 == rhs.c || p2 == rhs.p1 || p2 == rhs.p2);
}

bool Triplet::operator!=(const Triplet &rhs) const {
    return !(rhs == *this);
}