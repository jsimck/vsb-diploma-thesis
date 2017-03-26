#include <random>
#include <functional>
#include <iostream>
#include <cassert>
#include <opencv2/core/mat.hpp>
#include "triplet.h"

// Init uniform distribution engine
typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<float> Distribution;
auto uniformGenerator = std::bind(Distribution(0.0f, 1.0f), Engine(1));

cv::Point Triplet::getCoords(int pointNum, float offsetX, float stepX, float offsetY, float stepY) {
    cv::Point p;
    switch (pointNum) {
        case 1:
            p = p1;
            break;

        case 2:
            p = p2;
            break;

        case 3:
            p = p3;
            break;

        default:
            break;
    }

    return cv::Point(
        static_cast<int>(offsetX + (p.x * stepX)),
        static_cast<int>(offsetY + (p.y * stepY))
    );
}

cv::Point Triplet::getCoords(int index, const cv::Vec4f &coordinateParams) {
    return getCoords(index, coordinateParams[0], coordinateParams[1], coordinateParams[2], coordinateParams[3]);
}

cv::Point Triplet::getP1Coords(float offsetX, float stepX, float offsetY, float stepY) {
    return getCoords(1, offsetX, stepX, offsetY, stepY);
}

cv::Point Triplet::getP1Coords(const cv::Vec4f &coordinateParams) {
    return getCoords(1, coordinateParams);
}

cv::Point Triplet::getP2Coords(float offsetX, float stepX, float offsetY, float stepY) {
    return getCoords(2, offsetX, stepX, offsetY, stepY);
}

cv::Point Triplet::getP2Coords(const cv::Vec4f &coordinateParams) {
    return getCoords(2, coordinateParams);
}

cv::Point Triplet::getP3Coords(float offsetX, float stepX, float offsetY, float stepY) {
    return getCoords(3, offsetX, stepX, offsetY, stepY);
}

cv::Point Triplet::getP3Coords(const cv::Vec4f &coordinateParams) {
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

cv::Point Triplet::randomPoint(const cv::Size referencePointsGrid) {
    return cv::Point(
        static_cast<int>(random(0, referencePointsGrid.width - 1)),
        static_cast<int>(random(0, referencePointsGrid.height - 1))
    );
}

Triplet Triplet::createRandomTriplet(const cv::Size &referencePointsGrid) {
    // Checks
    assert(referencePointsGrid.width > 0);
    assert(referencePointsGrid.height > 0);

    // Generate points
    cv::Point c(randomPoint(referencePointsGrid));
    cv::Point p1(randomPoint(referencePointsGrid));
    cv::Point p2(randomPoint(referencePointsGrid));

    // Check if p1 == c -> generate new
    while (c == p1) {
        p1 = randomPoint(referencePointsGrid);
    }

    // Check if p3 == c || p3 == p1 -> generate new
    while (c == p2 || p1 == p2) {
        p2 = randomPoint(referencePointsGrid);
    }

    return Triplet(c, p1, p2);
}

cv::Vec4f Triplet::getCoordParams(const int width, const int height, const cv::Size &referencePointsGrid) {
    // Calculate offsets and steps
    float stepX = width / static_cast<float>(referencePointsGrid.width);
    float stepY = height / static_cast<float>(referencePointsGrid.height);
    float offsetX = stepX / 2.0f;
    float offsetY = stepY / 2.0f;

    // Form final vector [offsetX, stepX, offsetY, stepY]
    return cv::Vec4f(offsetX, stepX, offsetY, stepY);
}

std::ostream &operator<<(std::ostream &os, const Triplet &triplet) {
    os << "p1(" << triplet.p1.x  << ", " << triplet.p1.y << "), ";
    os << "p1(" << triplet.p2.x  << ", " << triplet.p2.y << "), ";
    os << "p2(" << triplet.p3.x  << ", " << triplet.p3.y << ") ";
    return os;
}

bool Triplet::operator==(const Triplet &rhs) const {
    // We ignore order of points
    return (p1 == rhs.p1 || p1 == rhs.p2 || p1 == rhs.p3) &&
        (p2 == rhs.p1 || p2 == rhs.p2 || p2 == rhs.p3) &&
        (p3 == rhs.p1 || p3 == rhs.p2 || p3 == rhs.p3);
}

bool Triplet::operator!=(const Triplet &rhs) const {
    return !(rhs == *this);
}