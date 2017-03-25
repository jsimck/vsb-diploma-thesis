#include <random>
#include <functional>
#include <iostream>
#include "triplet.h"

// Init uniform distribution engine
typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<float> Distribution;
auto uniformGenerator = std::bind(Distribution(0.0f, 1.0f), Engine(1));

cv::Point Triplet::getCoords(int pointNum, float offsetX, float stepX, float offsetY, float stepY) {
    cv::Point p;
    switch (pointNum) {
        case 1:
            p = this->p1;
            break;

        case 2:
            p = this->p2;
            break;

        case 3:
            p = this->p3;
            break;

        default:
            break;
    }

    return cv::Point(
        static_cast<int>(offsetX + (p.x * stepX)),
        static_cast<int>(offsetY + (p.y * stepY))
    );
}

cv::Point Triplet::getP1Coords(float offsetX, float stepX, float offsetY, float stepY) {
    return getCoords(1, offsetX, stepX, offsetY, stepY);
}

cv::Point Triplet::getP2Coords(float offsetX, float stepX, float offsetY, float stepY) {
    return getCoords(2, offsetX, stepX, offsetY, stepY);
}

cv::Point Triplet::getP3Coords(float offsetX, float stepX, float offsetY, float stepY) {
    return getCoords(3, offsetX, stepX, offsetY, stepY);
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

Triplet Triplet::createRandomTriplet(cv::Size referencePointsGrid) {
    cv::Point c(randomPoint(referencePointsGrid));
    cv::Point p1(randomPoint(referencePointsGrid));
    cv::Point p2(randomPoint(referencePointsGrid));

    // TODO - not ideal, maybe add fixed offset
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

std::ostream &operator<<(std::ostream &os, const Triplet &triplet) {
    os << "p1(" << triplet.p1.x  << ", " << triplet.p1.y << "), ";
    os << "p1(" << triplet.p2.x  << ", " << triplet.p2.y << "), ";
    os << "p2(" << triplet.p3.x  << ", " << triplet.p3.y << ") ";
    return os;
}
