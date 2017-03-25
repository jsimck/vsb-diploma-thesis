#include <random>
#include <functional>
#include <iostream>
#include "triplet.h"

// Init uniform distribution engine
typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<float> Distribution;
auto uniformGenerator = std::bind(Distribution(0.0f, 1.0f), Engine(1));

cv::Point Triplet::getCenterCoords(float offsetX, float stepX, float offsetY, float stepY) {
    return cv::Point(
        static_cast<int>(offsetX + (pc.x * stepX)),
        static_cast<int>(offsetY + (pc.y * stepY))
    );
}

cv::Point Triplet::getP1Coords(float offsetX, float stepX, float offsetY, float stepY) {
    return cv::Point(
        static_cast<int>(offsetX + (p1.x * stepX)),
        static_cast<int>(offsetY + (p1.y * stepY))
    );
}

cv::Point Triplet::getP2Coords(float offsetX, float stepX, float offsetY, float stepY) {
    return cv::Point(
        static_cast<int>(offsetX + (p2.x * stepX)),
        static_cast<int>(offsetY + (p2.y * stepY))
    );
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
        static_cast<int>(random(0, referencePointsGrid.width)),
        static_cast<int>(random(0, referencePointsGrid.height))
    );
}

Triplet Triplet::createRandomTriplet(cv::Size referencePointsGrid) {
    cv::Point c(randomPoint(referencePointsGrid));
    cv::Point p1(randomPoint(referencePointsGrid));
    cv::Point p2(randomPoint(referencePointsGrid));

    // TODO - not ideal, maybe add fixed offset
    // Check if pc == c -> generate new
    while (c == p1) {
        p1 = randomPoint(referencePointsGrid);
    }

    // Check if p2 == c || p2 == pc -> generate new
    while (c == p2 || p1 == p2) {
        p2 = randomPoint(referencePointsGrid);
    }

    return Triplet(c, p1, p2);
}

std::ostream &operator<<(std::ostream &os, const Triplet &triplet) {
    os << "pc: (" << triplet.pc.x  << ", " << triplet.pc.y << ") ";
    os << "pc: (" << triplet.p1.x  << ", " << triplet.p1.y << ") ";
    os << "p1: (" << triplet.p2.x  << ", " << triplet.p2.y << ") ";
    return os;
}
