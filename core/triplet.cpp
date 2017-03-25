#include <random>
#include <functional>
#include "triplet.h"

// 12x12 = 144 reference points -> size 12
int Triplet::TRIPLET_SIZE = 12;

// Init uniform distribution engine
typedef std::mt19937 Engine;
typedef std::uniform_int_distribution<int> Distribution;
auto uniformGenerator = std::bind(Distribution(0, Triplet::TRIPLET_SIZE), Engine(1));

cv::Point Triplet::getCenterPoint(int offsetX, int offsetY, int stepX, int stepY) {
    return cv::Point(offsetX + (pc.x * stepX), offsetY + (pc.x * stepY));
}

cv::Point Triplet::getFirstPoint(int offsetX, int offsetY, int stepX, int stepY) {
    return cv::Point(offsetX + (p1.x * stepX), offsetY + (p1.x * stepY));
}

cv::Point Triplet::getSecondPoint(int offsetX, int offsetY, int stepX, int stepY) {
    return cv::Point(offsetX + (p2.x * stepX), offsetY + (p2.x * stepY));
}

Triplet Triplet::createRandomTriplet() {
    cv::Point c(uniformGenerator(), uniformGenerator());
    cv::Point p1(uniformGenerator(), uniformGenerator());
    cv::Point p2(uniformGenerator(), uniformGenerator());

    // TODO - not ideal
    // Check if p1 == c -> generate new
    while (c == p1) {
        p1 = cv::Point(uniformGenerator(), uniformGenerator());
    }

    // Check if p2 == c || p2 == p1 -> generate new
    while (c == p2 || p1 == p2) {
        p2 = cv::Point(uniformGenerator(), uniformGenerator());
    }

    return Triplet(c, p1, p2);
}
