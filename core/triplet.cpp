#include <random>
#include <functional>
#include <iostream>
#include <cassert>
#include <opencv2/core/mat.hpp>
#include <opencv/cv.hpp>
#include "triplet.h"

std::random_device seed;
typedef std::mt19937 engine;
typedef std::uniform_real_distribution<float> distribution;
static auto uniformGenerator = std::bind(distribution(0.0f, 1.0f), engine(seed()));

float Triplet::random(const float min, const float max) {
    float rnd;

#pragma omp critical (random)
    {
        rnd = static_cast<float>(uniformGenerator());
    }

    return roundf(rnd * (max - min) + min);
}

cv::Point Triplet::randPoint(const cv::Size grid) {
    return cv::Point(
        static_cast<int>(random(0, grid.width - 1)),
        static_cast<int>(random(0, grid.height - 1))
    );
}

cv::Point Triplet::randChildPoint(int min, int max) {
    int y = static_cast<int>(random(min, max));
    int x = static_cast<int>(random(min, max));

    // Check if x || y equals zero, if yes, generate again
    while (x == 0 || y == 0) {
        y = static_cast<int>(random(min, max));
        x = static_cast<int>(random(min, max));
    }

    return cv::Point(x, y);
}

Triplet Triplet::create(const cv::Size &grid, const int distance) {
    // Checks
    assert(grid.width > 0);
    assert(grid.height > 0);

    // Generate points
    cv::Point p1, p2;
    cv::Point c(randPoint(grid));

    // Generate other 2 random points within boundaries to the center
    // and check for duplicates and valid coordinates
    do {
        p1 = cv::Point(c + randChildPoint(-distance, distance));
        p2 = cv::Point(c + randChildPoint(-distance, distance));

        // Check for negative values (simply multiply by -1 to get positive)
        if (p1.x < 0) p1.x *= -1;
        if (p1.y < 0) p1.y *= -1;
        if (p2.x < 0) p2.x *= -1;
        if (p2.y < 0) p2.y *= -1;
    } while (
        p1 == p2
        || p1.x >= grid.width
        || p2.x >= grid.width
        || p1.y >= grid.height
        || p2.y >= grid.height
    );

    return Triplet(c, p1, p2);
}

TripletParams Triplet::getParams(const int width, const int height, const cv::Size &grid, const int sOffsetX, const int sOffsetY) {
    // Calculate offsets and steps for relative grid
    float stepX = width / static_cast<float>(grid.width);
    float stepY = height / static_cast<float>(grid.height);
    float offsetX = stepX / 2.0f;
    float offsetY = stepY / 2.0f;

    return TripletParams(offsetX, offsetY, stepX, stepY, sOffsetX, sOffsetY);
}

cv::Point Triplet::getPoint(int x, int y, const TripletParams &params) {
    return cv::Point(
        static_cast<int>(params.sOffsetX + params.offsetX + (x * params.stepX)),
        static_cast<int>(params.sOffsetY + params.offsetY + (y * params.stepY))
    );
}

cv::Point Triplet::getPoint(int index, const TripletParams &params) {
    int x = 0, y = 0;

    switch (index) {
        case 0:
            x = c.x;
            y = c.y;
            break;

        case 1:
            x = p1.x;
            y = p1.y;
            break;

        case 2:
            x = p2.x;
            y = p2.y;
            break;

        default:
            break;
    }

    return getPoint(x, y, params);
}

cv::Point Triplet::getCenter(const TripletParams &params) {
    return getPoint(0, params);
}

cv::Point Triplet::getP1(const TripletParams &params) {
    return getPoint(1, params);
}

cv::Point Triplet::getP2(const TripletParams &params) {
    return getPoint(2, params);
}

void Triplet::visualize(const cv::Mat &src, const cv::Size &grid, bool showGrid) {
    // Checks
    assert(!src.empty());
    assert(src.rows >= grid.height);
    assert(src.cols >= grid.width);
    assert(src.type() == 21); // CV_32FC3

    // Get TripletCoord params
    TripletParams params = getParams(src.cols, src.rows, grid);

    // Generate grid
    if (showGrid) {
        for (int y = 0; y < grid.height; ++y) {
            for (int x = 0; x < grid.width; ++x) {
                cv::circle(src, getPoint(x, y, params), 1, cv::Scalar(1, 1, 1), -1);
            }
        }
    }

    // Draw triplets
    cv::line(src, getCenter(params), getP1(params), cv::Scalar(0, 0.5f, 0));
    cv::line(src, getCenter(params), getP2(params), cv::Scalar(0, 0.5f, 0));
    cv::circle(src, getCenter(params), 3, cv::Scalar(0, 0, 1), -1);
    cv::circle(src, getP1(params), 2, cv::Scalar(0, 1, 0), -1);
    cv::circle(src, getP2(params), 2, cv::Scalar(0, 1, 0), -1);
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