#include <functional>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv/cv.hpp>
#include "triplet.h"

namespace tless {
    std::random_device Triplet::seed;
    //std::mt19937 Triplet::rng(Triplet::seed());
    std::mt19937 Triplet::rng(1);

    float Triplet::random(const float min, const float max) {
        float rnd;
        std::uniform_real_distribution<float> randomizer(0.0f, 1.0f);

        #pragma omp critical (random)
        {
            rnd = randomizer(Triplet::rng);
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
        auto y = static_cast<int>(random(min, max));
        auto x = static_cast<int>(random(min, max));

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

        return {c, p1, p2};
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

    std::ostream &operator<<(std::ostream &os, const Triplet &triplet) {
        os << "c(" << triplet.c.x << ", " << triplet.c.y << "), ";
        os << "p1(" << triplet.p1.x << ", " << triplet.p1.y << "), ";
        os << "p2(" << triplet.p2.x << ", " << triplet.p2.y << ") ";
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
}