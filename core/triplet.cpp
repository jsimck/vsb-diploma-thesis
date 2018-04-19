#include <functional>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv/cv.hpp>
#include "triplet.h"
#include "../processing/computation.h"

namespace tless {
    cv::Point Triplet::randPoint(cv::Size grid) {
        static std::random_device rd;
        static std::mt19937 gen(1); //TODO random
        static std::uniform_int_distribution<> dX(0, grid.width - 1);
        static std::uniform_int_distribution<> dY(0, grid.height - 1);

        return {dX(gen), dY(gen)};
    }

    Triplet Triplet::create(cv::Size grid, cv::Size window) {
        assert(grid.area() > 0);
        assert(window.area() > 0);

        float p1Dist, p2Dist, angle;
        cv::Point c, p1, p2, p1Diff, p2Diff;
        cv::Point2f vP1, vP2;
        const float minDistance = grid.width * 0.1f;
        const float maxDistance = grid.width * 0.6f;
        const auto minAngle = rad<float>(30);

        // Generate unique points in relative coordinates (grid-space)
        do {
            c = randPoint(grid);
            p1= randPoint(grid);
            p2 = randPoint(grid);
            p1Diff = p1 - c;
            p2Diff = p2 - c;

            // Compute distance
            p1Dist = std::sqrtf(sqr<int>(p1Diff.x) + sqr<int>(p1Diff.y));
            p2Dist = std::sqrtf(sqr<int>(p2Diff.x) + sqr<int>(p2Diff.y));

            // Normalize points
            vP1 = static_cast<cv::Point2f>(p1Diff) / p1Dist;
            vP2 = static_cast<cv::Point2f>(p2Diff) / p2Dist;
            angle = std::acos(vP1.dot(vP2));
        } while (
            !(c != p1 && p1 != p2 && p2 != c) || // Equality ceck
            !(p1Dist >= minDistance && p1Dist <= maxDistance) || // min distance check
            !(p2Dist >= minDistance && p2Dist <= maxDistance) || // max distance check
            angle < minAngle // angle check
        );

        // Generate absolute offsets and steps
        auto stepX = window.width / static_cast<float>(grid.width);
        auto stepY = window.height / static_cast<float>(grid.height);
        auto offsetX = stepX * 0.5f;
        auto offsetY = stepY * 0.5f;

        // Convert to absolute coordinates (window-space)
        c.x = static_cast<int>(stepX * c.x + offsetX);
        c.y = static_cast<int>(stepY * c.y + offsetY);
        p1.x = static_cast<int>(stepX * p1.x + offsetX);
        p1.y = static_cast<int>(stepY * p1.y + offsetY);
        p2.x = static_cast<int>(stepX * p2.x + offsetX);
        p2.y = static_cast<int>(stepY * p2.y + offsetY);

        return {c, p1, p2};
    }

    std::ostream &operator<<(std::ostream &os, const Triplet &triplet) {
        os << "c(" << triplet.c.x << ", " << triplet.c.y << "), ";
        os << "p1(" << triplet.p1.x << ", " << triplet.p1.y << "), ";
        os << "p2(" << triplet.p2.x << ", " << triplet.p2.y << ") ";
        return os;
    }

    bool Triplet::operator==(const Triplet &rhs) const {
        return (c == rhs.c || c == rhs.p1 || c == rhs.p2) &&
               (p1 == rhs.c || p1 == rhs.p1 || p1 == rhs.p2) &&
               (p2 == rhs.c || p2 == rhs.p1 || p2 == rhs.p2);
    }

    bool Triplet::operator!=(const Triplet &rhs) const {
        return !(rhs == *this);
    }
}