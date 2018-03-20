#include <functional>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv/cv.hpp>
#include "triplet.h"

namespace tless {
    cv::Point Triplet::randPoint(cv::Size grid) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dX(0, grid.width - 1);
        static std::uniform_int_distribution<> dY(0, grid.height - 1);

        int x, y;
        do {
            x = static_cast<int>(std::abs(dX(gen)));
            y = static_cast<int>(std::abs(dY(gen)));
        } while (x >= grid.width || y >= grid.height);

        return {x, y};
    }

    Triplet Triplet::create(cv::Size grid, cv::Size window) {
        assert(grid.area() > 0);
        assert(window.area() > 0);

        // Generate unique points in relative coordinates (grid-space)
        cv::Point c, p1, p2;

        do {
            c = randPoint(grid);
            p1= randPoint(grid);
            p2 = randPoint(grid);
        } while (!(c != p1 && p1 != p2 && p2 != c));

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