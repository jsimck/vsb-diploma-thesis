#ifndef VSB_SEMESTRAL_PROJECT_TRIPLET_H
#define VSB_SEMESTRAL_PROJECT_TRIPLET_H

#include <opencv2/core/types.hpp>
#include <ostream>
#include <random>

namespace tless {
    /**
     * @brief Represents set of 3 points in absolute coordinates
     *
     * Relative points coordinates are calculated inside of defined grid of (criteria.tripletGrid)
     * these coordinates are then converted to absolute values (real x,y) based on criteria.info.largestTemplate
     */
    class Triplet {
    private:
        /**
         * @brief Generates random points in normal distribution (u = 0, dev = grid.width / 2)
         *
         * Generates positions in normal distributions, this allows us to generate more positions in top
         * left corner where most templates are placed to further improve the quantization of template values
         *
         * @param[in] grid Size of the grid, in which points are generated
         * @return         Random point generated inside the grid rectangle
         */
        static cv::Point randPoint(cv::Size grid);

    public:
        cv::Point c, p1, p2;

        /**
         * @brief Generate random point in given grid, inside window sized rect
         *
         * Generates triplets with positions generated on the given grid in normal distributions.
         * This is to further improve detection, since more points are in top left corner, where most
         * templates contain valid features. Relative coordinates of these points (in grid-space) are then
         * converted to absolute values (in window-space), so no further conversion is needed.
         *
         * @param[in] grid   Relative grid size to generate points in
         * @param[in] window Size of the window to generate absolute coordinates when relative grid is placed over this window
         * @return           Triplet with randomly generated points on given grid (in absolute values)
         */
        static Triplet create(cv::Size grid, cv::Size window);

        Triplet() = default;
        Triplet(cv::Point &c, cv::Point &p1, cv::Point &p2) : c(c), p1(p1), p2(p2) {}

        friend std::ostream &operator<<(std::ostream &os, const Triplet &triplet);
        bool operator==(const Triplet &rhs) const;
        bool operator!=(const Triplet &rhs) const;
    };
}

#endif
