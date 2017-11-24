#ifndef VSB_SEMESTRAL_PROJECT_TRIPLET_H
#define VSB_SEMESTRAL_PROJECT_TRIPLET_H

#include <opencv2/core/types.hpp>
#include <ostream>
#include <random>
#include "triplet_params.h"

// TODO refactor
namespace tless {
    /**
     * @brief Represents set of 3 points in relative coordinates, to compute hash table key on.
     *
     * All generated points are stored in relative coordinates, we use 12x12 reference points, so every
     * point has x and y coordinates in interval <0, 11>. This allows to adapt reference point locations
     * to each template bounding box. Grid size can be changed through classifier criteria
     */
    class Triplet {
    private:
        static std::random_device seed;
        static std::mt19937 rng;
        static cv::Point randPoint(cv::Size grid);
        static cv::Point randChildPoint(int min = -3, int max = 3);

    public:
        cv::Point c;
        cv::Point p1;
        cv::Point p2;

        // Statics
        static float random(float min = 0.0f, float max = 1.0f);
        static Triplet create(const cv::Size &grid, int distance = 3); // max distance from center triplet

        // Constructors
        Triplet() {}
        Triplet(cv::Point &c, cv::Point &p1, cv::Point &p2) : c(c), p1(p1), p2(p2) {}

        // Methods
        cv::Point getPoint(int index, const TripletParams &params);
        cv::Point getPoint(int x, int y, const TripletParams &params);
        cv::Point getCenter(const TripletParams &params);
        cv::Point getP1(const TripletParams &params);
        cv::Point getP2(const TripletParams &params);

        // Operators
        friend std::ostream &operator<<(std::ostream &os, const Triplet &triplet);
        bool operator==(const Triplet &rhs) const;
        bool operator!=(const Triplet &rhs) const;
    };
}

#endif
