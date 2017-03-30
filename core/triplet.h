#ifndef VSB_SEMESTRAL_PROJECT_TRIPLET_H
#define VSB_SEMESTRAL_PROJECT_TRIPLET_H

#include <opencv2/core/types.hpp>
#include <ostream>

/**
 * struct TripletCoords
 *
 * Used to pass triplet parameters across triplet helper functions, to minimize amount of
 * parameters in each function
 */
struct TripletCoords {
public:
    float offsetX;
    float stepX;
    float offsetY;
    float stepY;
    int sceneOffsetX;
    int sceneOffsetY;

    // Constructors
    TripletCoords(float offsetX, float stepX, float offsetY, float stepY, int sceneOffsetX = 0, int sceneOffsetY = 0) :
        offsetX(offsetX), stepX(stepX), offsetY(offsetY), stepY(stepY), sceneOffsetX(sceneOffsetX), sceneOffsetY(sceneOffsetY) {}
};

/**
 * struct Triplet
 *
 * All generated points are stored in relative locations, we use 12x12 reference points, so every
 * point has x and y coordinates in interval <0, 11>. This allows to adapt reference point locations
 * to each template bounding box.
 */
struct Triplet {
private:
    inline static cv::Point randomPoint(const cv::Size referencePointsGrid);
    inline static cv::Point randomPointBoundaries(int min = -4, int max = 4);
public:
    cv::Point c;
    cv::Point p1;
    cv::Point p2;

    // Statics
    static float random(const float rangeMin = 0.0f, const float rangeMax = 1.0f);
    static Triplet createRandomTriplet(const cv::Size &referencePointsGrid, int maxNeighbourhood = 3);
    static TripletCoords getCoordParams(const int width, const int height, const cv::Size &referencePointsGrid, int sceneOffsetX = 0, int sceneOffsetY = 0);

    // Constructors
    Triplet() {}
    Triplet(const cv::Point c, const cv::Point p1, const cv::Point p2) : c(c), p1(p1), p2(p2) {}

    // Methods
    cv::Point getPoint(int x, int y, float offsetX, float stepX, float offsetY, float stepY, int sceneOffsetX = 0, int sceneOffsetY = 0);
    cv::Point getPoint(int x, int y, const TripletCoords &coordinateParams);
    cv::Point getCoords(int index, float offsetX, float stepX, float offsetY, float stepY, int sceneOffsetX = 0, int sceneOffsetY = 0);
    cv::Point getCoords(int index, const TripletCoords &coordinateParams);
    cv::Point getCenterCoords(float offsetX, float stepX, float offsetY, float stepY, int sceneOffsetX = 0, int sceneOffsetY = 0);
    cv::Point getCenterCoords(const TripletCoords &coordinateParams);
    cv::Point getP1Coords(float offsetX, float stepX, float offsetY, float stepY, int sceneOffsetX = 0, int sceneOffsetY = 0);
    cv::Point getP1Coords(const TripletCoords &coordinateParams);
    cv::Point getP2Coords(float offsetX, float stepX, float offsetY, float stepY, int sceneOffsetX = 0, int sceneOffsetY = 0);
    cv::Point getP2Coords(const TripletCoords &coordinateParams);
    void visualize(const cv::Mat &src, const cv::Size &referencePointsGrid, bool grid = true);

    // Operators
    friend std::ostream &operator<<(std::ostream &os, const Triplet &triplet);
    bool operator==(const Triplet &rhs) const;
    bool operator!=(const Triplet &rhs) const;
};

#endif //VSB_SEMESTRAL_PROJECT_TRIPLET_H
