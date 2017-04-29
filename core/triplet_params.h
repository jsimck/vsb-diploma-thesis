#ifndef VSB_SEMESTRAL_PROJECT_TRIPLET_PARAMS_H
#define VSB_SEMESTRAL_PROJECT_TRIPLET_PARAMS_H

#include <opencv2/core/types.hpp>

/**
 * struct TripletParams
 *
 * Used to pass triplet parameters across triplet helper functions, to minimize amount of
 * parameters needed in each function. All steps are relative to the current template and
 * reference points grid, that's why we use real numbers to avoid rounding issues.
 */
struct TripletParams {
public:
    float offsetX;
    float offsetY;
    float stepX;
    float stepY;
    int sOffsetX; // Scene offset X
    int sOffsetY; // Scene offset Y

    // Constructors
    TripletParams(const float offsetX, const float offsetY, const float stepX, const float stepY, const int sOffsetX = 0, const int sOffsetY = 0) :
        offsetX(offsetX), offsetY(offsetY), stepX(stepX), stepY(stepY), sOffsetX(sOffsetX), sOffsetY(sOffsetY) {}
    TripletParams(const int width, const int height, const cv::Size &grid, const int sOffsetX = 0, const int sOffsetY = 0);
};

#endif //VSB_SEMESTRAL_PROJECT_TRIPLET_PARAMS_H
