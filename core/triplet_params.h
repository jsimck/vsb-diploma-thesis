#ifndef VSB_SEMESTRAL_PROJECT_TRIPLET_PARAMS_H
#define VSB_SEMESTRAL_PROJECT_TRIPLET_PARAMS_H

#include <opencv2/core/types.hpp>

// TOdo refactor tripletParams + triplet
namespace tless {
    /**
     * struct TripletParams
     *
     * Used to pass triplet parameters across triplet helper functions, to minimize amount of
     * parameters needed in each function. All steps are relative to the current template and
     * reference points grid, that's why we use real numbers to avoid rounding issues.
     */
    class TripletParams {
    public:
        float offsetX;
        float offsetY;
        float stepX;
        float stepY;
        int sOffsetX; // Scene offset X
        int sOffsetY; // Scene offset Y

        // Constructors
        TripletParams(float offsetX, float offsetY, float stepX, float stepY, int sOffsetX = 0, int sOffsetY = 0) :
                offsetX(offsetX), offsetY(offsetY), stepX(stepX), stepY(stepY), sOffsetX(sOffsetX), sOffsetY(sOffsetY) {}
        TripletParams(int width, int height, const cv::Size &grid, int sOffsetX = 0, int sOffsetY = 0);
    };
}

#endif
