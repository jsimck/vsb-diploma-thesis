#ifndef VSB_SEMESTRAL_PROJECT_TRIPLET_COORDS_H
#define VSB_SEMESTRAL_PROJECT_TRIPLET_COORDS_H

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
        offsetX(offsetX), stepX(stepX), offsetY(offsetY), stepY(stepY), sOffsetX(sOffsetX), sOffsetY(sOffsetY) {}
};

#endif //VSB_SEMESTRAL_PROJECT_TRIPLET_COORDS_H
