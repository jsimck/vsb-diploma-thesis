#include "triplet_params.h"

namespace tless {
    TripletParams::TripletParams(const int width, const int height, const cv::Size &grid, const int sOffsetX, const int sOffsetY) {
        // Calculate offsets and steps for relative grid
        stepX = width / static_cast<float>(grid.width);
        stepY = height / static_cast<float>(grid.height);
        offsetX = stepX / 2.0f;
        offsetY = stepY / 2.0f;
        this->sOffsetX = sOffsetX;
        this->sOffsetY = sOffsetY;
    }
}