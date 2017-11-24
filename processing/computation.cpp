#include <math.h>
#include "computation.h"

namespace tless {
    float rad(float deg) {
        return static_cast<float>(deg * (M_PI / 180.0f));
    }

    float deg(float rad) {
        return static_cast<float>(rad * (180.0f / M_PI));
    }
}