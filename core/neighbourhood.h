#ifndef VSB_SEMESTRAL_PROJECT_NEIGHBOURHOOD_H
#define VSB_SEMESTRAL_PROJECT_NEIGHBOURHOOD_H

#include <opencv2/core/hal/interface.h>

struct Neighbourhood {
public:
    int offsetX;
    int offsetY;
    int width;
    int height;

    Neighbourhood(int width, int height);
};

#endif //VSB_SEMESTRAL_PROJECT_NEIGHBOURHOOD_H
