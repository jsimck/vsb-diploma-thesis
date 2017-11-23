#ifndef VSB_SEMESTRAL_PROJECT_UTILS_H
#define VSB_SEMESTRAL_PROJECT_UTILS_H

#include <string>
#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv/cv.h>

#define SQR(x) ((x) * (x))

class Utils {
public:
    template<typename T>
    static float median(std::vector<T> &values) {
        float median;
        const size_t size = values.size();

        // Sort values
        std::sort(values.begin(), values.end());

        if (size  % 2 == 0) {
            median = (values[size / 2 - 1] + values[size / 2]) / 2;
        } else {
            median = values[size / 2];
        }

        return median;
    }
};

#endif
