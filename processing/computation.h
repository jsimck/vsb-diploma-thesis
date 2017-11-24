#ifndef VSB_SEMESTRAL_PROJECT_COMPUTATION_H
#define VSB_SEMESTRAL_PROJECT_COMPUTATION_H

#include <vector>
#include <algorithm>

namespace tless {
    float rad(float deg);
    float deg(float rad);

    template<typename T>
    T sqr(T x) {
        return x * x;
    }

    template<typename T>
    T median(std::vector<T> &values) {
        T median;

        // Sort values
        std::sort(values.begin(), values.end());
        const size_t size = values.size();

        if (size % 2 == 0) {
            median = (values[size / 2 - 1] + values[size / 2]) / 2;
        } else {
            median = values[size / 2];
        }

        return median;
    }
}

#endif
