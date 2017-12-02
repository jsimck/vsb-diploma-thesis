#ifndef VSB_SEMESTRAL_PROJECT_COMPUTATION_H
#define VSB_SEMESTRAL_PROJECT_COMPUTATION_H

#include <vector>
#include <algorithm>

namespace tless {
    template <typename T>
    T rad(T deg) {
        return static_cast<T>(deg * (M_PI / 180.0f));
    }

    template <typename T>
    T deg(T rad) {
        return static_cast<T>(rad * (180.0f / M_PI));
    }

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

    template<typename T>
    void removeIndex(std::vector<T> &vector, const std::vector<size_t> &to_remove) {
        auto vector_base = vector.begin();
        auto down_by = 0;

        for (auto iter = to_remove.cbegin(); iter < to_remove.cend(); iter++, down_by++) {
            auto next = (iter + 1 == to_remove.cend() ? vector.size() : *(iter + 1));
            std::move(vector_base + *iter + 1, vector_base + next, vector_base + *iter - down_by);
        }

        vector.resize(vector.size() - to_remove.size());
    }
}

#endif
