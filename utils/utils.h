#ifndef VSB_SEMESTRAL_PROJECT_UTILS_H
#define VSB_SEMESTRAL_PROJECT_UTILS_H

#include <string>
#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv/cv.h>

#define SQR(x) ((x) * (x))

class Utils {
public:
    // General
    // ---------------------------------------
    // | Mat type |  C1 |  C2  |  C3  |  C4  |
    // ---------------------------------------
    // | CV_8U    |  0  |  8   |  16  |  24  |
    // | CV_8S    |  1  |  9   |  17  |  25  |
    // | CV_16U   |  2  |  10  |  18  |  26  |
    // | CV_16S   |  3  |  11  |  19  |  27  |
    // | CV_32S   |  4  |  12  |  20  |  28  |
    // | CV_32F   |  5  |  13  |  21  |  29  |
    // | CV_64F   |  6  |  14  |  22  |  30  |
    // ---------------------------------------
    static std::string matType2Str(int type);

    template<typename T>
    static void removeIndex(std::vector<T> &vector, const std::vector<size_t> &to_remove) {
        auto vector_base = vector.begin();
        auto down_by = 0;

        for (auto iter = to_remove.cbegin(); iter < to_remove.cend(); iter++, down_by++) {
            auto next = (iter + 1 == to_remove.cend() ? vector.size() : *(iter + 1));
            std::move(vector_base + *iter + 1, vector_base + next, vector_base + *iter - down_by);
        }

        vector.resize(vector.size() - to_remove.size());
    }

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

#endif //VSB_SEMESTRAL_PROJECT_UTILS_H
