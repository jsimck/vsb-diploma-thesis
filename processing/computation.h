#ifndef VSB_SEMESTRAL_PROJECT_COMPUTATION_H
#define VSB_SEMESTRAL_PROJECT_COMPUTATION_H

#include <vector>
#include <algorithm>

namespace tless {
    /**
     * @brief Converts angle from degrees to radians.
     *
     * @tparam    T   Angle data type (decimal number)
     * @param[in] deg Angle in degrees
     * @return        Angle in radians
     */
    template <typename T>
    T rad(T deg) {
        return static_cast<T>(deg * (M_PI / 180.0f));
    }

    /**
     * @brief Converts angle from radians to degrees.
     *
     * @tparam    T   Angle data type (decimal number)
     * @param[in] rad Angle in radians
     * @return        Angle in degrees
     */
    template <typename T>
    T deg(T rad) {
        return static_cast<T>(rad * (180.0f / M_PI));
    }

    /**
     * @brief Computes power of 2 on value x.
     *
     * @tparam    T X value data type
     * @param[in] x Value to apply power of 2 on
     * @return      x * x
     */
    template<typename T>
    T sqr(T x) {
        return x * x;
    }

    /**
     * @brief Returns median value of input array.
     *
     * @tparam    T      Values data type
     * @param[in] values Array of input values
     * @return           Median of input array
     */
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

    /**
     * @brief Removes elements at indexes to_remove from the input array.
     *
     * @tparam    T        Elements data type
     * @param[in] elements Array of input elements
     * @param[in] toRemove Array of indices identifying elements to remove
     */
    template<typename T>
    void removeIndex(std::vector<T> &elements, const std::vector<size_t> &toRemove) {
        auto vector_base = elements.begin();
        auto down_by = 0;

        for (auto iter = toRemove.cbegin(); iter < toRemove.cend(); iter++, down_by++) {
            auto next = (iter + 1 == toRemove.cend() ? elements.size() : *(iter + 1));
            std::move(vector_base + *iter + 1, vector_base + next, vector_base + *iter - down_by);
        }

        elements.resize(elements.size() - toRemove.size());
    }

    /**
     * @brief Calculates mean and sd of numerical array.
     *
     * @tparam    T      Elements data type
     * @param[in] values Array of numerical values
     * @param[in] mean   Values mean
     * @param[in] sd     Values standard deviation
     */
    template<typename T>
    void standardDeviation(const std::vector<T> &values, T &mean, T &sd) {
        // Calculate mean
        mean = 0;
        for (auto &val : values) {
            mean += val;
        }
        mean = mean / static_cast<float>(values.size());

        // Calculate sd
        T sum = 0;
        for (auto &val : values) {
            sum += sqr<T>(val - mean);
        }

        sd = std::sqrt((1 / (static_cast<float>(values.size()) - 1)) * sum);
    }

    /**
     * @brief Removes outliers from numerical array using three sigma rule.
     *
     * @tparam    T         Elements data type
     * @param[in] sigmaRule Number [1-3] defining three sigma rule
     * @param[in] values    Array of numerical values
     */
    template<typename T>
    void removeOutliers(std::vector<T> &values, int sigmaRule = 3) {
        // Find mean and sd
        T mean, sd;
        standardDeviation<T>(values, mean, sd);

        // Find outliers
        std::vector<size_t> toRemove;
        T negThreeSigma = mean - sigmaRule * sd;
        T posThreeSigma = mean + sigmaRule * sd;

        for (int i = 0; i < values.size(); ++i) {
            if (values[i] < negThreeSigma || values[i] > posThreeSigma) {
                toRemove.push_back(i);
            }
        }

        // Remove outliers
        removeIndex<T>(values, toRemove);
    }
}

#endif
