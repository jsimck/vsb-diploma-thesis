#ifndef VSB_SEMESTRAL_PROJECT_RESULT_H
#define VSB_SEMESTRAL_PROJECT_RESULT_H

#include <opencv2/core/types.hpp>
#include <opencv2/core/persistence.hpp>
#include "match.h"

namespace tless {
    /**
     * @brief Utility structure which is used in evaluation of detection results.
     */
    struct Result {
    public:
        int id, objId;
        float scale, score;
        Camera camera;
        cv::Rect objBB;
        bool validated = false;

        Result() = default;
        Result(const Match &m);

        /**
         * @brief Computes jaccard index for given rect with objBB (intersect over union)
         *
         * @param[in] r1 Rectangle to compare objBB with
         * @return       Amount of jaccard index (how much the rects are overlapping)
         */
        float jaccard(const cv::Rect &r1) const;

        friend void operator>>(const cv::FileNode &node, Result &r);
        friend cv::FileStorage &operator<<(cv::FileStorage &fs, const Result &r);
        friend std::ostream &operator<<(std::ostream &os, const Result &r);
    };
}

#endif
