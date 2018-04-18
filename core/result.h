#ifndef VSB_SEMESTRAL_PROJECT_RESULT_H
#define VSB_SEMESTRAL_PROJECT_RESULT_H

#include <opencv2/core/types.hpp>
#include <opencv2/core/persistence.hpp>
#include "match.h"

namespace tless {
    struct Result {
    public:
        int objId;
        cv::Rect objBB;
        bool validated = false;

        Result() = default;
        Result(const Match &m);

        float jaccard(const cv::Rect &r1) const;

        friend void operator>>(const cv::FileNode &node, Result &r);
        friend cv::FileStorage &operator<<(cv::FileStorage &fs, const Result &r);
        friend std::ostream &operator<<(std::ostream &os, const Result &r);
    };
}

#endif
