#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H

#include <ostream>
#include <memory>
#include <utility>
#include "template.h"

namespace tless {
    /**
     * @brief Holds matched templates along with their location and final score
     */
    struct Match {
    public:
        Template *t;
        cv::Rect objBB;
        float score;
        float areaScore; //!< score * object.area()

        Match() = default;
        Match(Template *t, cv::Rect &bb, float score, float areaScore)
                : t(t), objBB(bb), score(score), areaScore(areaScore) {}

        bool operator<(const Match &rhs) const;
        bool operator>(const Match &rhs) const;
        bool operator<=(const Match &rhs) const;
        bool operator>=(const Match &rhs) const;
        friend std::ostream &operator<<(std::ostream &os, const Match &match);
    };
}

#endif
