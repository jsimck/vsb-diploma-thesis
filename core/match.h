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
        // TODO - area score is in this case probably area in pixels covered by the template, not the object bounding box
        float areaScore; //!< score * object.area()
        int sI, sII, sIII, sIV, sV;

        Match() = default;
        Match(Template *t, cv::Rect &bb, float score, float areaScore, int sI, int sII, int sIII, int sIV, int sV)
                : t(t), objBB(bb), score(score), areaScore(areaScore), sI(sI), sII(sII), sIII(sIII), sIV(sIV), sV(sV) {}

        bool operator<(const Match &rhs) const;
        bool operator>(const Match &rhs) const;
        bool operator<=(const Match &rhs) const;
        bool operator>=(const Match &rhs) const;
        friend std::ostream &operator<<(std::ostream &os, const Match &match);
    };
}

#endif
