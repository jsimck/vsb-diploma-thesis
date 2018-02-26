#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H

#include <ostream>
#include <memory>
#include <utility>
#include "template.h"

namespace tless {
    /**
     * @brief Holds matched templates along with their location and final score.
     */
    struct Match {
    public:
        Template *t;
        cv::Rect objBB;
        float scale = 1.0f;
        float score = 0;
        // TODO - area score is in this case probably area in pixels covered by the template, not the object bounding box
        float areaScore = 0; //!< score * object.area()
        int sI = 0, sII = 0, sIII = 0, sIV = 0, sV = 0;

        Match() = default;
        Match(Template *t, cv::Rect bb, float scale, float score, float areaScore, int sI, int sII, int sIII, int sIV, int sV)
                : t(t), objBB(bb), scale(scale), score(score), areaScore(areaScore), sI(sI), sII(sII), sIII(sIII), sIV(sIV), sV(sV) {}

        /**
         * @brief Scales the object bounding box at matched scale to the required scale provided in the param.
         *
         * @param[in] scale Current scene scale to which we want to scale objBB to
         * @return          Scaled objBB to fit scene at wanted scale
         */
        cv::Rect scaledBB(float scale);

        bool operator<(const Match &rhs) const;
        bool operator>(const Match &rhs) const;
        bool operator<=(const Match &rhs) const;
        bool operator>=(const Match &rhs) const;
        friend std::ostream &operator<<(std::ostream &os, const Match &match);
    };
}

#endif
