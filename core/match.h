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
        cv::Rect objBB, normObjBB; //!< Bounding box normalized into 1.0f scale
        float scale = 1.0f;
        float score = 0; //!< score * object.area()

        Match() = default;
        Match(Template *t, cv::Rect bb, float scale, float score)
                : t(t), objBB(bb), normObjBB(scaledBB(bb, scale)), scale(scale), score(score) {}

        /**
         * @brief Scales the object bounding box at matched scale to the required new scale provided in the param.
         *
         * @param[in] rect     Bounding box we want to normalize by given scale
         * @param[in] scale    Current scale of the obj bounding box
         * @param[in] newScale New scale we want to normalize bounding box into
         * @return             Scaled objBB to fit scene at wanted scale
         */
        cv::Rect scaledBB(const cv::Rect &rect, float scale, float newScale = 1.0f);

        /**
         * @brief Returns overlap [0-1] between two matches, normObjBB is used for overlap calculation
         *
         * @param[in] m Second match we want to perform overlap checking on
         * @return      Amount of overlap [0-1] between two matches
         */
        float overlap(const Match &m);

        bool operator<(const Match &rhs) const;
        bool operator>(const Match &rhs) const;
        bool operator<=(const Match &rhs) const;
        bool operator>=(const Match &rhs) const;
        friend std::ostream &operator<<(std::ostream &os, const Match &match);
        friend cv::FileStorage &operator<<(cv::FileStorage &fs, const Match &match);
    };
}

#endif
