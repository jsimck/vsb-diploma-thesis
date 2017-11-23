#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H

#include <ostream>
#include <memory>
#include <utility>
#include "template.h"

/**
 * @brief Holds matched templates along with their location and final score
 */
class Match {
public:
    std::shared_ptr<Template> tpl;
    cv::Rect objBB;
    float score, areaScore;

    // Constructors
    Match(std::shared_ptr<Template> tpl, cv::Rect &bb, float score, float areaScore) : tpl(tpl), objBB(bb), score(score), areaScore(areaScore) {}

    // Friends
    bool operator==(const Match &rhs) const;
    bool operator!=(const Match &rhs) const;
    bool operator<(const Match &rhs) const;
    bool operator>(const Match &rhs) const;
    bool operator<=(const Match &rhs) const;
    bool operator>=(const Match &rhs) const;
    friend std::ostream &operator<<(std::ostream &os, const Match &match);
};

#endif
