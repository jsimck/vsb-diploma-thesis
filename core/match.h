#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H

#include <ostream>
#include "template.h"

/**
 * struct Match
 *
 * Holds template that passed all tests in template matching, stores object bounding box
 * relative to the current scene and score computed from template matching tests.
 */
struct Match {
public:
    const Template *tpl;
    cv::Rect objBB;
    float score, areaScore;

    // Constructors
    Match(const Template *tpl, cv::Rect &bb, float score, float areaScore) : tpl(tpl), objBB(bb), score(score), areaScore(areaScore) {}

    // Friends
    bool operator==(const Match &rhs) const;
    bool operator!=(const Match &rhs) const;
    bool operator<(const Match &rhs) const;
    bool operator>(const Match &rhs) const;
    bool operator<=(const Match &rhs) const;
    bool operator>=(const Match &rhs) const;
    friend std::ostream &operator<<(std::ostream &os, const Match &match);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H
