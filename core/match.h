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
    float score;
    cv::Rect objBB;
    const Template *t;

    // Constructors
    Match(const Template *t, const cv::Rect bb, const float score) : objBB(bb), t(t), score(score) {}

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
