#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H

#include <ostream>
#include "template.h"

/**
 * struct Match
 *
 * Holds template that passed all tests in template matching with object bounding box
 * relative to the current scene and score computed from template matching tests
 */
struct Match {
public:
    cv::Rect objBB;
    Template *t;
    float score;

    // Constructors
    Match(cv::Rect bb, Template *t, float score = 0) : objBB(bb), t(t), score(score) {}

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
