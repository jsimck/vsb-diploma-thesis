#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H

#include <ostream>
#include "template.h"

struct TemplateMatch {
public:
    cv::Rect bb;
    Template *t;
    float score;

    // Constructors
    TemplateMatch(cv::Rect bb, Template *t, float score = 0) : bb(bb), t(t), score(score) {}

    // Friends
    bool operator==(const TemplateMatch &rhs) const;
    bool operator!=(const TemplateMatch &rhs) const;
    bool operator<(const TemplateMatch &rhs) const;
    bool operator>(const TemplateMatch &rhs) const;
    bool operator<=(const TemplateMatch &rhs) const;
    bool operator>=(const TemplateMatch &rhs) const;
    friend std::ostream &operator<<(std::ostream &os, const TemplateMatch &match);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCH_H
