#include "template_match.h"

bool TemplateMatch::operator==(const TemplateMatch &rhs) const {
    return score == rhs.score && t->id == rhs.t->id;
}

bool TemplateMatch::operator!=(const TemplateMatch &rhs) const {
    return !(rhs == *this);
}

bool TemplateMatch::operator<(const TemplateMatch &rhs) const {
    return score < rhs.score;
}

bool TemplateMatch::operator>(const TemplateMatch &rhs) const {
    return rhs < *this;
}

bool TemplateMatch::operator<=(const TemplateMatch &rhs) const {
    return !(rhs < *this);
}

bool TemplateMatch::operator>=(const TemplateMatch &rhs) const {
    return !(*this < rhs);
}

std::ostream &operator<<(std::ostream &os, const TemplateMatch &match) {
    os << "tl: " << match.tl << " t: " << match.t << " score: " << match.score;
    return os;
}
