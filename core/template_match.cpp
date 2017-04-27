#include "template_match.h"

bool TemplateMatch::operator==(const TemplateMatch &rhs) const {
    return t->id == rhs.t->id;
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
    os << "bb: " << match.bb << " t: " << match.t << " score: " << match.score;
    return os;
}
