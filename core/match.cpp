#include "match.h"

bool Match::operator==(const Match &rhs) const {
    return tpl->id == rhs.tpl->id;
}

bool Match::operator!=(const Match &rhs) const {
    return !(rhs == *this);
}

bool Match::operator<(const Match &rhs) const {
    return score < rhs.score;
}

bool Match::operator>(const Match &rhs) const {
    return rhs < *this;
}

bool Match::operator<=(const Match &rhs) const {
    return !(rhs < *this);
}

bool Match::operator>=(const Match &rhs) const {
    return !(*this < rhs);
}

std::ostream &operator<<(std::ostream &os, const Match &match) {
    os << "objBB: " << match.objBB << " tpl: " << match.tpl << " score: " << match.score;
    return os;
}
