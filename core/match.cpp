#include "match.h"

namespace tless {
    bool Match::operator<(const Match &rhs) const {
        return areaScore < rhs.areaScore;
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
        os << "id: " << match.t->id
           << " objBB: " << match.objBB
           << " scale: " << match.scale
           << " t: " << match.t
           << " score: " << match.score
           << " areaScore: " << match.areaScore
           << " sI: " << match.sI
           << " sII: " << match.sII
           << " sIII: " << match.sIII
           << " sIV: " << match.sIV
           << " sV: " << match.sV;

        return os;
    }

    cv::Rect Match::scaledBB(float scale) {
        float multiplier = scale / this->scale;

        return cv::Rect(
            static_cast<int>(this->objBB.x * multiplier),
            static_cast<int>(this->objBB.y * multiplier),
            static_cast<int>(this->objBB.width * multiplier),
            static_cast<int>(this->objBB.height * multiplier)
        );
    }
}