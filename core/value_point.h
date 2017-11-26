#ifndef VSB_SEMESTRAL_PROJECT_VALUE_POINT_H
#define VSB_SEMESTRAL_PROJECT_VALUE_POINT_H

#include <ostream>
#include <opencv2/core/types.hpp>

namespace tless {
    template<typename T>
    struct ValuePoint {
    public:
        cv::Point p;
        T value;

        ValuePoint(const cv::Point &p, T value) : p(p), value(value) {}

        bool operator<(const ValuePoint &rhs) const {
            return value < rhs.value;
        }

        bool operator>(const ValuePoint &rhs) const {
            return rhs < *this;
        }

        bool operator<=(const ValuePoint &rhs) const {
            return !(rhs < *this);
        }

        bool operator>=(const ValuePoint &rhs) const {
            return !(*this < rhs);
        }

        friend std::ostream &operator<<(std::ostream &os, const ValuePoint &point) {
            os << "p: (" << point.p.x << ", " << point.p.y << ") value: " << point.value;

            return os;
        }
    };
}

#endif
