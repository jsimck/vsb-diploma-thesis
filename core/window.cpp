#include "window.h"

namespace tless {
    cv::Point Window::tl() {
        return cv::Point(x, y);
    }

    cv::Point Window::tr() {
        return cv::Point(x + width, y);
    }

    cv::Point Window::bl() {
        return cv::Point(x, y + height);
    }

    cv::Point Window::br() {
        return cv::Point(x + width, y + height);
    }

    cv::Size Window::size() {
        return cv::Size(width, height);
    }

    bool Window::hasCandidates() {
        return !candidates.empty();
    }

    std::ostream &operator<<(std::ostream &os, const Window &w) {
        os << "[" << w.width << "," << w.height << "]" << " at" << "(" << w.x << "," << w.y << ")" << " candidates["
           << w.candidates.size() << "](";
        for (const auto &c : w.candidates) {
            os << c->id << ", ";
        }
        os << ")";
        return os;
    }

    bool Window::operator<(const Window &rhs) const {
        return candidates.size() < rhs.candidates.size();
    }

    bool Window::operator>(const Window &rhs) const {
        return rhs < *this;
    }

    bool Window::operator<=(const Window &rhs) const {
        return !(rhs < *this);
    }

    bool Window::operator>=(const Window &rhs) const {
        return !(*this < rhs);
    }

    bool Window::operator==(const Window &rhs) const {
        return candidates.size() == rhs.candidates.size();
    }

    bool Window::operator!=(const Window &rhs) const {
        return !(rhs == *this);
    }
}