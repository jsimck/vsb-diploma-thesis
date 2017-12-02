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

    cv::Rect Window::rect() {
        return cv::Rect(x, y, width, height);
    }

    bool Window::hasCandidates() {
        return !candidates.empty();
    }

    // TODO - verify that it works correctly, e.g. doen't override better candidates
    void Window::pushUnique(Template *t, int N, int minVotes) {
        if (t->votes < minVotes) return;

        // Check if candidate list is not full
        const size_t cSize = candidates.size();
        if (cSize >= N) {
            size_t minI = 0;
            int minVotes = N + 1;

            for (size_t i = 0; i < cSize; i++) {
                if (candidates[i] == t) return; // Check for duplicates
                if (candidates[i]->votes < minVotes) {
                    minVotes = candidates[i]->votes;
                    minI = i;
                }
            }

            // Replace template with least amount of votes
            candidates[minI] = t;
        } else {
            if (hasCandidates()) {
                for (const auto &candidate : candidates) {
                    if (candidate == t) return;  // Check for duplicates
                }
            }

            candidates.emplace_back(t);
        }
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
}