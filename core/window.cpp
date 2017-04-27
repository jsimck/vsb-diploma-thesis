#include "window.h"

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
    return candidates.size() > 0;
}

void Window::pushUnique(Template *t, uint N, int v) {
    if (t->votes < v) return;

    // Check if candidate list is not full
    if (candidates.size() >= N) {
        int minI = 0, minVotes = N + 1;

        for (int i = 0; i < candidates.size(); i++) {
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

        candidates.push_back(t);
    }
}

std::ostream& operator<<(std::ostream &os, const Window &w) {
    os << "[" << w.width << "," << w.height << "]" << " at" << "(" << w.x << "," << w.y << ")" << " candidates[" << w.candidates.size() << "](";
    for (const auto &c : w.candidates) {
        os << c->id << ", ";
    }
    os << ")";
    return os;
}