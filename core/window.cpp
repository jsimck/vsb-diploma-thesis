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

bool Window::hasCandidates() {
    return candidates.size() > 0;
}

void Window::pushUnique(Template *t, unsigned int N, int v) {
    // Check if number of minVotesPerTemplate is > than minimum
    if (t->votes < v) return;

    // Check if candidate list is not full
    if (candidates.size() >= N) {
        int minIndex = 0, minVotes = N + 1; // template can have max N minVotesPerTemplate

        for (int i = 0; i < candidates.size(); i++) {
            if (candidates[i] == t) return; // Check for duplicates
            if (candidates[i]->votes < minVotes) {
                minVotes = candidates[i]->votes;
                minIndex = i;
            }
        }

        // Replace template with least amount of minVotesPerTemplate
        candidates[minIndex] = t;
    } else {
        // Check for duplicates
        if (hasCandidates()) {
            for (auto &&candidate : candidates) {
                if (candidate == t) return;
            }
        }

        // Push candidate to list
        candidates.push_back(t);
    }
}

unsigned long Window::candidatesSize() {
    return candidates.size();
}

std::ostream& operator<<(std::ostream &os, const Window &w) {
    os << "[" << w.width << "," << w.height << "]" << " at" << "(" << w.x << "," << w.y << ")" << " candidates[" << w.candidates.size() << "](";
    for (const auto &c : w.candidates) {
        os << c->id << ", ";
    }
    os << ")";
    return os;
}

cv::Size Window::size() {
    return cv::Size(width, height);
}
