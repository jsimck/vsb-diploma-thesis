#include "window.h"

cv::Point Window::tl() {
    return p;
}

cv::Point Window::tr() {
    return cv::Point(p.x + size.width, p.y);
}

cv::Point Window::bl() {
    return cv::Point(p.x, p.y + size.height);
}

cv::Point Window::br() {
    return cv::Point(p.x + size.width, p.y + size.height);
}

bool Window::hasCandidates() {
    return candidates.size() > 0;
}

void Window::pushUnique(Template *t) {
    for (auto &&candidate : candidates) {
        if (candidate == t) return;
    }

    candidates.push_back(t);
}

unsigned long Window::candidatesCount() {
    return candidates.size();
}

std::ostream& operator<<(std::ostream &os, const Window &w) {
    os << "Window: " << w.size << " starts at: " << w.p << " candidates: (";
    for (const auto &c : w.candidates) {
        os << c->id << ", ";
    }
    os << ")";
    return os;
}