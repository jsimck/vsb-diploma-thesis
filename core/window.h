#ifndef VSB_SEMESTRAL_PROJECT_WINDOW_H
#define VSB_SEMESTRAL_PROJECT_WINDOW_H

#include <opencv2/core/types.hpp>
#include "template.h"

struct Window {
public:
    cv::Point p;
    cv::Size size;
    unsigned int edgels;
    std::vector<Template *> candidates;

    // Constructors
    Window(cv::Point p, cv::Size size) : p(p), size(size) {}
    Window(cv::Point p, cv::Size size, unsigned int edgels) : p(p), size(size), edgels(edgels) {}
    Window(cv::Point p, cv::Size size, std::vector<Template *> candidates, unsigned int edgels) : p(p), size(size), candidates(candidates), edgels(edgels) {}

    // Methods
    cv::Point tl();
    cv::Point tr();
    cv::Point bl();
    cv::Point br();
    bool hasCandidates();
    void pushUnique(Template *t, unsigned int N = 100, int v = 3);
    unsigned long candidatesSize();

    // Friends
    friend std::ostream &operator<<(std::ostream &os, const Window &w);
};

#endif //VSB_SEMESTRAL_PROJECT_WINDOW_H
