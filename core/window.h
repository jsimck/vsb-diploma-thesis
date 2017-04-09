#ifndef VSB_SEMESTRAL_PROJECT_WINDOW_H
#define VSB_SEMESTRAL_PROJECT_WINDOW_H

#include <opencv2/core/types.hpp>
#include "template.h"

struct Window {
public:
    int x;
    int y;
    int width;
    int height;
    unsigned int edgels;
    std::vector<Template *> candidates;

    // Constructors
    Window(int x = 0, int y = 0, int width = 0, int height = 0, unsigned int edgels = 0, std::vector<Template *> candidates = {}) : x(x), y(y), width(width), height(height), candidates(candidates), edgels(edgels) {}

    // Methods
    cv::Point tl();
    cv::Point tr();
    cv::Point bl();
    cv::Point br();
    cv::Size size();
    bool hasCandidates();
    void pushUnique(Template *t, unsigned int N = 100, int v = 3);
    unsigned long candidatesSize();

    // Friends
    friend std::ostream &operator<<(std::ostream &os, const Window &w);
};

#endif //VSB_SEMESTRAL_PROJECT_WINDOW_H
