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
    Window(int x, int y, int width, int height) : x(x), y(y), width(width), height(height) {}
    Window(int x, int y, int width, int height, unsigned int edgels) : x(x), y(y), width(width), height(height), edgels(edgels) {}
    Window(int x, int y, int width, int height, std::vector<Template *> candidates, unsigned int edgels) : x(x), y(y), width(width), height(height), candidates(candidates), edgels(edgels) {}

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
