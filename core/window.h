#ifndef VSB_SEMESTRAL_PROJECT_WINDOW_H
#define VSB_SEMESTRAL_PROJECT_WINDOW_H

#include <opencv2/core/types.hpp>
#include "template.h"

/**
 * struct Window
 *
 * After objectness detection, each sliding window containing object is represented using this structure.
 * These windows are then used further down the cascade in template matching.
 */
struct Window {
public:
    int x;
    int y;
    uint width;
    uint height;
    uint edgels;
    std::vector<Template *> candidates;

    // Constructors
    Window(const int x = 0, const int y = 0, uint width = 0, uint height = 0, uint edgels = 0, std::vector<Template *> candidates = {})
        : x(x), y(y), width(width), height(height), candidates(candidates), edgels(edgels) {}

    // Methods
    cv::Point tl();
    cv::Point tr();
    cv::Point bl();
    cv::Point br();

    cv::Size size();
    bool hasCandidates();
    void pushUnique(Template *t, uint N = 100, int v = 3);

    // Friends
    friend std::ostream &operator<<(std::ostream &os, const Window &w);
};

#endif //VSB_SEMESTRAL_PROJECT_WINDOW_H
