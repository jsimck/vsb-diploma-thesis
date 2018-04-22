#ifndef VSB_SEMESTRAL_PROJECT_WINDOW_H
#define VSB_SEMESTRAL_PROJECT_WINDOW_H

#include <opencv2/core/types.hpp>
#include "template.h"

namespace tless {
    /**
     * @brief Contains location of windows that passed objectness detection.
     */
    class Window {
    public:
        int x = 0, y = 0;
        int width = 0, height = 0;
        int edgels = 0; //!< Number of edgels this window contain (detected in objectness detection)
        std::vector<Template *> candidates;
#ifdef VIZ_HASHING
        std::vector<int> votes;
        std::vector<std::vector<Triplet>> triplets;
#endif

        Window() = default;
        Window(int x, int y, int width, int height, int edgels)
                : x(x), y(y), width(width), height(height), edgels(edgels) {}
        Window(cv::Rect rect, int edgels)
                : x(rect.tl().x), y(rect.tl().y), width(rect.width), height(rect.height), edgels(edgels) {}

        cv::Point tl();
        cv::Point tr();
        cv::Point bl();
        cv::Point br();
        cv::Rect rect();

        /**
         * @brief Returns true whether there are any templates (candidates) in candidates array.
         *
         * @return True if candidates array is not empty
         */
        bool hasCandidates();

        bool operator<(const Window &rhs) const;
        bool operator>(const Window &rhs) const;
        bool operator<=(const Window &rhs) const;
        bool operator>=(const Window &rhs) const;
        friend std::ostream &operator<<(std::ostream &os, const Window &w);
    };
}

#endif
