#ifndef VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
#define VSB_SEMESTRAL_PROJECT_OBJECTNESS_H

#include <string>
#include <memory>
#include <utility>
#include "../core/template.h"
#include "../core/window.h"
#include "../core/classifier_criteria.h"

namespace tless {
    /**
     * @brief Class providing objectness detection over depth images
     */
    class Objectness {
    private:
        cv::Ptr<ClassifierCriteria> criteria;

    public:
        Objectness(cv::Ptr<ClassifierCriteria> criteria) : criteria(criteria) {}

        /**
         * @brief Applies simple objectness detection on source depth image based on depth discontinuities
         *
         * Depth discontinuities are areas where pixel arise on the edges of objects. Sliding window
         * is used to slide through the scene (using size of a smallest template in dataset) and calculating
         * amount of depth pixels in the scene. Window is classified as containing object if it contains
         * at least 30% (criteria->objectnessFactor) of edgels of the template containing least amount
         * of them (criteria->info.minEdgels), extracted during training phase
         *
         * @param[in]  src     Source 16-bit depth image (in mm)
         * @param[out] windows Contains all window positions, that were detected as containing object
         * @param[in]  scale   Optional scale parameter, to scale depth values according to current level of image scale pyramid
         */
        void objectness(cv::Mat &src, std::vector<Window> &windows, float scale = 1.0f);
    };
}

#endif
