#ifndef VSB_SEMESTRAL_PROJECT_HASHING_H
#define VSB_SEMESTRAL_PROJECT_HASHING_H

#include <opencv2/opencv.hpp>
#include <memory>
#include "../core/hash_table.h"
#include "../core/classifier_criteria.h"
#include "../core/window.h"

namespace tless {
    /**
     * @brief Class used to train HashTables and quickly verify what templates should be matched per each window passed form objectness detection
     */
    class Hasher {
    private:
        cv::Ptr<ClassifierCriteria> criteria;

        /**
         * @brief Computes bin ranges for each table (triplet) across all templates based on relative depths
         *
         * @param[in]     templates Input array of all templates parsed for detection
         * @param[in,out] tables    Input array of tables, which are then updated with their computed bin range
         */
        void initializeBinRanges(std::vector<Template> &templates, std::vector<HashTable> &tables);

    public:
        // Statics
        static const int IMG_16BIT_MAX = 65535;

        // Constructors
        Hasher(cv::Ptr<ClassifierCriteria> criteria) : criteria(criteria) {}

        // Methods
        void verifyCandidates(const cv::Mat &sceneDepth, const cv::Mat &sceneSurfaceNormalsQuantized,
                              std::vector<HashTable> &tables, std::vector<Window> &windows);


        void train(std::vector<Template> &templates, std::vector<HashTable> &tables);
    };
}

#endif
