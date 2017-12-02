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
         * @brief Applies validations to triplet points (out of objBB, wrong depth), generates normalized triplet points and relative depths
         *
         * @param[in]  triplet Generated triplet from a hash table
         * @param[in]  depth   16-bit depth image to compute relative depths from
         * @param[in]  window  Triplet positions are being offset to this window size
         * @param[out] p1Diff  Relative depth between depths at c and p1 triplet points locations (cD - p1D)
         * @param[out] p2Diff  Relative depth between depths at c and p2 triplet points locations (cD - p2D)
         * @param[out] nC      Center triplet point, normalized into objBB coordinates
         * @param[out] nP1     P1 triplet point, normalized into objBB coordinates
         * @param[out] nP2     P2 triplet point, normalized into objBB coordinates
         * @return             Return true if triplet passed all validation tests
         */
        bool validateTripletPoints(const Triplet &triplet, const cv::Mat &depth, cv::Rect window,
                                   int &p1Diff, int &p2Diff, cv::Point &nC, cv::Point &nP1, cv::Point &nP2);

        /**
         * @brief Computes bin ranges for each table (triplet) across all templates based on relative depths
         *
         * @param[in]     templates Input array of all templates parsed for detection
         * @param[in,out] tables    Input array of tables, which are then updated with their computed bin range
         */
        void initializeBinRanges(std::vector<Template> &templates, std::vector<HashTable> &tables);

    public:
        static const int IMG_16BIT_MAX = 65535;

        Hasher(cv::Ptr<ClassifierCriteria> criteria) : criteria(criteria) {}

        /**
         * @brief Generates and trains hash table on given array of Templates
         *
         * This function computes [criteria.tablesCount] hash tables. To provide better
         * results and fill tables as much as possible we first generate [50 * criteria.tablesCount]
         * hash tables, train them and retain only the amount of [criteria.tablesCount] of the most
         * covered values i.e. containing most templates at generated hash keys.
         *
         * @param[in]  templates Input array of templates from given dataset
         * @param[out] tables    Generated and trained hash tables for the set of given templates
         */
        void train(std::vector<Template> &templates, std::vector<HashTable> &tables);

        // TODO - Refactor function to perform better in parallel
        /**
         * @brief Picks first 100 best candidates for each window from included hashing tables
         *
         * This function computes hash keys on hash table triplets per each window. Then it looks
         * at the contents of hash table at computed key and votes for templates located at that key.
         * This is done for all hash tables. After that we pick 100 best templates (most votes) as
         * candidates for that specific window.
         *
         * @param[in]     depth   16-bit Scene depth image
         * @param[in]     normals 8-bit Image of quantized surface normals of scene depth image
         * @param[in]     tables  Array of pre-computed tables (with generated triplets) in training stage
         * @param[in,out] windows Array of windows that passed objectness detection test
         */
        void verifyCandidates(const cv::Mat &depth, const cv::Mat &normals, std::vector<HashTable> &tables, std::vector<Window> &windows);
    };
}

#endif
