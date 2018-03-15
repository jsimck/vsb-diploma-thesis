#ifndef VSB_SEMESTRAL_PROJECT_HASHING_H
#define VSB_SEMESTRAL_PROJECT_HASHING_H

#include <opencv2/opencv.hpp>
#include <memory>
#include "../core/hash_table.h"
#include "../core/classifier_criteria.h"
#include "../core/window.h"

namespace tless {
    /**
     * @brief Class used to train HashTables and quickly verify what templates should be matched per each window passed form objectness detection.
     */
    class Hasher {
    private:
        cv::Ptr<ClassifierCriteria> criteria;

        /**
         * @brief Validates and generates hash key at triplet position if it's valid.
         *
         * First the triplet points are offset by window.tl() position (which is mostly used in scene verification).
         * After then we do checks whether the triplet is not out of window boundary, if gray value at triplet points
         * is bigger than minGray (if gray provided). Finally we get quantized normals and perform validity check (n != 0),
         * compute depth differences between each triplet point and return valid HashKey if all those above are valid.
         *
         * @param[in] triplet   Generated triplet from a hash table
         * @param[in] binRanges Array of binRanges of quantized depths computed for each hash table, if not provided hash key is still valid but
         *                      with random depths, this is to assure that validation passes in bin ranges initialization.
         * @param[in] depth     16-bit depth image to compute relative depths from
         * @param[in] normals   8-bit uchar image of quantized normals
         * @param[in] gray      8-bit optional gray image, in training phase we check if the value is above threshold (to validate there's an object)
         * @param[in] window    Triplet positions are being offset to this window size
         * @param[in] minGray   Minimum value of gray image to be considered as containing object
         * @return              Returns valid HashKey if all points and quantized values were valid, otherwise returns empty HashKey
         */
        HashKey validateTripletAndComputeHashKey(const Triplet &triplet, const std::vector<cv::Range> &binRanges, const cv::Mat &depth, const cv::Mat &normals,
                                              const cv::Mat &gray, cv::Rect window, uchar minGray = 40);

        /**
         * @brief Computes bin ranges for each table (triplet) across all templates based on relative depths.
         *
         * @param[in]     templates Input array of all templates parsed for detection
         * @param[in,out] tables    Input array of tables, which are then updated with their computed bin range
         */
        void initializeBinRanges(std::vector<Template> &templates, std::vector<HashTable> &tables);

    public:
        static const int IMG_16BIT_MAX = 65535;

        Hasher(cv::Ptr<ClassifierCriteria> criteria) : criteria(criteria) {}

        /**
         * @brief Generates and trains hash table on given array of Templates.
         *
         * This function computes [criteria.tablesCount] hash tables. To provide better
         * results and fill tables as much as possible we first generate
         * [criteria.tablesTrainingMultiplier * criteria.tablesCount]
         * hash tables, train them and retain only the amount of [criteria.tablesCount] of the most
         * covered values i.e. containing most templates at generated hash keys.
         *
         * @param[in]  templates Input array of templates from given dataset
         * @param[out] tables    Generated and trained hash tables for the set of given templates
         */
        void train(std::vector<Template> &templates, std::vector<HashTable> &tables);

        // TODO - Refactor function to perform better in parallel
        /**
         * @brief Picks first 100 best candidates for each window from included hashing tables.
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
