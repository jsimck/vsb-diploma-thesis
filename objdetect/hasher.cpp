#include <unordered_set>
#include "hasher.h"
#include "../utils/timer.h"
#include "../processing/processing.h"
#include "../processing/computation.h"

namespace tless {
    HashKey Hasher::validateTripletAndComputeHashKey(const Triplet &triplet, const std::vector<cv::Range> &binRanges, const cv::Mat &depth,
                                                     const cv::Mat &normals, const cv::Mat &gray, cv::Rect window, uchar minGray) {
        // Checks
        assert(depth.type() == CV_16UC1);
        assert(normals.type() == CV_8UC1);
        assert(window.area() > 0);

        // Offset triplet points by template bounding box
        cv::Point nP1 = triplet.p1 + window.tl();
        cv::Point nP2 = triplet.p2 + window.tl();
        cv::Point nC = triplet.c + window.tl();

        const int brX = window.br().x;
        const int brY = window.br().y;

        // Ignore if we're out of object bounding box
        if (nC.x >= brX || nP1.x >= brX || nP2.x >= brX || nC.y >= brY || nP1.y >= brY || nP2.y >= brY) {
            return {};
        }

        // Check for minimal gray value (triplet is on an object)
        if (!gray.empty()) {
            assert(gray.type() == CV_8UC1);
            if (gray.at<uchar>(nP1) < minGray) return {};
            if (gray.at<uchar>(nP2) < minGray) return {};
            if (gray.at<uchar>(nC) < minGray) return {};
        }

        // Get quantized normals at triplet points
        uchar n1 = normals.at<uchar>(nP1);
        uchar n2 = normals.at<uchar>(nP2);
        uchar n3 = normals.at<uchar>(nC);

        // Validate quantized srcNormals
        if (n1 == 0 || n2 == 0 || n3 == 0) {
            return {};
        }

        // Get depth value at each triplet point
        auto p1D = static_cast<int>(depth.at<ushort>(nP1));
        auto p2D = static_cast<int>(depth.at<ushort>(nP2));
        auto cD = static_cast<int>(depth.at<ushort>(nC));

        // Ignore if there are any incorrect depth values
        if (cD <= 0 || p1D <= 0 || p2D <= 0) {
            return {};
        }

        // Initialize to invalid value, but != 0 to pass validation in bin Ranges generation
        uchar d1 = 200, d2 = 200;

        // Quantize depths
        if (!binRanges.empty()) {
            d1 = quantizeDepth(p1D - cD, binRanges);
            d2 = quantizeDepth(p2D - cD, binRanges);
        }

        // Skip wrong depths
        if (d1 == 0 || d2 == 0) {
            return {};
        }

        return {d1, d2, n1, n2, n3};
    }

    void Hasher::initializeBinRanges(std::vector<Template> &templates, std::vector<HashTable> &tables) {
        #pragma omp parallel for shared(templates, tables)
        for (size_t i = 0; i < tables.size(); i++) {
            const int binCount = criteria->depthBinCount;
            std::vector<int> rDepths;

            for (auto &t : templates) {
                assert(!t.srcDepth.empty());

                // Validate triplet
                if (validateTripletAndComputeHashKey(tables[i].triplet, {}, t.srcDepth, t.srcNormals, t.srcGray, t.objBB).empty()) {
                    continue;
                }

                // Offset triplet points by template bounding box
                cv::Point nP1 = tables[i].triplet.p1 + t.objBB.tl();
                cv::Point nP2 = tables[i].triplet.p2 + t.objBB.tl();
                cv::Point nC = tables[i].triplet.c + t.objBB.tl();

                // Get depth value at each triplet point
                auto p1D = static_cast<int>(t.srcDepth.at<ushort>(nP1));
                auto p2D = static_cast<int>(t.srcDepth.at<ushort>(nP2));
                auto cD = static_cast<int>(t.srcDepth.at<ushort>(nC));

                // Push relative depths
                rDepths.push_back(p1D - cD);
                rDepths.push_back(p2D - cD);
            }

            // Sort depths to calculate bin ranges
            std::sort(rDepths.begin(), rDepths.end());
            const size_t binSize = rDepths.size() / binCount;
            std::vector<cv::Range> ranges;

            // Skip tables with no valid relative depths
            if (binSize == 0) {
                continue;
            } else {
                for (int j = 0; j < binCount; j++) {
                    int min = rDepths[j * binSize];
                    int max = rDepths[(j + 1) * binSize];

                    if (j == 0) {
                        min += min / 5;
                    } else if (j + 1 == binCount) {
                        max += max * 0.2;
                    }

                    ranges.emplace_back(min, max);
                }

                assert(static_cast<int>(ranges.size()) == binCount);
                tables[i].binRanges = std::move(ranges);
            }
        }
    }

    void Hasher::train(std::vector<Template> &templates, std::vector<HashTable> &tables) {
        assert(!templates.empty());
        assert(criteria->tablesCount > 0);
        assert(criteria->tripletGrid.width > 0);
        assert(criteria->tripletGrid.height > 0);
        assert(criteria->info.largestArea.area() > 0);

        // Generate triplets
        const uint N = criteria->tablesCount * criteria->tablesTrainingMultiplier;
        for (uint i = 0; i < N; ++i) {
            tables.emplace_back(Triplet::create(criteria->tripletGrid, criteria->info.largestArea));
        }

        // Initialize bin ranges for each table
        initializeBinRanges(templates, tables);

        // Fill hash tables with templates at quantized keys
        #pragma omp parallel for shared(templates, tables) firstprivate(criteria)
        for (size_t i = 0; i < tables.size(); i++) {
            for (auto &t : templates) {
                // Skip tables with no no defined ranges
                if (tables[i].binRanges.empty()) {
                    continue;
                }

                // Validate and generate hash key at given triplet point
                HashKey key = validateTripletAndComputeHashKey(tables[i].triplet, tables[i].binRanges, t.srcDepth, t.srcNormals, t.srcGray, t.objBB);

                // Skip if validation failed, e.g. key is empty
                if (key.empty()) {
                    continue;
                }

                // Push unique templates to table
                tables[i].pushUnique(key, t);
            }
        }

        // Pick only first 100 tables with the most quantized templates
        std::stable_sort(tables.rbegin(), tables.rend());
        tables.resize(criteria->tablesCount);
    }

    void Hasher::verifyCandidates(const cv::Mat &depth, const cv::Mat &normals, std::vector<HashTable> &tables, std::vector<Window> &windows) {
        assert(!normals.empty());
        assert(!depth.empty());
        assert(!windows.empty());
        assert(!tables.empty());
        assert(criteria->info.largestArea.area() > 0);

        std::vector<Template *> usedTemplates;
        std::vector<size_t> emptyIndexes;

        for (size_t i = 0; i < windows.size(); ++i) {
            for (auto &table : tables) {
                // Validate and generate hash key at given triplet point
                HashKey key = validateTripletAndComputeHashKey(table.triplet, table.binRanges, depth, normals, cv::Mat(), windows[i].rect());

                // Skip if validation failed, e.g. key is empty
                if (key.empty()) {
                    continue;
                }

                // Vote for each template in hash table at specific key and push unique to window candidates
                for (auto &entry : table.templates[key]) {
                    entry->votes++;
                    entry->triplets.push_back(table.triplet); // TODO remove, mostly for debugging

                    // pushes only unique templates with minimum of votes (minVotes) building vector of size up to N
                    windows[i].pushUnique(entry, criteria->tablesCount, criteria->minVotes);
                    usedTemplates.push_back(entry);
                }
            }

            // Sort candidates based on the votes
            std::stable_sort(windows[i].candidates.begin(), windows[i].candidates.end(), [](Template *t1, Template *t2) { return t1->votes > t2->votes; }); // TODO remove, mostly for debugging

            // Save votes for current window in separate array
            for (auto &candidate : windows[i].candidates) {
                windows[i].votes.push_back(candidate->votes);
                windows[i].triplets.push_back(candidate->triplets); // TODO remove, mostly for debugging
            }

            // Reset votes for all used templates
            for (auto &t : usedTemplates) {
                t->votes = 0;
                t->triplets.clear(); // TODO remove, mostly for debugging
            }

            usedTemplates.clear();

            // Save empty windows indexes
            if (!windows[i].hasCandidates()) {
                emptyIndexes.emplace_back(i);
            }
        }

        // Remove empty windows
        removeIndex<Window>(windows, emptyIndexes);
    }
}