#include <unordered_set>
#include "hasher.h"
#include "../utils/timer.h"
#include "../processing/processing.h"
#include "../core/hash_table_candidate.h"
#include "../processing/computation.h"

namespace tless {
    bool Hasher::validateTripletPoints(const Triplet &triplet, const cv::Mat &depth, cv::Rect window,
                                       int &p1Diff, int &p2Diff, cv::Point &nC, cv::Point &nP1, cv::Point &nP2) {
        // Offset triplet points by template bounding box
        nC = triplet.c + window.tl();
        nP1 = triplet.p1 + window.tl();
        nP2 = triplet.p2 + window.tl();

        const int brX = window.br().x;
        const int brY = window.br().y;

        // Ignore if we're out of object bounding box
        if (nC.x >= brX || nP1.x >= brX || nP2.x >= brX || nC.y >= brY || nP1.y >= brY || nP2.y >= brY) {
            return false;
        }

        // Get depth value at each triplet point
        auto cD = static_cast<int>(depth.at<ushort>(nC));
        auto p1D = static_cast<int>(depth.at<ushort>(nP1));
        auto p2D = static_cast<int>(depth.at<ushort>(nP2));

        // Ignore if there are any incorrect depth values
        if (cD <= 0 || p1D <= 0 || p2D <= 0) {
            return false;
        }

        // Calculate relative depths
        p1Diff = p1D - cD;
        p2Diff = p2D - cD;

        return true;
    }

    void Hasher::initializeBinRanges(std::vector<Template> &templates, std::vector<HashTable> &tables) {
        #pragma omp parallel for shared(templates, tables)
        for (size_t i = 0; i < tables.size(); i++) {
            const int binCount = 5;
            std::vector<int> rDepths;

            for (auto &t : templates) {
                assert(!t.srcDepth.empty());

                cv::Point c, p1, p2;
                int p1Diff, p2Diff;

                // Skip if points are not valid
                if (!validateTripletPoints(tables[i].triplet, t.srcDepth, t.objBB, p1Diff, p2Diff, c, p1, p2)) {
                    continue;
                }

                // Push relative depths
                rDepths.push_back(p1Diff);
                rDepths.push_back(p2Diff);
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
                        min = -IMG_16BIT_MAX;
                    } else if (j + 1 == binCount) {
                        max = IMG_16BIT_MAX;
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
        const uint N = criteria->tablesCount * 50;
        for (uint i = 0; i < N; ++i) {
            tables.emplace_back(std::move(Triplet::create(criteria->tripletGrid, criteria->info.largestArea)));
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

                cv::Point c, p1, p2;
                int p1Diff, p2Diff;

                // Skip if points are not valid
                if (!validateTripletPoints(tables[i].triplet, t.srcDepth, t.objBB, p1Diff, p2Diff, c, p1, p2)) {
                    continue;
                }

                // Generate hash key
                HashKey key(
                        quantizeDepth(p1Diff, tables[i].binRanges),
                    quantizeDepth(p2Diff, tables[i].binRanges),
                    t.srcNormals.at<uchar>(c),
                    t.srcNormals.at<uchar>(p1),
                    t.srcNormals.at<uchar>(p2)
                );

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
                // Skip tables with no no defined ranges
                if (table.binRanges.empty()) {
                    continue;
                }

                cv::Point c, p1, p2;
                int p1Diff, p2Diff;

                // Skip if points are not valid
                if (!validateTripletPoints(table.triplet, depth, windows[i].rect(), p1Diff, p2Diff, c, p1, p2)) {
                    continue;
                }

                HashKey key(
                    quantizeDepth(p1Diff, table.binRanges),
                    quantizeDepth(p2Diff, table.binRanges),
                    normals.at<uchar>(c),
                    normals.at<uchar>(p1),
                    normals.at<uchar>(p2)
                );

                // Vote for each template in hash table at specific key and push unique to window candidates
                for (auto &entry : table.templates[key]) {
                    entry->votes++;

                    // pushes only unique templates with minimum of votes (minVotes) building vector of size up to N
                    windows[i].pushUnique(entry, criteria->tablesCount, criteria->minVotes);
                    usedTemplates.emplace_back(entry);
                }
            }

            // Reset votes for all used templates
            for (auto &t : usedTemplates) {
                t->votes = 0;
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