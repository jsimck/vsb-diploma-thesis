#include <unordered_set>
#include "hasher.h"
#include "../utils/timer.h"
#include "../processing/processing.h"
#include "../core/hash_table_candidate.h"

namespace tless {
    bool Hasher::validateTripletPoints(const Triplet &triplet, Template &tpl, int &p1Diff, int &p2Diff, cv::Point &nC, cv::Point &nP1, cv::Point &nP2) {
        // Offset triplet points by template bounding box
        nC = triplet.c + tpl.objBB.tl();
        nP1 = triplet.p1 + tpl.objBB.tl();
        nP2 = triplet.p2 + tpl.objBB.tl();

        const int brX = tpl.objBB.br().x;
        const int brY = tpl.objBB.br().y;

        // Ignore if we're out of object bounding box
        if (nC.x >= brX || nP1.x >= brX || nP2.x >= brX || nC.y >= brY || nP1.y >= brY || nP2.y >= brY) {
            return false;
        }

        // Get depth value at each triplet point
        auto cD = static_cast<int>(tpl.srcDepth.at<ushort>(nC));
        auto p1D = static_cast<int>(tpl.srcDepth.at<ushort>(nP1));
        auto p2D = static_cast<int>(tpl.srcDepth.at<ushort>(nP2));

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
                if (!validateTripletPoints(tables[i].triplet, t, p1Diff, p2Diff, c, p1, p2)) {
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
                if (!validateTripletPoints(tables[i].triplet, t, p1Diff, p2Diff, c, p1, p2)) {
                    continue;
                }

                // Generate hash key
                HashKey key(
                    quantizedDepth(p1Diff, tables[i].binRanges),
                    quantizedDepth(p2Diff, tables[i].binRanges),
                    t.srcNormals.at<uchar>(c),
                    t.srcNormals.at<uchar>(p1),
                    t.srcNormals.at<uchar>(p2)
                );

                // Push unique templates to table
                tables[i].pushUnique(key, t);
            }
        }

        // Pick only first 100 tables with the most quantized templates
        std::sort(tables.rbegin(), tables.rend());
        tables.resize(criteria->tablesCount);
    }

// #define VISUALIZE
// TODO skip wrong depths
    void Hasher::verifyCandidates(const cv::Mat &sceneDepth, const cv::Mat &sceneSurfaceNormalsQuantized,
                                         std::vector<HashTable> &tables, std::vector<Window> &windows) {
        // Checks
        assert(!sceneSurfaceNormalsQuantized.empty());
        assert(!sceneDepth.empty());
        assert(!windows.empty());
        assert(!tables.empty());
        assert(criteria->info.largestArea.area() > 0);

        std::vector<Window> newWindows;
        const size_t windowsSize = windows.size();

        // TODO find and fix memory leak
#ifdef VISUALIZE
        //    #pragma omp parallel for default(none) shared(windows, newWindows, sceneDepth, sceneSurfaceNormalsQuantized, tables) firstprivate(criteria) ordered
#else
//    #pragma omp parallel for default(none) shared(windows, newWindows, sceneDepth, sceneSurfaceNormalsQuantized, tables) firstprivate(criteria)
#endif
        for (size_t i = 0; i < windowsSize; ++i) {
            std::unordered_map<int, HashTableCandidate> tableCandidates;

            for (auto &table : tables) {
                // Get triplet points
//                TripletParams tParams(criteria->info.largestArea.width, criteria->info.largestArea.height,
//                                      criteria->tripletGrid, windows[i].tl().x, windows[i].tl().y);
//                cv::Point c = table.triplet.getCenter(tParams);
//                cv::Point p1 = table.triplet.getP1(tParams);
//                cv::Point p2 = table.triplet.getP2(tParams);
                cv::Point c = table.triplet.c;
                cv::Point p1 = table.triplet.p1;
                cv::Point p2 = table.triplet.p2;

                // If any point of triplet is out of scene boundaries, ignore it to not get false data
                if ((c.x < 0 || c.x >= sceneDepth.cols || c.y < 0 || c.y >= sceneDepth.rows) ||
                    (p1.x < 0 || p1.x >= sceneDepth.cols || p1.y < 0 || p1.y >= sceneDepth.rows) ||
                    (p2.x < 0 || p2.x >= sceneDepth.cols || p2.y < 0 || p2.y >= sceneDepth.rows))
                    continue;

                // Relative depths
                int d[2];
                relativeDepths(sceneDepth, c, p1, p2, d);

                // Generate hash key
                HashKey key(
                        quantizedDepth(d[0], table.binRanges),
                        quantizedDepth(d[1], table.binRanges),
                        sceneSurfaceNormalsQuantized.at<uchar>(c),
                        sceneSurfaceNormalsQuantized.at<uchar>(p1),
                        sceneSurfaceNormalsQuantized.at<uchar>(p2)
                );

                // Put each candidate to hash table, increase votes for existing tableCandidates
//                for (auto &entry : table.templates[key]) {
//                    if (tableCandidates.count(entry->id) == 0) {
//                        tableCandidates[entry->id] = HashTableCandidate(entry);
//                    }
//
//                    // Increase votes for candidate
//                    tableCandidates[entry->id].vote();
//                }
            }

            Timer t;
            // Insert all tableCandidates with above min votes to helper vector
            std::vector<HashTableCandidate> passedCandidates;
            for (auto &c : tableCandidates) {
                if (c.second.votes >= criteria->minVotes) {
                    passedCandidates.push_back(c.second);
                }
            }

            // Sort and pick first 100 (DESCENDING)
            if (!passedCandidates.empty()) {
                std::sort(passedCandidates.rbegin(), passedCandidates.rend());

                // Put only first 100 candidates
                size_t pcSize = passedCandidates.size();
                pcSize = (pcSize > criteria->tablesCount) ? criteria->tablesCount : pcSize;

                for (size_t j = 0; j < pcSize; ++j) {
                    windows[i].candidates.push_back(passedCandidates[j].candidate);
                }

#ifdef VISUALIZE
                //            #pragma omp ordered
#else
//            #pragma omp critical
#endif
                newWindows.push_back(std::move(windows[i]));
            }
        }

        // Remove empty windows
        windows = newWindows;
    }
}