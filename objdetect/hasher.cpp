#include <unordered_set>
#include <utility>
#include "hasher.h"
#include "../utils/timer.h"
#include "../core/classifier_criteria.h"
#include "../processing/processing.h"
#include "../core/hash_table_candidate.h"

namespace tless {
    void Hasher::initializeBinRanges(std::vector<Template> &templates, std::vector<HashTable> &tables) {
        #pragma omp parallel for shared(templates, tables)
        for (size_t i = 0; i < tables.size(); i++) {
            const int binCount = 5;
            std::vector<int> rDepths;

            for (auto &t : templates) {
                assert(!t.srcDepth.empty());

                // Offset triplet points by template bounding box
                cv::Point c = tables[i].triplet.c + t.objBB.tl();
                cv::Point p1 = tables[i].triplet.p1 + t.objBB.tl();
                cv::Point p2 = tables[i].triplet.p2 + t.objBB.tl();

                const int brX = t.objBB.br().x;
                const int brY = t.objBB.br().y;

                // Ignore if we're out of object bounding box
                if (c.x >= brX || p1.x >= brX || p2.x >= brX || c.y >= brY || p1.y >= brY || p2.y >= brY) {
                    continue;
                }

                // Get depth value at each triplet point
                auto cD = static_cast<int>(t.srcDepth.at<ushort>(c));
                auto p1D = static_cast<int>(t.srcDepth.at<ushort>(p1));
                auto p2D = static_cast<int>(t.srcDepth.at<ushort>(p2));

                // Ignore if there are any incorrect depth values
                if (cD <= 0 || p1D <= 0 || p2D <= 0) {
                    continue;
                }

                // Extract relative depths
                rDepths.push_back(p1D - cD);
                rDepths.push_back(p2D - cD);
            }

            // Sort depths to calculate bin ranges
            std::sort(rDepths.begin(), rDepths.end());
            const size_t binSize = rDepths.size() / binCount;
            std::vector<cv::Range> ranges;

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

    void Hasher::train(std::vector<Template> &templates, std::vector<HashTable> &tables) {
        assert(!templates.empty());
        assert(criteria->tablesCount > 0);
        assert(criteria->tripletGrid.width > 0);
        assert(criteria->tripletGrid.height > 0);
        assert(criteria->info.largestArea.area() > 0);

        // Generate triplets
        const uint N = criteria->tablesCount * 50;
        for (uint i = 0; i < N; ++i) {
            tables.emplace_back(Triplet::create(criteria->tripletGrid, criteria->info.largestArea));
        }

        // Initialize bin ranges for each table
        initializeBinRanges(templates, tables);

//        // Prepare hash tables and histogram bin ranges
//        initialize(templates, tables);
//        const size_t iSize = tables.size();
//
//        // Fill hash tables with templates and keys quantized from measured values
//        #pragma omp parallel for shared(templates, tables) firstprivate(criteria)
//        for (size_t i = 0; i < iSize; i++) {
//            for (auto &t : templates) {
//                // Checks
//                assert(!t.srcDepth.empty());
//
//                // Get triplet points
//                TripletParams coordParams(criteria->info.largestArea.width, criteria->info.largestArea.height,
//                                          criteria->tripletGrid, t.objBB.tl().x, t.objBB.tl().y);
//                cv::Point c = tables[i].triplet.getCenter(coordParams);
//                cv::Point p1 = tables[i].triplet.getP1(coordParams);
//                cv::Point p2 = tables[i].triplet.getP2(coordParams);
//
//                // Check if we're not out of bounds
//                assert(c.x >= 0 && c.x < t.srcGray.cols);
//                assert(c.y >= 0 && c.y < t.srcGray.rows);
//                assert(p1.x >= 0 && p1.x < t.srcGray.cols);
//                assert(p1.y >= 0 && p1.y < t.srcGray.rows);
//                assert(p2.x >= 0 && p2.x < t.srcGray.cols);
//                assert(p2.y >= 0 && p2.y < t.srcGray.rows);
//
//                // Relative depths
//                int d[2];
//                relativeDepths(t.srcDepth, c, p1, p2, d);
//
//                // Generate hash key
//                HashKey key(
//                        quantizedDepth(d[0], tables[i].binRanges),
//                        quantizedDepth(d[1], tables[i].binRanges),
//                        t.srcNormals.at<uchar>(c),
//                        t.srcNormals.at<uchar>(p1),
//                        t.srcNormals.at<uchar>(p2)
//                );
//
//                // Check if key exists, if not initialize it
//                tables[i].pushUnique(key, t);
//            }
//        }
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
                TripletParams tParams(criteria->info.largestArea.width, criteria->info.largestArea.height,
                                      criteria->tripletGrid, windows[i].tl().x, windows[i].tl().y);
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
                for (auto &entry : table.templates[key]) {
                    if (tableCandidates.count(entry->id) == 0) {
                        tableCandidates[entry->id] = HashTableCandidate(entry);
                    }

                    // Increase votes for candidate
                    tableCandidates[entry->id].vote();
                }
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