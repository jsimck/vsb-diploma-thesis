#include <unordered_set>
#include <utility>
#include "hasher.h"
#include "../utils/utils.h"
#include "../utils/timer.h"
#include "../core/classifier_criteria.h"
#include "../processing/processing.h"
#include "../core/hash_table_candidate.h"

const int Hasher::IMG_16BIT_MAX = 65535; // <0, 65535> => 65536 values

void Hasher::generateTriplets(std::vector<HashTable> &tables) {
    // Generate triplets
    for (int i = 0; i < criteria->train.hasher.tablesCount; ++i) {
        tables.emplace_back(Triplet::create(criteria->train.hasher.grid, criteria->train.hasher.maxDistance));
    }

    // TODO - joint entropy of 5k triplets, instead of 100 random triplets
    // Check for unique triplets, and regenerate duplicates
    bool duplicate;
    do {
        duplicate = false;
        for (int i = 0; i < criteria->train.hasher.tablesCount; ++i) {
            for (int j = 0; j < criteria->train.hasher.tablesCount; ++j) {
                // Don't compare same triplets
                if (i == j) continue;
                if (tables[i].triplet == tables[j].triplet) {
                    // Duplicate, generate new triplet
                    duplicate = true;
                    tables[j].triplet = std::move(Triplet::create(criteria->train.hasher.grid));
                }
            }
        }
    } while (duplicate);
}

void Hasher::initializeBinRanges(std::vector<Template> &templates, std::vector<HashTable> &tables) {
    // Checks
    assert(!tables.empty());
    assert(!templates.empty());

    const size_t iSize = tables.size();

    #pragma omp parallel for shared(templates, tables) firstprivate(criteria)
    for (size_t i = 0; i < iSize; i++) {
        std::vector<int> rDepths;

        for (auto &t : templates) {
            // Checks
            assert(!t.srcDepth.empty());

            // Offset for the triplet grid
            cv::Point gridOffset(
                t.objBB.tl().x - (criteria->info.maxTemplate.width - t.objBB.width) / 2,
                t.objBB.tl().y - (criteria->info.maxTemplate.height - t.objBB.height) / 2
            );

            // Absolute triplet points
            TripletParams tParams(criteria->info.maxTemplate.width, criteria->info.maxTemplate.height, criteria->train.hasher.grid, gridOffset.x, gridOffset.y);
            cv::Point c = tables[i].triplet.getCenter(tParams);
            cv::Point p1 = tables[i].triplet.getP1(tParams);
            cv::Point p2 = tables[i].triplet.getP2(tParams);

            // Check if we're not out of bounds
            assert(c.x >= 0 && c.x < t.srcDepth.cols);
            assert(c.y >= 0 && c.y < t.srcDepth.rows);
            assert(p1.x >= 0 && p1.x < t.srcDepth.cols);
            assert(p1.y >= 0 && p1.y < t.srcDepth.rows);
            assert(p2.x >= 0 && p2.x < t.srcDepth.cols);
            assert(p2.y >= 0 && p2.y < t.srcDepth.rows);

            // Relative depths
            int d[2];
            Processing::relativeDepths(t.srcDepth, c, p1, p2, d);
            rDepths.push_back(d[0]);
            rDepths.push_back(d[1]);
        }

        // Sort depths Calculate bin ranges
        std::sort(rDepths.begin(), rDepths.end());
        const size_t rDSize = rDepths.size();
        const size_t binSize = rDSize / criteria->train.hasher.binCount;
        std::vector<cv::Range> ranges;

        for (int j = 0; j < criteria->train.hasher.binCount; j++) {
            int min = rDepths[j * binSize];
            int max = rDepths[(j + 1) * binSize];

            if (j == 0) {
                min = -IMG_16BIT_MAX;
            } else if (j + 1 == criteria->train.hasher.binCount) {
                max = IMG_16BIT_MAX;
            }

            ranges.emplace_back(min, max);
        }

        // Set table bin ranges
        assert(static_cast<int>(ranges.size()) == criteria->train.hasher.binCount);
        tables[i].binRanges = ranges;
    }
}

void Hasher::initialize(std::vector<Template> &templates, std::vector<HashTable> &tables) {
    // Init hash tables
    tables.reserve(criteria->train.hasher.tablesCount);
    std::cout << "    |_ Generating triplets... " << std::endl;
    generateTriplets(tables);

    // Calculate bin ranges for depth quantization
    std::cout << "    |_ Calculating depths bin ranges... " << std::endl;
    initializeBinRanges(templates, tables);
}

void Hasher::train(std::vector<Template> &templates, std::vector<HashTable> &tables) {
    // Checks
    assert(!templates.empty());
    assert(criteria->train.hasher.tablesCount > 0);
    assert(criteria->train.hasher.grid.width > 0);
    assert(criteria->train.hasher.grid.height > 0);
    assert(criteria->info.maxTemplate.area() > 0);

    // Prepare hash tables and histogram bin ranges
    initialize(templates, tables);
    const size_t iSize = tables.size();

    // Fill hash tables with templates and keys quantized from measured values
    #pragma omp parallel for shared(templates, tables) firstprivate(criteria)
    for (size_t i = 0; i < iSize; i++) {
        for (auto &t : templates) {
            // Checks
            assert(!t.srcDepth.empty());

            // Get triplet points
            TripletParams coordParams(criteria->info.maxTemplate.width, criteria->info.maxTemplate.height, criteria->train.hasher.grid, t.objBB.tl().x, t.objBB.tl().y);
            cv::Point c = tables[i].triplet.getCenter(coordParams);
            cv::Point p1 = tables[i].triplet.getP1(coordParams);
            cv::Point p2 = tables[i].triplet.getP2(coordParams);

            // Check if we're not out of bounds
            assert(c.x >= 0 && c.x < t.srcGray.cols);
            assert(c.y >= 0 && c.y < t.srcGray.rows);
            assert(p1.x >= 0 && p1.x < t.srcGray.cols);
            assert(p1.y >= 0 && p1.y < t.srcGray.rows);
            assert(p2.x >= 0 && p2.x < t.srcGray.cols);
            assert(p2.y >= 0 && p2.y < t.srcGray.rows);

            // Relative depths
            int d[2];
            Processing::relativeDepths(t.srcDepth, c, p1, p2, d);

            // Generate hash key
            HashKey key(
                Processing::quantizeDepth(d[0], tables[i].binRanges),
                Processing::quantizeDepth(d[1], tables[i].binRanges),
                t.quantizedNormals.at<uchar>(c),
                t.quantizedNormals.at<uchar>(p1),
                t.quantizedNormals.at<uchar>(p2)
            );

            // Check if key exists, if not initialize it
            tables[i].pushUnique(key, t);
        }
    }
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
    assert(criteria->info.maxTemplate.area() > 0);

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
            TripletParams tParams(criteria->info.maxTemplate.width, criteria->info.maxTemplate.height, criteria->train.hasher.grid, windows[i].tl().x, windows[i].tl().y);
            cv::Point c = table.triplet.getCenter(tParams);
            cv::Point p1 = table.triplet.getP1(tParams);
            cv::Point p2 = table.triplet.getP2(tParams);

            // If any point of triplet is out of scene boundaries, ignore it to not get false data
            if ((c.x < 0 || c.x >= sceneDepth.cols || c.y < 0 || c.y >= sceneDepth.rows) ||
                (p1.x < 0 || p1.x >= sceneDepth.cols || p1.y < 0 || p1.y >= sceneDepth.rows) ||
                (p2.x < 0 || p2.x >= sceneDepth.cols || p2.y < 0 || p2.y >= sceneDepth.rows)) continue;

            // Relative depths
            int d[2];
            Processing::relativeDepths(sceneDepth, c, p1, p2, d);

            // Generate hash key
            HashKey key(
                Processing::quantizeDepth(d[0], table.binRanges),
                Processing::quantizeDepth(d[1], table.binRanges),
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
            if (c.second.votes >= criteria->detect.hasher.minVotes) {
                passedCandidates.push_back(c.second);
            }
        }

        // Sort and pick first 100 (DESCENDING)
        if (!passedCandidates.empty()) {
            std::sort(passedCandidates.rbegin(), passedCandidates.rend());

            // Put only first 100 candidates
            size_t pcSize = passedCandidates.size();
            pcSize = (pcSize > criteria->train.hasher.tablesCount) ? criteria->train.hasher.tablesCount : pcSize;

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

void Hasher::setCriteria(std::shared_ptr<ClassifierCriteria> criteria) {
    this->criteria = criteria;
}