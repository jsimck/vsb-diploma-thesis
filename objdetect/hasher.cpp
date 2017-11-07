#include <unordered_set>
#include "hasher.h"
#include "../utils/utils.h"
#include "../utils/timer.h"
#include "../core/classifier_terms.h"

const int Hasher::IMG_16BIT_MAX = 65535; // <0, 65535> => 65536 values

cv::Vec3f Hasher::surfaceNormal(const cv::Mat &src, const cv::Point &c) {
    assert(!src.empty());
    assert(src.type() == CV_32FC1);

    float dzdx = (src.at<float>(c.y, c.x + 1) - src.at<float>(c.y, c.x - 1)) / 2.0f;
    float dzdy = (src.at<float>(c.y + 1, c.x) - src.at<float>(c.y - 1, c.x)) / 2.0f;
    cv::Vec3f d(-dzdy, -dzdx, 1.0f);

    return cv::normalize(d);
}

cv::Vec2i Hasher::relativeDepths(const cv::Mat &src, const cv::Point &c, const cv::Point &p1, const cv::Point &p2) {
    assert(!src.empty());
    assert(src.type() == CV_32FC1);

    return cv::Vec2i(
        static_cast<int>(src.at<float>(p1) - src.at<float>(c)),
        static_cast<int>(src.at<float>(p2) - src.at<float>(c))
    );
}

uchar Hasher::quantizeSurfaceNormal(const cv::Vec3f &normal) {
    // Normal z coordinate should not be < 0
    assert(normal[2] >= 0);

    // TODO - asi spis nefunguje
    // In our case z is always positive, that's why we're using
    // 8 octants in top half of sphere only to quantize into 8 bins
    static cv::Vec3f octantNormals[8] = {
        cv::Vec3f(0.707107f, 0.0f, 0.707107f), // 0. octant
        cv::Vec3f(0.57735f, 0.57735f, 0.57735f), // 1. octant
        cv::Vec3f(0.0f, 0.707107f, 0.707107f), // 2. octant
        cv::Vec3f(-0.57735f, 0.57735f, 0.57735f), // 3. octant
        cv::Vec3f(-0.707107f, 0.0f, 0.707107f), // 4. octant
        cv::Vec3f(-0.57735f, -0.57735f, 0.57735f), // 5. octant
        cv::Vec3f(0.0f, -0.707107f, 0.707107f), // 6. octant
        cv::Vec3f(0.57735f, -0.57735f, 0.57735f), // 7. octant
    };

    uchar minIndex = 9;
    float maxDot = 0, dot = 0;
    for (uchar i = 0; i < 8; i++) {
        // By doing dot product between octant octantNormals and calculated normal
        // we can find maximum -> index of octant where the vector belongs to
        dot = normal.dot(octantNormals[i]);

        if (dot > maxDot) {
            maxDot = dot;
            minIndex = i;
        }
    }

    // Index should in interval <0,7>
    assert(minIndex >= 0 && minIndex < 8);

    return minIndex;
}

uchar Hasher::quantizeDepth(float depth, const std::vector<cv::Range> &ranges) {
    // Depth should have max value of <-65536, +65536>
    assert(depth >= -IMG_16BIT_MAX && depth <= IMG_16BIT_MAX);
    assert(!ranges.empty());

    // Loop through histogram ranges and return quantized index
    const size_t iSize = ranges.size();
    for (size_t i = 0; i < iSize; i++) {
        if (ranges[i].start >= depth && depth < ranges[i].end) {
            return static_cast<uchar>(i);
        }
    }

    // If value is IMG_16BIT_MAX it belongs to last bin
    return static_cast<uchar>(iSize - 1);
}

void Hasher::generateTriplets(std::vector<HashTable> &tables) {
    // Generate triplets
    for (size_t i = 0; i < terms->params.hasher.tablesCount; ++i) {
        tables.emplace_back(Triplet::create(terms->params.hasher.grid, terms->params.hasher.maxDistance));
    }

    // TODO - joint entropy of 5k triplets, instead of 100 random triplets
    // Check for unique triplets, and regenerate duplicates
    bool duplicate;
    do {
        duplicate = false;
        for (size_t i = 0; i < terms->params.hasher.tablesCount; ++i) {
            for (size_t j = 0; j < terms->params.hasher.tablesCount; ++j) {
                // Don't compare same triplets
                if (i == j) continue;
                if (tables[i].triplet == tables[j].triplet) {
                    // Duplicate, generate new triplet
                    duplicate = true;
                    tables[j].triplet = Triplet::create(terms->params.hasher.grid);
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

    #pragma omp parallel for
    for (size_t i = 0; i < iSize; i++) {
        std::vector<int> rDepths;

        for (auto &t : templates) {
            // Checks
            assert(!t.srcDepth.empty());

            // Offset for the triplet grid
            cv::Point gridOffset(
                t.objBB.tl().x - (terms->info.maxTemplate.width - t.objBB.width) / 2,
                t.objBB.tl().y - (terms->info.maxTemplate.height - t.objBB.height) / 2
            );

            // Absolute triplet points
            TripletParams tParams(terms->info.maxTemplate.width, terms->info.maxTemplate.height, terms->params.hasher.grid, gridOffset.x, gridOffset.y);
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
            cv::Vec2i d = relativeDepths(t.srcDepth, c, p1, p2);
            rDepths.emplace_back(d[0]);
            rDepths.emplace_back(d[1]);
        }

        // Sort depths Calculate bin ranges
        std::sort(rDepths.begin(), rDepths.end());
        const size_t rDSize = rDepths.size();
        const size_t binSize = rDSize / terms->params.hasher.binCount;
        std::vector<cv::Range> ranges;

        for (uint j = 0; j < terms->params.hasher.binCount; j++) {
            int min = rDepths[j * binSize];
            int max = rDepths[(j + 1) * binSize];

            if (j == 0) {
                min = -IMG_16BIT_MAX;
            } else if (j + 1 == terms->params.hasher.binCount) {
                max = IMG_16BIT_MAX;
            }

            ranges.emplace_back(min, max);
        }

        // Set table bin ranges
        assert(ranges.size() == terms->params.hasher.binCount);
        tables[i].binRanges = ranges;
    }
}

void Hasher::initialize(std::vector<Template> &templates, std::vector<HashTable> &tables) {
    // Init hash tables
    tables.reserve(terms->params.hasher.tablesCount);
    std::cout << "    |_ Generating triplets... " << std::endl;
    generateTriplets(tables);

    // Calculate bin ranges for depth quantization
    std::cout << "    |_ Calculating depths bin ranges... " << std::endl;
    initializeBinRanges(templates, tables);
}

void Hasher::train(std::vector<Template> &templates, std::vector<HashTable> &tables) {
    // Checks
    assert(!templates.empty());
    assert(terms->params.hasher.tablesCount > 0);
    assert(terms->params.hasher.grid.width > 0);
    assert(terms->params.hasher.grid.height > 0);
    assert(info.maxTemplate.area() > 0);

    // Prepare hash tables and histogram bin ranges
    initialize(templates, tables);
    const size_t iSize = tables.size();

    // Fill hash tables with templates and keys quantized from measured values
    #pragma omp parallel for
    for (size_t i = 0; i < iSize; i++) {
        for (auto &t : templates) {
            // Checks
            assert(!t.srcDepth.empty());

            // Get triplet points
            TripletParams coordParams(terms->info.maxTemplate.width, terms->info.maxTemplate.height, terms->params.hasher.grid, t.objBB.tl().x, t.objBB.tl().y);
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
            cv::Vec2i d = relativeDepths(t.srcDepth, c, p1, p2);

            // Generate hash key
            HashKey key(
                quantizeDepth(d[0], tables[i].binRanges),
                quantizeDepth(d[1], tables[i].binRanges),
                quantizeSurfaceNormal(surfaceNormal(t.srcDepth, c)),
                quantizeSurfaceNormal(surfaceNormal(t.srcDepth, p1)),
                quantizeSurfaceNormal(surfaceNormal(t.srcDepth, p2))
            );

            // Check if key exists, if not initialize it
            tables[i].pushUnique(key, t);
        }
    }
}

void Hasher::verifyCandidates(cv::Mat &sceneDepth, std::vector<HashTable> &tables, std::vector<Window> &windows) {
    // Checks
    assert(!sceneDepth.empty());
    assert(!windows.empty());
    assert(!tables.empty());
    assert(terms->info.maxTemplate.area() > 0);

    const size_t windowsSize = windows.size();
    std::vector<Template *> usedTemplates;
    std::vector<size_t> emptyIndexes;

    for (size_t i = 0; i < windowsSize; ++i) {
        for (auto &table : tables) {
            // Get triplet points
            TripletParams tParams(terms->info.maxTemplate.width, terms->info.maxTemplate.height, terms->params.hasher.grid, windows[i].tl().x, windows[i].tl().y);
            cv::Point c = table.triplet.getCenter(tParams);
            cv::Point p1 = table.triplet.getP1(tParams);
            cv::Point p2 = table.triplet.getP2(tParams);

            // If any point of triplet is out of scene boundaries, ignore it to not get false data
            if ((c.x < 0 || c.x >= sceneDepth.cols || c.y < 0 || c.y >= sceneDepth.rows) ||
                (p1.x < 0 || p1.x >= sceneDepth.cols || p1.y < 0 || p1.y >= sceneDepth.rows) ||
                (p2.x < 0 || p2.x >= sceneDepth.cols || p2.y < 0 || p2.y >= sceneDepth.rows)) continue;

            // Relative depths
            cv::Vec2i d = relativeDepths(sceneDepth, c, p1, p2);

            // Generate hash key
            HashKey key(
                quantizeDepth(d[0], table.binRanges),
                quantizeDepth(d[1], table.binRanges),
                quantizeSurfaceNormal(surfaceNormal(sceneDepth, c)),
                quantizeSurfaceNormal(surfaceNormal(sceneDepth, p1)),
                quantizeSurfaceNormal(surfaceNormal(sceneDepth, p2))
            );

            // Vote for each template in hash table at specific key and push unique to window candidates
            for (auto &entry : table.templates[key]) {
                entry->vote();

                // pushes only unique templates with minimum of votes (minVotes) building vector of size up to N
                windows[i].pushUnique(entry, terms->params.hasher.tablesCount, terms->params.hasher.minVotes);
                usedTemplates.emplace_back(entry);
            }
        }

        // Reset votes for all used templates
        for (auto &t : usedTemplates) {
            t->resetVotes();
        }

        // Clear used templates vector
        usedTemplates.clear();

        // Save empty windows indexes
        if (!windows[i].hasCandidates()) {
            emptyIndexes.emplace_back(i);
        }
    }

    // Remove empty windows
    Utils::removeIndex<Window>(windows, emptyIndexes);
    std::cout << "  |_ Number of windows pass to next stage: " << windows.size() << std::endl;
}

void Hasher::setTerms(std::shared_ptr<ClassifierTerms> terms) {
    this->terms = terms;
}
