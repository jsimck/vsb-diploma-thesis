#include <unordered_set>
#include "hasher.h"
#include "../utils/utils.h"
#include "../utils/timer.h"

const int Hasher::IMG_16BIT_MAX = 65535; // <0, 65535> => 65536 values
const int Hasher::IMG_16BIT_RANGE = (IMG_16BIT_MAX * 2) + 1; // <-65535, 65535> => 131070 values + (one zero) == 131070

cv::Vec3f Hasher::surfaceNormal(const cv::Mat &src, const cv::Point c) {
    assert(!src.empty());
    assert(src.type() == 5); // CV_32FC1

    float dzdx = (src.at<float>(c.y, c.x + 1) - src.at<float>(c.y, c.x - 1)) / 2.0f;
    float dzdy = (src.at<float>(c.y + 1, c.x) - src.at<float>(c.y - 1, c.x)) / 2.0f;
    cv::Vec3f d(-dzdy, -dzdx, 1.0f);

    return cv::normalize(d);
}

cv::Vec2i Hasher::relativeDepths(const cv::Mat &src, const cv::Point c, const cv::Point p1, const cv::Point p2) {
    assert(!src.empty());
    assert(src.type() == 5); // CV_32FC1

    return cv::Vec2i(
        static_cast<int>(src.at<float>(p1) - src.at<float>(c)),
        static_cast<int>(src.at<float>(p2) - src.at<float>(c))
    );
}

uchar Hasher::quantizeSurfaceNormal(cv::Vec3f normal) {
    // Normal z coordinate should not be < 0
    assert(normal[2] >= 0);

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

    uchar minIndex = 0;
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

uchar Hasher::quantizeDepth(float depth) {
    // Depth should have max value of <-65536, +65536>
    assert(depth >= -IMG_16BIT_MAX && depth <= IMG_16BIT_MAX);
    assert(binRanges.size() > 0);

    // Loop through histogram ranges and return quantized index
    for (int i = 0; i < binRanges.size(); i++) {
        if (binRanges[i].start >= depth && depth < binRanges[i].end) {
            return static_cast<uchar>(i);
        }
    }

    assert(binRanges.size() == binCount);

    // If value is IMG_16BIT_MAX it belongs to last bin
    return static_cast<uchar>(binRanges.size() - 1);
}

void Hasher::generateTriplets(std::vector<HashTable> &tables) {
    // Generate triplets
    for (int i = 0; i < tablesCount; ++i) {
        HashTable h(Triplet::create(grid, maxDistance));
        tables.push_back(h);
    }

    // TODO - joint entropy of 5k triplets, instead of 100 random triplets
    // Check for unique triplets, and regenerate duplicates
    bool duplicate;
    do {
        duplicate = false;
        for (int i = 0; i < tables.size(); ++i) {
            for (int j = 0; j < tables.size(); ++j) {
                // Don't compare same triplets
                if (i == j) continue;
                if (tables[i].triplet == tables[j].triplet) {
                    // Duplicate, generate new triplet
                    duplicate = true;
                    tables[j].triplet = Triplet::create(grid);
                }
            }
        }
    } while (duplicate);
}

void Hasher::computeBinRanges(unsigned long sum, unsigned long *values) {
    // Checks
    assert(sum > 0);
    assert(values != nullptr);

    std::vector<cv::Range> ranges;
    unsigned long tmpBinCount = 0;
    int rangeStart = 0, binsCreated = 0;

    // Calculate approximate size of every bin
    const unsigned long binSize = sum / binCount;

    // Loop trough histogram and determine ranges
    for (int i = 0; i < IMG_16BIT_RANGE; i++) {
        if (values[i] > 0) {
            tmpBinCount += values[i];
        }

        // Check if bin is full -> save range and start again
        if (tmpBinCount >= binSize) {
            tmpBinCount = 0;
            ranges.push_back(cv::Range(rangeStart - IMG_16BIT_MAX, i - IMG_16BIT_MAX));
            rangeStart = i;
            binsCreated++;
            i--;
        }

        // Define last range from current i to end [IMG_16BIT_MAX]
        if ((binsCreated + 1) >= binSize) {
            ranges.push_back(cv::Range(i - IMG_16BIT_MAX + 1, IMG_16BIT_MAX));
            break;
        }
    }

    // Print results
    std::cout << "DONE! Approximate " << binSize << " values per bin" << std::endl;
    for (int i = 0; i < ranges.size(); i++) {
        std::cout << "       |_ " << i << ". <" << ranges[i].start << ", " << ranges[i].end << (i + 1 == ranges.size() ? ">" : ")") << std::endl;
    }

    // Set histogram ranges
    setBinRanges(ranges);
}

void Hasher::initializeBinRanges(const std::vector<Group> &groups, std::vector<HashTable> &tables, const DataSetInfo &info) {
    // Histogram values <-65535, +65535>
    unsigned long histogramValues[IMG_16BIT_RANGE];
    unsigned long histogramSum = 0;

    // Reset histogram
    for (int i = 0; i < IMG_16BIT_RANGE; i++) {
        histogramValues[i] = 0;
    }

    // Fill histogram with generated triplets and relative depths
    for (auto &table : tables) {
        for (auto &group : groups) {
            for (auto &t : group.templates) {
                // Checks
                assert(!t.srcDepth.empty());

                // Offset for the triplet grid
                cv::Point gridOffset(
                    t.objBB.tl().x - (info.maxTemplate.width - t.objBB.width) / 2,
                    t.objBB.tl().y - (info.maxTemplate.height - t.objBB.height) / 2
                );

                // Absolute triplet points
                TripletParams params = Triplet::getParams(info.maxTemplate.width, info.maxTemplate.height, grid, gridOffset.x, gridOffset.y);
                cv::Point c = table.triplet.getCenter(params);
                cv::Point p1 = table.triplet.getP1(params);
                cv::Point p2 = table.triplet.getP2(params);

                // Check if we're not out of bounds
                assert(c.x >= 0 && c.x < t.srcDepth.cols);
                assert(c.y >= 0 && c.y < t.srcDepth.rows);
                assert(p1.x >= 0 && p1.x < t.srcDepth.cols);
                assert(p1.y >= 0 && p1.y < t.srcDepth.rows);
                assert(p2.x >= 0 && p2.x < t.srcDepth.cols);
                assert(p2.y >= 0 && p2.y < t.srcDepth.rows);

                // Relative depths
                cv::Vec2i d = relativeDepths(t.srcDepth, c, p1, p2);

                // Offset depths from <-65535, +65535> to <0, 131070> array histogram counter
                histogramValues[d[0] + IMG_16BIT_MAX] += 1;
                histogramValues[d[1] + IMG_16BIT_MAX] += 1;
                histogramSum += 2; // Total number of histogram values
            }
        }
    }

    // Calculate ranges from retrieved histogram
    computeBinRanges(histogramSum, histogramValues);
}

void Hasher::initialize(const std::vector<Group> &groups, std::vector<HashTable> &tables, const DataSetInfo &info) {
    // Init hash tables
    tables.reserve(tablesCount);
    generateTriplets(tables);

    // Calculate bin ranges for depth quantization
    std::cout << "  |_ Calculating depths bin ranges... ";
    initializeBinRanges(groups, tables, info);
}

void Hasher::train(std::vector<Group> &groups, std::vector<HashTable> &tables, const DataSetInfo &info) {
    // Checks
    assert(groups.size() > 0);
    assert(tablesCount > 0);
    assert(grid.width > 0);
    assert(grid.height > 0);
    assert(info.maxTemplate.area() > 0);

    // Prepare hash tables and histogram bin ranges
    initialize(groups, tables, info);

    // Fill hash tables with templates and keys quantized from measured values
    for (auto &table : tables) {
        for (auto &group : groups) {
            for (auto &t : group.templates) {
                // Checks
                assert(!t.srcDepth.empty());

                // Calculate offset for the triplet grid
                cv::Point gridOffset(
                    t.objBB.tl().x - (info.maxTemplate.width - t.objBB.width) / 2,
                    t.objBB.tl().y - (info.maxTemplate.height - t.objBB.height) / 2
                );

                // Get triplet points
                TripletParams coordParams = Triplet::getParams(info.maxTemplate.width, info.maxTemplate.height, grid, gridOffset.x, gridOffset.y);
                cv::Point c = table.triplet.getCenter(coordParams);
                cv::Point p1 = table.triplet.getP1(coordParams);
                cv::Point p2 = table.triplet.getP2(coordParams);

#ifndef NDEBUG
//                // Visualize triplets
//                cv::Mat triplet = t.srcGray.clone();
//                cv::cvtColor(triplet, triplet, CV_GRAY2BGR);
//                cv::rectangle(triplet, gridOffset, cv::Point(gridOffset.x + info.maxTemplate.width, gridOffset.y + info.maxTemplate.height), cv::Scalar(0, 255, 0));
//                cv::rectangle(triplet, t.objBB.tl(), t.objBB.br(), cv::Scalar(0, 0, 255));
//                for (int i = 0; i < 12; i++) {
//                    Triplet tpl(cv::Point(i,0), cv::Point(i,1), cv::Point(i,2));
//                    cv::circle(triplet, tpl.getPoint(0, coordParams), 2, cv::Scalar(0, 200, 0), -1);
//                    cv::circle(triplet, tpl.getPoint(1, coordParams), 2, cv::Scalar(0, 200, 0), -1);
//                    cv::circle(triplet, tpl.getPoint(2, coordParams), 2, cv::Scalar(0, 200, 0), -1);
//                    Triplet tMax(cv::Point(i,3), cv::Point(i,4), cv::Point(i,5));
//                    cv::circle(triplet, tMax.getPoint(0, coordParams), 2, cv::Scalar(0, 200, 0), -1);
//                    cv::circle(triplet, tMax.getPoint(1, coordParams), 2, cv::Scalar(0, 200, 0), -1);
//                    cv::circle(triplet, tMax.getPoint(2, coordParams), 2, cv::Scalar(0, 200, 0), -1);
//                    Triplet t3(cv::Point(i,6), cv::Point(i,7), cv::Point(i,8));
//                    cv::circle(triplet, t3.getPoint(0, coordParams), 2, cv::Scalar(0, 200, 0), -1);
//                    cv::circle(triplet, t3.getPoint(1, coordParams), 2, cv::Scalar(0, 200, 0), -1);
//                    cv::circle(triplet, t3.getPoint(2, coordParams), 2, cv::Scalar(0, 200, 0), -1);
//                    Triplet t4(cv::Point(i,9), cv::Point(i,10), cv::Point(i,11));
//                    cv::circle(triplet, t4.getPoint(0, coordParams), 2, cv::Scalar(0, 200, 0), -1);
//                    cv::circle(triplet, t4.getPoint(1, coordParams), 2, cv::Scalar(0, 200, 0), -1);
//                    cv::circle(triplet, t4.getPoint(2, coordParams), 2, cv::Scalar(0, 200, 0), -1);
//
//                    cv::circle(triplet, c, 2, cv::Scalar(0, 0, 255), -1);
//                    cv::circle(triplet, p1, 2, cv::Scalar(0, 0, 255), -1);
//                    cv::circle(triplet, p2, 2, cv::Scalar(0, 0, 255), -1);
//                    cv::line(triplet, c, p1, cv::Scalar(0, 0, 255));
//                    cv::line(triplet, c, p2, cv::Scalar(0, 0, 255));
//                }
//                cv::imshow("Classifier::Hash table triplet", triplet);
//                cv::waitKey(0);
#endif

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
                    quantizeDepth(d[0]),
                    quantizeDepth(d[1]),
                    quantizeSurfaceNormal(surfaceNormal(t.srcDepth, c)),
                    quantizeSurfaceNormal(surfaceNormal(t.srcDepth, p1)),
                    quantizeSurfaceNormal(surfaceNormal(t.srcDepth, p2))
                );

                // Check if key exists, if not initialize it
                table.pushUnique(key, t);
            }
        }
    }

#ifndef NDEBUG
//    // Visualize triplets
//    cv::Mat triplet = cv::Mat::zeros(400, 400, CV_32FC3), triplets = cv::Mat::zeros(400, 400, CV_32FC3);
//    tables[0].triplet.visualize(triplet, getGrid()); // generate grid
//    tables[0].triplet.visualize(triplets, getGrid()); // generate grid
//    cv::imshow("Classifier::Hash table triplets", triplets);
//    cv::imshow("Classifier::Hash table triplet", triplet);
//    cv::waitKey(0);
//
//    for (auto &table : tables) {
//        std::cout << table << std::endl;
//        triplet = cv::Mat::zeros(400, 400, CV_32FC3);
//        table.triplet.visualize(triplets, getGrid(), false);
//        table.triplet.visualize(triplet, getGrid(), true);
//        cv::imshow("Classifier::Hash table triplets", triplets);
//        cv::imshow("Classifier::Hash table triplet", triplet);
//        cv::waitKey(1);
//    }
#endif
}

void Hasher::verifyCandidates(const cv::Mat &sceneDepth, std::vector<HashTable> &tables, std::vector<Window> &windows, const DataSetInfo &info) {
    // Checks
    assert(!sceneDepth.empty());
    assert(windows.size() > 0);
    assert(tables.size() > 0);
    assert(info.maxTemplate.area() > 0);

    std::vector<Template *> usedTemplates;
    std::vector<int> emptyIndexes;

    for (int i = 0; i < windows.size(); ++i) {
        // Calculate new rectangle that's placed over the center of image from sliding window with the maxTemplate size
        cv::Point gridOffset((windows[i].width / 2 + windows[i].tl().x) - (info.maxTemplate.width / 2), (windows[i].height / 2 + windows[i].tl().y) - info.maxTemplate.height / 2);

        for (auto &table : tables) {
            // Get triplet points
            TripletParams coordParams = Triplet::getParams(info.maxTemplate.width, info.maxTemplate.height, grid, gridOffset.x, gridOffset.y);
            cv::Point c = table.triplet.getCenter(coordParams);
            cv::Point p1 = table.triplet.getP1(coordParams);
            cv::Point p2 = table.triplet.getP2(coordParams);

#ifndef NDEBUG
//            // Visualize triplets
//            cv::Mat triplet = sceneDepth.clone();
//            cv::cvtColor(triplet, triplet, CV_GRAY2BGR);
//            cv::rectangle(triplet, gridOffset, cv::Point(gridOffset.x + info.maxTemplate.width, gridOffset.y + info.maxTemplate.height), cv::Scalar(0, 255, 0));
//            cv::circle(triplet, c, 2, cv::Scalar(0, 0, 255), -1);
//            cv::circle(triplet, p1, 2, cv::Scalar(0, 0, 255), -1);
//            cv::circle(triplet, p2, 2, cv::Scalar(0, 0, 255), -1);
//            cv::line(triplet, c, p1, cv::Scalar(0, 0, 255));
//            cv::line(triplet, c, p2, cv::Scalar(0, 0, 255));
//            cv::imshow("Classifier::Hash table triplet", triplet);
//            cv::waitKey(1);
#endif

            // If any point of triplet is out of scene boundaries, ignore it to not get false data
            if ((c.x < 0 || c.x >= sceneDepth.cols || c.y < 0 || c.y >= sceneDepth.rows) ||
                (p1.x < 0 || p1.x >= sceneDepth.cols || p1.y < 0 || p1.y >= sceneDepth.rows) ||
                (p2.x < 0 || p2.x >= sceneDepth.cols || p2.y < 0 || p2.y >= sceneDepth.rows)) continue;

            // Relative depths
            cv::Vec2i d = relativeDepths(sceneDepth, c, p1, p2);

            // Generate hash key
            HashKey key(
                quantizeDepth(d[0]),
                quantizeDepth(d[1]),
                quantizeSurfaceNormal(surfaceNormal(sceneDepth, c)),
                quantizeSurfaceNormal(surfaceNormal(sceneDepth, p1)),
                quantizeSurfaceNormal(surfaceNormal(sceneDepth, p2))
            );

            // Vote for each template in hash table at specific key and push unique to window candidates
            for (auto &entry : table.templates[key]) {
                entry->vote();

                // automatically pushes only unique templates with minimum of v minVotes and up to N of templates
                windows[i].pushUnique(entry, tablesCount, minVotes);
                usedTemplates.push_back(entry);
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
            emptyIndexes.push_back(i);
        }
    }

    // Remove empty windows
    utils::removeIndex<Window>(windows, emptyIndexes);
    std::cout << "  |_ Number of windows pass to next stage: " << windows.size() << std::endl;
}

const cv::Size Hasher::getGrid() {
    return grid;
}

uint Hasher::getTablesCount() const {
    return tablesCount;
}

const std::vector<cv::Range> &Hasher::getBinRanges() const {
    return binRanges;
}

uint Hasher::getBinCount() const {
    return binCount;
}

int Hasher::getMinVotes() const {
    return minVotes;
}

void Hasher::setMinVotes(int votes) {
    this->minVotes = votes;
}

uint Hasher::getMaxDistance() const {
    return maxDistance;
}

void Hasher::setGrid(cv::Size grid) {
    assert(grid.height > 0 && grid.width > 0);
    this->grid = grid;
}

void Hasher::setTablesCount(uint tablesCount) {
    assert(tablesCount > 0);
    this->tablesCount = tablesCount;
}

void Hasher::setBinRanges(const std::vector<cv::Range> &binRanges) {
    assert(binRanges.size() == binCount);
    this->binRanges = binRanges;
}

void Hasher::setBinCount(uint binCount) {
    assert(binCount > 0);
    assert(binCount < 8); // Max due to hasher function
    this->binCount = binCount;
}

void Hasher::setMaxDistance(uint maxDistance) {
    assert(maxDistance > 1);
    this->maxDistance = maxDistance;
}
