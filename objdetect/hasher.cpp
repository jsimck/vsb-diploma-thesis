#include <unordered_set>
#include "hasher.h"
#include "../utils/utils.h"

const int Hasher::IMG_16BIT_VALUE_MAX = 65535; // <0, 65535> => 65536 values
const int Hasher::IMG_16BIT_VALUES_RANGE = (IMG_16BIT_VALUE_MAX * 2) + 1; // <-65535, 65535> => 131071 values + (one zero)

cv::Vec3f Hasher::extractSurfaceNormal(const cv::Mat &src, const cv::Point c) {
    // Checks
    assert(!src.empty());

    float dzdx = (src.at<float>(c.y, c.x + 1) - src.at<float>(c.y, c.x - 1)) / 2.0f;
    float dzdy = (src.at<float>(c.y + 1, c.x) - src.at<float>(c.y - 1, c.x)) / 2.0f;
    cv::Vec3f d(-dzdy, -dzdx, 1.0f);

    return cv::normalize(d);
}

cv::Vec2i Hasher::extractRelativeDepths(const cv::Mat &src, const cv::Point c, const cv::Point p1, const cv::Point p2) {
    return cv::Vec2i(
        static_cast<int>(src.at<float>(p1) - src.at<float>(c)),
        static_cast<int>(src.at<float>(p2) - src.at<float>(c))
    );
}

int Hasher::quantizeSurfaceNormals(cv::Vec3f normal) {
    // Normal z coordinate should not be > 0
    assert(normal[2] >= 0);

    // For quantization of surface normals into 8 bins
    // in our case z is always positive, that's why we're using
    // 8 octants in top half of sphere only
    cv::Vec3f octantsNormals[8] = {
        cv::Vec3f(1.0f, 0, 1.0f), // 0. octant
        cv::Vec3f(1.0f, 1.0f, 1.0f), // 1. octant
        cv::Vec3f(0, 1.0f, 1.0f), // 2. octant
        cv::Vec3f(-1.0f, 1.0f, 1.0f), // 3. octant
        cv::Vec3f(-1.0f, 0.0f, 1.0f), // 4. octant
        cv::Vec3f(-1.0f, -1.0f, 1.0f), // 5. octant
        cv::Vec3f(0, -1.0f, 1.0f), // 6. octant
        cv::Vec3f(1.0f, -1.0f, 1.0f), // 7. octant
    };

    int minIndex = -1;
    float maxDot = 0, dot;
    for (int i = 0; i < 8; i++) {
        // By doing dot product between octant normals and calculated normal
        // we can find maximum -> index of octant where the vector belongs to
        cv::Vec3f octaNormal = cv::normalize(octantsNormals[i]);
        dot = normal.dot(octaNormal);

        if (dot > maxDot) {
            maxDot = dot;
            minIndex = i;
        }
    }

    // Index should in interval <0,7>
    assert(minIndex >= 0 && minIndex < 8);

    return minIndex;
}

int Hasher::quantizeDepths(float depth) {
    // Depth should have max value of <-65536, +65536>
    assert(depth >= -IMG_16BIT_VALUE_MAX && depth <= IMG_16BIT_VALUE_MAX);
    assert(histogramBinRanges.size() > 0);

    // Loop through histogram ranges and return quantized index
    for (int i = 0; i < histogramBinRanges.size(); i++) {
        if (histogramBinRanges[i].start >= depth && depth < histogramBinRanges[i].end) {
            return i;
        }
    }

    // If value is IMG_16BIT_VALUE_MAX it belongs to last bin
    return (int) histogramBinRanges.size() - 1;
}

void Hasher::generateTriplets(std::vector<HashTable> &hashTables) {
    // Generate triplets
    for (int i = 0; i < hashTableCount; ++i) {
        HashTable h(Triplet::createRandomTriplet(referencePointsGrid, maxTripletDistance));
        hashTables.push_back(h);
    }

    // TODO - joint entropy of 5k triplets, instead of 100 random triplets
    // Check for unique triplets, and regenerate duplicates
    bool duplicate;
    do {
        duplicate = false;
        for (int i = 0; i < hashTables.size(); ++i) {
            for (int j = 0; j < hashTables.size(); ++j) {
                // Don't compare same triplets
                if (i == j) continue;
                if (hashTables[i].triplet == hashTables[j].triplet) {
                    // Duplicate generate new triplet
                    duplicate = true;
                    hashTables[j].triplet = Triplet::createRandomTriplet(referencePointsGrid);
                }
            }
        }
    } while (duplicate);
}

void Hasher::calculateDepthHistogramRanges(unsigned long histogramSum, unsigned long *histogramValues) {
    // Checks
    assert(histogramSum > 0);

    std::vector<cv::Range> ranges;
    unsigned long tmpBinCount = 0;
    int rangeStart = 0, binsCreated = 0;

    // Calculate approximate size of every bin
    const unsigned long binCount = histogramSum / histogramBinCount;

    // Loop trough histogram and determine ranges
    for (int i = 0; i < IMG_16BIT_VALUES_RANGE; i++) {
        if (histogramValues[i] > 0) {
            tmpBinCount += histogramValues[i];
        }

        // Check if we filled one bin, if yes, save range and reset tmpBinCount
        if (tmpBinCount >= binCount) {
            tmpBinCount = 0;
            ranges.push_back(cv::Range(rangeStart - IMG_16BIT_VALUE_MAX, i - IMG_16BIT_VALUE_MAX));
            rangeStart = i;
            binsCreated++;
            i--;
        }

        // Define last range from current i to end [IMG_16BIT_VALUE_MAX]
        if ((binsCreated + 1) >= histogramBinCount) {
            ranges.push_back(cv::Range(i - IMG_16BIT_VALUE_MAX + 1, IMG_16BIT_VALUE_MAX));
            break;
        }
    }

    // Print results
    std::cout << "DONE! Approximate " << binCount << " values per bin" << std::endl;
    for (int i = 0; i < ranges.size(); i++) {
        std::cout << "       |_ " << i << ". <" << ranges[i].start << ", " << ranges[i].end << (i + 1 == ranges.size() ? ">" : ")") << std::endl;
    }

    // Set histogram ranges
    setHistogramBinRanges(ranges);
}

void Hasher::calculateDepthBinRanges(const std::vector<TemplateGroup> &groups, std::vector<HashTable> &hashTables) {
    // Histogram values <-65535, +65535> possible values
    unsigned long histogramValues[IMG_16BIT_VALUES_RANGE];
    unsigned long histogramSum = 0;

    // Reset histogram values
    for (int i = 0; i < IMG_16BIT_VALUES_RANGE; i++) {
        histogramValues[i] = 0;
    }

    // Calculate histogram values, using generated triplets and relative depths calculation
    for (auto &hashTable : hashTables) {
        for (auto &group : groups) {
            for (auto &t : group.templates) {
                // Checks
                assert(!t.srcDepth.empty());

                // Get triplet points
                TripletCoords coordParams = Triplet::getCoordParams(t.srcDepth.cols, t.srcDepth.rows, referencePointsGrid);
                cv::Point c = hashTable.triplet.getCenterCoords(coordParams);
                cv::Point p1 = hashTable.triplet.getP1Coords(coordParams);
                cv::Point p2 = hashTable.triplet.getP2Coords(coordParams);

                // Check if we're not out of bounds
                assert(c.x >= 0 && c.x < t.srcDepth.cols);
                assert(c.y >= 0 && c.y < t.srcDepth.rows);
                assert(p1.x >= 0 && p1.x < t.srcDepth.cols);
                assert(p1.y >= 0 && p1.y < t.srcDepth.rows);
                assert(p2.x >= 0 && p2.x < t.srcDepth.cols);
                assert(p2.y >= 0 && p2.y < t.srcDepth.rows);

                // Relative depths
                cv::Vec2i relativeDepths = extractRelativeDepths(t.srcDepth, c, p1, p2);

                // Add offset and count given values
                histogramValues[relativeDepths[0] + IMG_16BIT_VALUE_MAX] += 1;
                histogramValues[relativeDepths[1] + IMG_16BIT_VALUE_MAX] += 1;
                histogramSum += 2; // Add 2 to sum of histogram values
            }
        }
    }

    // Calculate ranges from retrieved data
    calculateDepthHistogramRanges(histogramSum, histogramValues);
}

void Hasher::initialize(const std::vector<TemplateGroup> &groups, std::vector<HashTable> &hashTables) {
    // Checks
    assert(groups.size() > 0);
    assert(hashTableCount > 0);
    assert(referencePointsGrid.width > 0);
    assert(referencePointsGrid.height > 0);

    // Init hash tables
    hashTables.reserve(hashTableCount);
    generateTriplets(hashTables);

    // Calculate ranges of depth bins for training
    std::cout << "  |_ Calculating depth bin ranges... ";
    calculateDepthBinRanges(groups, hashTables);
}

void Hasher::train(std::vector<TemplateGroup> &groups, std::vector<HashTable> &hashTables) {
    // Prepare hash tables and histogram bin ranges
    initialize(groups, hashTables);

    // Fill hash tables with templates and keys quantized from measured values
    for (auto &hashTable : hashTables) {
        for (auto &group : groups) {
            for (auto &t : group.templates) {
                // Checks
                assert(!t.srcDepth.empty());

                // Get triplet points
                TripletCoords coordParams = Triplet::getCoordParams(t.srcDepth.cols, t.srcDepth.rows, referencePointsGrid);
                cv::Point c = hashTable.triplet.getCenterCoords(coordParams);
                cv::Point p1 = hashTable.triplet.getP1Coords(coordParams);
                cv::Point p2 = hashTable.triplet.getP2Coords(coordParams);

                // Check if we're not out of bounds
                assert(c.x >= 0 && c.x < t.srcDepth.cols);
                assert(c.y >= 0 && c.y < t.srcDepth.rows);
                assert(p1.x >= 0 && p1.x < t.srcDepth.cols);
                assert(p1.y >= 0 && p1.y < t.srcDepth.rows);
                assert(p2.x >= 0 && p2.x < t.srcDepth.cols);
                assert(p2.y >= 0 && p2.y < t.srcDepth.rows);

                // Relative depths
                cv::Vec2i relativeDepths = extractRelativeDepths(t.srcDepth, c, p1, p2);

                // Generate hash key
                HashKey key(
                    quantizeDepths(relativeDepths[0]),
                    quantizeDepths(relativeDepths[1]),
                    quantizeSurfaceNormals(extractSurfaceNormal(t.srcDepth, c)),
                    quantizeSurfaceNormals(extractSurfaceNormal(t.srcDepth, p1)),
                    quantizeSurfaceNormals(extractSurfaceNormal(t.srcDepth, p2))
                );

                // Check if key exists, if not initialize it
                if (hashTable.templates.find(key) == hashTable.templates.end()) {
                    std::vector<Template *> hashTemplates;
                    hashTable.templates[key] = hashTemplates;
                }

                // Check for duplicates in hash table and push unique
                auto found = std::find_if(hashTable.templates[key].begin(), hashTable.templates[key].end(), [&t](const Template* tt) { return t == *tt; });
                if (found == hashTable.templates[key].end()) {
                    hashTable.templates[key].push_back(&t);
                }
            }
        }
    }

#ifndef NDEBUG
//    // Visualize triplets
//    cv::Mat triplet = cv::Mat::zeros(400, 400, CV_32FC3), triplets = cv::Mat::zeros(400, 400, CV_32FC3);
//    hashTables[0].triplet.visualize(triplet, getReferencePointsGrid()); // generate grid
//    hashTables[0].triplet.visualize(triplets, getReferencePointsGrid()); // generate grid
//    cv::imshow("Classifier::Hash table triplets", triplets);
//    cv::imshow("Classifier::Hash table triplet", triplet);
//    cv::waitKey(0);
//
//    for (auto &&table : hashTables) {
//        triplet = cv::Mat::zeros(400, 400, CV_32FC3);
//        table.triplet.visualize(triplets, getReferencePointsGrid(), false);
//        table.triplet.visualize(triplet, getReferencePointsGrid(), true);
//        cv::imshow("Classifier::Hash table triplets", triplets);
//        cv::imshow("Classifier::Hash table triplet", triplet);
//        cv::waitKey(1);
//    }
#endif
}

void Hasher::verifyTemplateCandidates(const cv::Mat &sceneDepth, std::vector<HashTable> &hashTables, std::vector<Window> &windows) {
    // Checks
    assert(!sceneDepth.empty());
    assert(windows.size() > 0);
    assert(hashTables.size() > 0);

    unsigned long reduced = 0;
    std::vector<Template *> usedTemplates;
    std::vector<int> emptyIndexes;

    for (int i = 0; i < windows.size(); ++i) {
        for (auto &table : hashTables) {
            // Get triplet points
            TripletCoords coordParams = Triplet::getCoordParams(windows[i].width, windows[i].height, referencePointsGrid, windows[i].tl().x, windows[i].tl().y);
            cv::Point c = table.triplet.getCenterCoords(coordParams);
            cv::Point p1 = table.triplet.getP1Coords(coordParams);
            cv::Point p2 = table.triplet.getP2Coords(coordParams);

            // Check if we're not out of bounds
            assert(c.x >= 0 && c.x < sceneDepth.cols);
            assert(c.y >= 0 && c.y < sceneDepth.rows);
            assert(p1.x >= 0 && p1.x < sceneDepth.cols);
            assert(p1.y >= 0 && p1.y < sceneDepth.rows);
            assert(p2.x >= 0 && p2.x < sceneDepth.cols);
            assert(p2.y >= 0 && p2.y < sceneDepth.rows);

            // Relative depths
            cv::Vec2i relativeDepths = extractRelativeDepths(sceneDepth, c, p1, p2);

            // Generate hash key
            HashKey key(
                quantizeDepths(relativeDepths[0]),
                quantizeDepths(relativeDepths[1]),
                quantizeSurfaceNormals(extractSurfaceNormal(sceneDepth, c)),
                quantizeSurfaceNormals(extractSurfaceNormal(sceneDepth, p1)),
                quantizeSurfaceNormals(extractSurfaceNormal(sceneDepth, p2))
            );

            // Vote for each template in hash table at specific key and push unique to window candidates
            for (auto &entry : table.templates[key]) {
                entry->voteUp();

                // automatically pushes only unique templates with minimum of v minVotesPerTemplate and up to N of templates
                windows[i].pushUnique(entry, hashTableCount, minVotesPerTemplate);
                usedTemplates.push_back(entry);
            }

            reduced += windows[i].candidatesSize();
        }

        // Reset minVotesPerTemplate for all used templates
        for (auto &&t : usedTemplates) {
            t->resetVotes();
        }

        // Clear used templates vector
        usedTemplates.clear();

        if (!windows[i].hasCandidates()) {
            emptyIndexes.push_back(i);
        }
    }

    // Remove empty windows
    utils::removeIndex<Window>(windows, emptyIndexes);

    std::cout << "  |_ Number of windows pass to next stage: " << windows.size() << std::endl;
    std::cout << "  |_ Total number of templates in windows reduced to approx: " << reduced / windows.size() << std::endl;
}

const cv::Size Hasher::getReferencePointsGrid() {
    return referencePointsGrid;
}

unsigned int Hasher::getHashTableCount() const {
    return hashTableCount;
}

const std::vector<cv::Range> &Hasher::getHistogramBinRanges() const {
    return histogramBinRanges;
}

unsigned int Hasher::getHistogramBinCount() const {
    return histogramBinCount;
}

int Hasher::getMinVotesPerTemplate() const {
    return minVotesPerTemplate;
}

void Hasher::setMinVotesPerTemplate(int votes) {
    this->minVotesPerTemplate = votes;
}

unsigned int Hasher::getMaxTripletDistance() const {
    return maxTripletDistance;
}

void Hasher::setReferencePointsGrid(cv::Size featurePointsGrid) {
    assert(featurePointsGrid.height > 0 && featurePointsGrid.width > 0);
    this->referencePointsGrid = featurePointsGrid;
}

void Hasher::setHashTableCount(unsigned int hashTableCount) {
    assert(hashTableCount > 0);
    this->hashTableCount = hashTableCount;
}

void Hasher::setHistogramBinRanges(const std::vector<cv::Range> &histogramBinRanges) {
    assert(histogramBinRanges.size() == 5);
    this->histogramBinRanges = histogramBinRanges;
}

void Hasher::setHistogramBinCount(unsigned int histogramBinCount) {
    assert(histogramBinCount > 0);
    assert(histogramBinCount < 8); // Max due to hasher function
    this->histogramBinCount = histogramBinCount;
}

void Hasher::setMaxTripletDistance(unsigned int maxTripletDistance) {
    assert(maxTripletDistance > 1);
    this->maxTripletDistance = maxTripletDistance;
}
