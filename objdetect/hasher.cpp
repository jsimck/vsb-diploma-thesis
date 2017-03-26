#include <unordered_set>
#include "hasher.h"
#include "matching.h"

cv::Vec3d Hasher::extractSurfaceNormal(const cv::Mat &src, cv::Point c) {
    // Checks
    assert(!src.empty());

    float dzdx = (src.at<float>(c.y, c.x + 1) - src.at<float>(c.y, c.x - 1)) / 2.0f;
    float dzdy = (src.at<float>(c.y + 1, c.x) - src.at<float>(c.y - 1, c.x)) / 2.0f;
    cv::Vec3f d(-dzdy, -dzdx, 1.0f);

    return cv::normalize(d);
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
    assert(depth >= -65536 && depth <= 65536);

    // TODO WRONG - relative depths can have <-65k, +65k> values
    if (depth >= -500 && depth <= 500) {
        return 0; // 1. bin
    } else if (depth >= -1000 && depth <= 1000) {
        return 1; // 2. bin
    } else if (depth >= -3000 && depth <= 3000) {
        return 2; // 3. bin
    } else if (depth >= -6000 && depth <= 6000) {
        return 3; // 4. bin
    } else {
        return 4; // 5. bin
    }
}

void Hasher::train(const std::vector<TemplateGroup> &groups, std::vector<HashTable> &hashTables, unsigned int numberOfHashTables) {
    // Checks
    assert(groups.size() > 0);
    assert(numberOfHashTables > 0);

    // Init hash tables
    hashTables.reserve(numberOfHashTables);

    // Generate triplets
    for (int i = 0; i < 100; ++i) {
        // Init hash table
        HashTable hashTable;
        // TODO - make sure triplets are different for each table
        hashTable.triplet = Triplet::createRandomTriplet(featurePointsGrid); // one per hash table

        for (auto &group : groups) {
            for (auto &t : group.templates) {
                // Generate triplet params
                float stepX = t.srcDepth.cols / static_cast<float>(featurePointsGrid.width);
                float stepY = t.srcDepth.rows / static_cast<float>(featurePointsGrid.height);
                float offsetX = stepX / 2.0f;
                float offsetY = stepY / 2.0f;

                // Get triplet points
                cv::Point p1 = hashTable.triplet.getP1Coords(offsetX, stepX, offsetY, stepY);
                cv::Point p2 = hashTable.triplet.getP2Coords(offsetX, stepX, offsetY, stepY);
                cv::Point p3 = hashTable.triplet.getP3Coords(offsetX, stepX, offsetY, stepY);

                // Check if we're not out of bounds
                assert(p1.x >= 0 && p1.x < t.srcDepth.cols);
                assert(p1.y >= 0 && p1.y < t.srcDepth.rows);
                assert(p2.x >= 0 && p2.x < t.srcDepth.cols);
                assert(p2.y >= 0 && p2.y < t.srcDepth.rows);
                assert(p3.x >= 0 && p3.x < t.srcDepth.cols);
                assert(p3.y >= 0 && p3.y < t.srcDepth.rows);

                // Relative depths
                float d1 = t.srcDepth.at<float>(p2) - t.srcDepth.at<float>(p1);
                float d2 = t.srcDepth.at<float>(p3) - t.srcDepth.at<float>(p1);

                // Generate hash key
                HashKey key(
                    quantizeDepths(d1),
                    quantizeDepths(d2),
                    quantizeSurfaceNormals(extractSurfaceNormal(t.srcDepth, p1)),
                    quantizeSurfaceNormals(extractSurfaceNormal(t.srcDepth, p2)),
                    quantizeSurfaceNormals(extractSurfaceNormal(t.srcDepth, p3))
                );

                // TODO - maybe can be done better
                // Check if key exists, if not initialize it
                if (hashTable.templates.find(key) == hashTable.templates.end()) {
                    std::vector<Template> hashTemplates;
                    hashTable.templates[key] = hashTemplates;
                }

                // Get reference to entry and check for duplicates, if it passes, insert
                auto &entry = hashTable.templates.at(key);
                if (std::find(entry.begin(), entry.end(), t) == entry.end()) {
                    entry.push_back(t);
                }
            }
        }

        // Push hash table to list
        hashTables.push_back(hashTable);
    }
}

void Hasher::verifyTemplateCandidates(const cv::Mat &sceneGrayscale, cv::Rect &objectnessROI, std::vector<HashTable> &hashTables, std::vector<TemplateGroup> &groups) {
    // Sliding window - fixed WIDTH
    cv::Size w(120, 120);
    cv::Mat s = sceneGrayscale.clone();
    int step = 5;
    
    // Passed templates
    std::vector<std::vector<Template*>> windowTemplates; // Each window can receive up to N templates
    const int v = 3;
    int templatesPassedCount = 0;

    for (int y = objectnessROI.y; y < (objectnessROI.y + objectnessROI.height) - w.height; y += step) {
        for (int x = objectnessROI.x; x < (objectnessROI.x + objectnessROI.width) - w.width; x += step) {
            s = sceneGrayscale.clone();
            cv::rectangle(s, cv::Point(x, y), cv::Point(x + w.width, y + w.height), cv::Scalar(1.0f));

            // Init templates for this window
            std::vector<Template*> templates;

            for (auto &hashTable : hashTables) {
                // Generate triplet params
                float stepX = w.width / static_cast<float>(featurePointsGrid.width);
                float stepY = w.height / static_cast<float>(featurePointsGrid.height);
                float offsetX = stepX / 2.0f;
                float offsetY = stepY / 2.0f;

                // Get triplet points
                cv::Point p1 = hashTable.triplet.getP1Coords(offsetX, stepX, offsetY, stepY);
                cv::Point p2 = hashTable.triplet.getP2Coords(offsetX, stepX, offsetY, stepY);
                cv::Point p3 = hashTable.triplet.getP3Coords(offsetX, stepX, offsetY, stepY);

                // Check if we're not out of bounds
                assert(p1.x >= 0 && p1.x < s.cols);
                assert(p1.y >= 0 && p1.y < s.rows);
                assert(p2.x >= 0 && p2.x < s.cols);
                assert(p2.y >= 0 && p2.y < s.rows);
                assert(p3.x >= 0 && p3.x < s.cols);
                assert(p3.y >= 0 && p3.y < s.rows);

                // Relative depths
                float d1 = s.at<float>(p2) - s.at<float>(p1);
                float d2 = s.at<float>(p3) - s.at<float>(p1);

                // Generate hash key
                HashKey key(
                    quantizeDepths(d1),
                    quantizeDepths(d2),
                    quantizeSurfaceNormals(extractSurfaceNormal(s, p1)),
                    quantizeSurfaceNormals(extractSurfaceNormal(s, p2)),
                    quantizeSurfaceNormals(extractSurfaceNormal(s, p3))
                );

                // Up votes
                // TODO - we should probably use only up to N templates with highest votes
                for (auto &entry : hashTable.templates[key]) {
                    entry.voteUp();
//                    if (entry.votes >= v) {
//                        // Check if it is not yet in templates
//                        if (std::find(templates.begin(), templates.end(), &entry) == templates.end()) {
//                            templates.push_back(&entry);
//                        }
//                    }
                }
            }

            for (auto &&group : groups) {
                for (auto &&gt : group.templates) {
                    std::cout << gt.votes << std::endl;
                }
            }


            std::cout << "Templates size for current window: " << templates.size() << std::endl;

            // Sort by votes and
            std::sort(templates.begin(), templates.end(), [](const Template* lhs, const Template *rhs) {
                return lhs->votes < rhs->votes;
            });

            for (auto && t : templates) {
                std::cout << t->votes << std::endl;
            }

            // Reset votes for templates for each new sliding window
            for (auto &&t : templates) {
                t->resetVotes();
            }

            // Push filtered templates for this window to new array
            windowTemplates.push_back(templates);
            templatesPassedCount += templates.size();
        }
    }

    // Print filtered size
    std::cout << "Total number of templates passed: " << templatesPassedCount << std::endl;
    std::cout << "Total number of windows: " << windowTemplates.size() << std::endl;

    // Todo pass window locations with template candidates to template matching
}

const cv::Size Hasher::getFeaturePointsGrid() {
    return featurePointsGrid;
}

void Hasher::setFeaturePointsGrid(cv::Size featurePointsGrid) {
    assert(featurePointsGrid.height > 0 && featurePointsGrid.width > 0);
    this->featurePointsGrid = featurePointsGrid;
}
