#include <unordered_set>
#include "hashing.h"

cv::Vec3d Hashing::extractSurfaceNormal(cv::Mat &src, cv::Point c) {
    float dzdx = (src.at<float>(c.y, c.x + 1) - src.at<float>(c.y, c.x - 1)) / 2.0f;
    float dzdy = (src.at<float>(c.y + 1, c.x) - src.at<float>(c.y - 1, c.x)) / 2.0f;
    cv::Vec3f d(-dzdy, -dzdx, 1.0f);

    return cv::normalize(d);
}

int Hashing::quantizeSurfaceNormals(cv::Vec3f normal) {
    // Normal should not be === 0, always at least z > 2
    if (normal[0] < 0 && normal[1] < 0 && normal[2] < 0) {
        std::cout << normal << std::endl;
    }

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

int Hashing::quantizeDepths(float depth) {
    // TODO WRONG - relative depths can have <-65k, +65k> values
    if (depth <= 13107) {
        return 0; // 1. bin
    } else if (depth > 13107 && depth <= 26214) {
        return 1; // 2. bin
    } else if (depth > 26214 && depth <= 39321) {
        return 2; // 3. bin
    } else if (depth > 39321 && depth <= 52428) {
        return 3; // 4. bin
    } else {
        return 4; // 5. bin
    }
}

void Hashing::train(std::vector<TemplateGroup> &groups) {
    // Checks
    assert(groups.size() > 0);

    // Init hash tables
    std::vector<HashTable> hashTables(100);
    for (int i = 0; i < 100; ++i) {
        // Init hash table
        HashTable hashTable;
        // TODO - make sure triplets are different for each table
        hashTable.triplet = Triplet::createRandomTriplet(this->featurePointsGrid); // one per hash table

        for (auto &group : groups) {
            for (auto &t : group.templates) {
                // Generate triplet params
                float stepX = t.srcDepth.cols / static_cast<float>(this->featurePointsGrid.width);
                float stepY = t.srcDepth.cols / static_cast<float>(this->featurePointsGrid.height);
                float offsetX = stepX / 2.0f;
                float offsetY = stepY / 2.0f;

                // Get triplet points
                cv::Point p1 = hashTable.triplet.getP1Coords(offsetX, stepX, offsetY, stepY);
                cv::Point p2 = hashTable.triplet.getP2Coords(offsetX, stepX, offsetY, stepY);
                cv::Point p3 = hashTable.triplet.getP3Coords(offsetX, stepX, offsetY, stepY);

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

    // Print hash tables
    for (auto &&hasht : hashTables) {
        std::cout << hasht << std::endl;
    }
}

const cv::Size Hashing::getFeaturePointsGrid() {
    return this->featurePointsGrid;
}

const std::vector<HashTable> &Hashing::getHashTables() {
    return this->hashTables;
}

void Hashing::setFeaturePointsGrid(cv::Size featurePointsGrid) {
    this->featurePointsGrid = featurePointsGrid;
}

void Hashing::setHashTables(const std::vector<HashTable> &hashTables) {
    this->hashTables = hashTables;
}
