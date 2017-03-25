#include "hashing.h"

cv::Vec3d Hashing::extractSurfaceNormal(cv::Mat &src, cv::Point c) {
    float dzdx = (src.at<float>(c.y, c.x + 1) - src.at<float>(c.y, c.x - 1)) / 2.0f;
    float dzdy = (src.at<float>(c.y + 1, c.x) - src.at<float>(c.y - 1, c.x)) / 2.0f;
    cv::Vec3f d(-dzdy, -dzdx, 1.0f);

    return cv::normalize(d);
}

std::vector<Triplet> Hashing::generateTripletsSubset(int k) {
    assert(k > 0);
    std::vector<Triplet> triplets(k);

    for (int i = 0; i < k; ++i) {
        triplets.push_back(Triplet::createRandomTriplet(this->featurePointsGrid));
    }

    return triplets;
}

int Hashing::quantizeSurfaceNormals(cv::Vec3f normal) {
    // Normal should not be === 0, always at least z > 2
    assert(normal[0] >= 0 || normal[1] >= 0 || normal[2] >= 0);

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

    int minIndex = 0;
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

    // Init hash table
    HashTable hashTable;
    hashTable.triplets = generateTripletsSubset();

    for (auto &group : groups) {
        for (auto &t : group.templates) {
            // Generate triplet params
            float stepX = t.srcDepth.cols / static_cast<float>(this->featurePointsGrid.width);
            float stepY = t.srcDepth.cols / static_cast<float>(this->featurePointsGrid.height);
            float offsetX = stepX / 2.0f;
            float offsetY = stepY / 2.0f;

            // Assign this template to hash table key
            for (auto &triplet : hashTable.triplets) {

            }
        }
    }

    // vizualization of triplets
//    for (int i = 0; i < 100; ++i) {
//
//        // Triplet points
//        cv::Point p1 = triplets[i].getP1Coords(offsetX, stepX, offsetY, stepY);
//        cv::Point p2 = triplets[i].getCoords(2 ,offsetX, stepX, offsetY, stepY);
//        cv::Point p3 = triplets[i].getP3Coords(offsetX, stepX, offsetY, stepY);
//
//        // Relative depths
//        float d1 = t.at<float>(p2) - t.at<float>(p1);
//        float d2 = t.at<float>(p3) - t.at<float>(p1);
//
//        std::cout << d1 << std::endl;
//        std::cout << d2 << std::endl;
//
//        HashKey key(
//            quantizeDepths(d1),
//            quantizeDepths(d2),
//            quantizeSurfaceNormals(extractSurfaceNormal(t, p1)),
//            quantizeSurfaceNormals(extractSurfaceNormal(t, p2)),
//            quantizeSurfaceNormals(extractSurfaceNormal(t, p3))
//        );
//        std::cout << "HashKey: " << key << std::endl;

//        cv::Point c = triplets[i].getP1Coords(offsetX, stepX, offsetY, stepY);
//        cv::Point p2 = triplets[i].getP2Coords(offsetX, stepX, offsetY, stepY);
//        cv::Point p3 = triplets[i].getP3Coords(offsetX, stepX, offsetY, stepY);
//        std::cout << "Triplet coords (" << c.x << ", " << c.y << ") "
//                                  << " (" << p2.x << ", " << p2.y << ") "
//                                  << " (" << p3.x << ", " << p3.y << ") "<< std::endl;
//
//
//        cv::circle(t, c, 1, cv::Scalar(1.0), -1);
//        cv::circle(t, p2, 1, cv::Scalar(0.65), -1);
//        cv::circle(t, p3, 1, cv::Scalar(0.4), -1);
//
//        cv::line(t, c, p2, cv::Scalar(1.0));
//        cv::line(t, c, p3, cv::Scalar(1.0));
//
//        std::cout << "Triplet " << i << ": " << triplets[i] << std::endl;

//    cv::Mat t_copy = cv::Mat(t.size(), CV_32FC3);
//    for (int y = 1; y < t.rows - 1; y++) {
//        for (int x = 1; x < t.cols - 1; x++) {
//            t_copy.at<cv::Vec3f>(y, x) = extractSurfaceNormal(t, cv::Point(x, y));
//        }
//    }
//
//    cv::imshow("Train hashing, test template", t_copy);
//    cv::waitKey(0);
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
