#include "classifier.h"
#include <boost/filesystem.hpp>
#include "../utils/timer.h"
#include "../utils/visualizer.h"
#include "../processing/processing.h"

namespace tless {
    void Classifier::train(std::string templatesListPath, std::string resultPath, std::vector<uint> indices) {
        std::ifstream ifs(templatesListPath);
        assert(ifs.is_open());

        // Init classifiers and parser
        Hasher hasher(criteria);
        Matcher matcher(criteria);
        Parser parser(criteria);

        // Init common
        std::ostringstream oss;
        std::vector<Template> templates, allTemplates;
        std::string path;

        // Create directories if they don't exist
        boost::filesystem::create_directories(resultPath);

        Timer tTraining;
        std::cout << "Training... " << std::endl;

        while (ifs >> path) {
            std::cout << "  |_ " << path;

            // Parse each object by one and save it
            parser.parseObject(path, templates, indices);

            // Train features for loaded templates
            matcher.train(templates);

            // Save templates for later hash table generation
            allTemplates.insert(allTemplates.end(), templates.begin(), templates.end());

            // Persist trained data
            oss << resultPath << std::setw(2) << std::setfill('0') << templates[0].objId << ".yml.gz";
            std::string trainedPath = oss.str();
            cv::FileStorage fsw(trainedPath, cv::FileStorage::WRITE);

            // Save templates data
            fsw << "templates" << "[";
            for (auto &t : templates) {
                fsw << t;
            }
            fsw << "]";

            // Cleanup
            oss.str("");
            fsw.release();
            templates.clear();
            std::cout << " -> " << trainedPath << std::endl;
        }

        ifs.close();

        // Save classifier info
        cv::FileStorage fsw(resultPath + "classifier.yml.gz", cv::FileStorage::WRITE);
        fsw << "criteria" << *criteria;
        std::cout << "  |_ info -> " << resultPath + "classifier.yml.gz" << std::endl;

        // Train hash tables
        std::cout << "  |_ Training hash tables... " << std::endl;
        hasher.train(allTemplates, tables);
        assert(!tables.empty());
        std::cout << "    |_ " << tables.size() << " hash tables generated" << std::endl;

        // Persist hashTables
        fsw << "tables" << "[";
        for (auto &table : tables) {
            fsw << table;
        }
        fsw << "]";
        fsw.release();

        std::cout << "  |_ tables -> " << resultPath + "classifier.yml.gz" << std::endl;
        std::cout << "DONE!, took: " << tTraining.elapsed() << " s" << std::endl << std::endl;
    }

    void Classifier::load(const std::string &trainedTemplatesListPath, const std::string &trainedPath) {
        std::ifstream ifs(trainedTemplatesListPath);
        assert(ifs.is_open());

        Timer tLoading;
        std::string path;
        std::cout << "Loading trained templates... " << std::endl;

        while (ifs >> path) {
            std::cout << "  |_ " << path;

            // Load trained data
            cv::FileStorage fsr(path, cv::FileStorage::READ);
            cv::FileNode tpls = fsr["templates"];

            // Loop through templates
            for (auto &&t : tpls) {
                Template nTpl;
                t >> nTpl;
                templates.push_back(nTpl);
            }

            fsr.release();
            std::cout << " -> LOADED" << std::endl;
        }

        // Load data set
        cv::FileStorage fsr(trainedPath + "classifier.yml.gz", cv::FileStorage::READ);
        fsr["criteria"] >> criteria;
        std::cout << "  |_ info -> LOADED" << std::endl;
        std::cout << "  |_ loading hashtables..." << std::endl;

        // Load hash tables
        cv::FileNode hashTables = fsr["tables"];
        for (auto &&table : hashTables) {
            tables.emplace_back(HashTable::load(table, templates));
        }

        fsr.release();
        std::cout << "  |_ hashTables -> LOADED (" << tables.size() << ")" << std::endl;
        std::cout << "DONE!, took: " << tLoading.elapsed() << " s" << std::endl << std::endl;
    }

    void Classifier::detect(std::string trainedTemplatesListPath, std::string trainedPath, std::string scenePath) {
        // Checks
        assert(criteria->info.smallestTemplate.area() > 0);
        assert(criteria->info.minEdgels > 0);

        // Init classifiers
        Parser parser(criteria);
        Objectness objectness(criteria);
        Hasher hasher(criteria);
        Matcher matcher(criteria);
        Visualizer viz(criteria);
        Scene scene;

        // Load trained template data
        load(trainedTemplatesListPath, trainedPath);

        // Image pyramid
        const int pyrLevels = criteria->pyrLvlsDown + criteria->pyrLvlsUp;

        // Timing
        Timer tTotal;
        double ttSceneLoading, ttObjectness, ttVerification, ttMatching, ttNMS;
        std::cout << "Matching started..." << std::endl << std::endl;

        for (int i = 0; i < 503; ++i) {
            // Reset timers
            ttObjectness = ttVerification = ttMatching = 0;
            tTotal.reset();

            // Load scene
            Timer tSceneLoading;
            scene = parser.parseScene(scenePath, i, criteria->pyrScaleFactor, criteria->pyrLvlsDown, criteria->pyrLvlsUp);
            ttSceneLoading = tSceneLoading.elapsed();

            // Verification for a pyramid
            for (int l = 0; l <= pyrLevels; ++l) {
                // Objectness detection
                Timer tObjectness;
                objectness.objectness(scene.pyramid[l].srcDepth, windows);
                ttObjectness += tObjectness.elapsed();
//                viz.objectness(scene.pyramid[l], windows);

                /// Verification and filtering of template candidates
                if (windows.empty()) {
                    continue;
                }

                Timer tVerification;
                hasher.verifyCandidates(scene.pyramid[l].srcDepth, scene.pyramid[l].srcNormals, tables, windows);
                ttVerification += tVerification.elapsed();
//                viz.windowsCandidates(scene.pyramid[l], windows);

                /// Match templates
                Timer tMatching;
                matcher.match(scene.pyramid[l], windows, matches);
                ttMatching += tMatching.elapsed();
                windows.clear();
            }

            // Apply non-maxima suppression
//            viz.preNonMaxima(scene.pyramid[criteria->pyrLvlsDown], matches);
            Timer tNMS;
            nms(matches, criteria->overlapFactor);
            ttNMS = tNMS.elapsed();

            // Print results
            std::cout << std::endl << "Classification took: " << tTotal.elapsed() << "s" << std::endl;
            std::cout << "  |_ Scene loading took: " << ttSceneLoading << "s" << std::endl;
            std::cout << "  |_ Objectness detection took: " << ttObjectness << "s" << std::endl;
            std::cout << "  |_ Hashing verification took: " << ttVerification << "s" << std::endl;
            std::cout << "  |_ Template matching took: " << ttMatching << "s" << std::endl;
            std::cout << "  |_ NMS took: " << ttNMS << "s" << std::endl;

            // Vizualize results and clear current matches
            viz.matches(scene.pyramid[criteria->pyrLvlsDown], matches, 1);
            matches.clear();
        }
    }
}