#include "classifier.h"
#include "../utils/timer.h"
#include "../utils/visualizer.h"
#include "../core/classifier_criteria.h"
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
            oss << resultPath << "trained_" << std::setw(2) << std::setfill('0') << templates[0].id / 2000 << ".yml.gz";
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
        // Init classifiers
        Parser parser(criteria);
        Objectness objectness(criteria);
        Hasher hasher(criteria);
        Matcher matcher(criteria);
        Visualizer viz(criteria);

        // Load trained template data
        load(trainedTemplatesListPath, trainedPath);

        cv::Mat result;
        const float initialScale = .4f;
        const float scaleFactor = 1.25f;
        const int finalScaleLevel = 9;
        float scale = initialScale;

        for (int i = 0; i < 503; ++i) {
            for (int pyramidLevel = 0; pyramidLevel < finalScaleLevel; ++pyramidLevel) {
                Timer tTotal;

                // Load scene
                Timer tSceneLoading;
                Scene scene = parser.parseScene(scenePath, i, scale);
                std::cout << "  |_ Scene loaded in: " << tSceneLoading.elapsed() << "s" << std::endl;

                // Save scene at scale 1.0f for visualization
                if (scale <= 1.1f && scale >= 0.9f) {
                    result = scene.srcRGB.clone();
                }

                /// Objectness detection
                assert(criteria->info.smallestTemplate.area() > 0);
                assert(criteria->info.minEdgels > 0);

                Timer tObjectness;
                objectness.objectness(scene.srcDepth, windows);
                std::cout << "  |_ Objectness detection took: " << tObjectness.elapsed() << "s" << std::endl;
//                viz.objectness(scene, windows);

                /// Verification and filtering of template candidates
                if (windows.empty()) {
                    continue;
                }

                Timer tVerification;
                hasher.verifyCandidates(scene.srcDepth, scene.srcNormals, tables, windows); // TODO refactor to use Scene as input
                std::cout << "  |_ Hashing verification took: " << tVerification.elapsed() << "s" << std::endl;
                viz.windowsCandidates(scene, windows);

                /// Match templates
                Timer tMatching;
                matcher.match(scale, scene, windows, matches);
                std::cout << "  |_ Template matching took: " << tMatching.elapsed() << "s" << std::endl;

//                /// Show matched template results
//                std::cout << std::endl << "Matches size: " << matches.size() << std::endl;
////                Visualizer::visualizeMatches(scene.srcRGB, scale, matches, "data/400x400/", 0);
//
//                std::cout << "Classification took: " << tTotal.elapsed() << "s" << std::endl;
                scale *= scaleFactor;
                windows.clear();
            }

            // Apply non-maxima suppression
//            nonMaximaSuppression(matches, criteria->overlapFactor);

            // Visualize results
//            Visualizer::visualizeMatches(result, 1.0f, matches, "data/400x400/", 1, "Pyramid results");
            scale = initialScale;
//            matches.clear();
        }
    }
}