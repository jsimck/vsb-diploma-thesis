#include "classifier.h"
#include "../utils/timer.h"
#include "../utils/visualizer.h"
#include "../core/classifier_criteria.h"
#include "../processing/processing.h"

namespace tless {
    void Classifier::train(std::string templatesListPath, std::string resultPath, std::string modelsPath, std::vector<uint> indices) {
        std::ifstream ifs(templatesListPath);
        assert(ifs.is_open());

        // Init classifiers
        Objectness objectness(criteria);
        Hasher hasher(criteria);
        Matcher matcher(criteria);

        // Init parser and common
        Parser parser(criteria);

        std::ostringstream oss;
        std::vector<Template> tpls, allTemplates;
        std::string path;

        Timer t;
        std::cout << "Training... " << std::endl;

        while (ifs >> path) {
            std::cout << "  |_ " << path;

            // Parse each object by one and save it
            parser.parseTemplate(path, modelsPath, tpls, indices);

            // Train features for loaded templates
            matcher.train(tpls);

            // Save templates for later hash table generation
            allTemplates.insert(allTemplates.end(), tpls.begin(), tpls.end());

            // Persist trained data
            oss.str("");
            oss << resultPath << "trained_" << std::setw(2) << std::setfill('0') << tpls[0].id / 2000 << ".yml.gz";
            std::string trainedPath = oss.str();
            cv::FileStorage fsw(trainedPath, cv::FileStorage::WRITE);

            fsw << "templates" << "[";
            for (auto &tpl : tpls) {
                fsw << tpl;
            }
            fsw << "]";

            fsw.release();
            tpls.clear();
            std::cout << " -> " << trainedPath << std::endl;
        }

        // Save data set
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

        std::cout << "DONE!, took: " << t.elapsed() << " s" << std::endl << std::endl;
    }

    void Classifier::load(const std::string &trainedTemplatesListPath, const std::string &trainedPath) {
        std::ifstream ifs(trainedTemplatesListPath);
        assert(ifs.is_open());

        Timer t;
        std::string path;
        std::cout << "Loading trained templates... " << std::endl;

        while (ifs >> path) {
            std::cout << "  |_ " << path;

            // Load trained data
            cv::FileStorage fsr(path, cv::FileStorage::READ);
            cv::FileNode tpls = fsr["templates"];

            // Loop through templates
            for (auto &&tpl : tpls) {
                Template nTpl;
                tpl >> nTpl;
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
        std::cout << "DONE!, took: " << t.elapsed() << " s" << std::endl << std::endl;
    }

    void Classifier::loadScene(const std::string &scenePath, const std::string &sceneName) {
        // Checks
        assert(scenePath.length() > 0);
        assert(sceneName.length() > 0);

        const float scale = 1.2f;

        // Load scenes
        cv::Mat srcScene = cv::imread(scenePath + "rgb/" + sceneName + ".png", CV_LOAD_IMAGE_COLOR);
        cv::Mat srcSceneDepth = cv::imread(scenePath + "depth/" + sceneName + ".png", CV_LOAD_IMAGE_UNCHANGED);

        // Resize - TODO remove and add scale pyramid functionality
        cv::resize(srcScene, scene, cv::Size(), scale, scale);
        cv::resize(srcSceneDepth, sceneDepth, cv::Size(), scale, scale);

        // Convert and normalize
        cv::cvtColor(scene, sceneHSV, CV_BGR2HSV);
        cv::cvtColor(scene, sceneGray, CV_BGR2GRAY);

        // Check if conversion went ok
        assert(!sceneHSV.empty());
        assert(!sceneGray.empty());
        assert(scene.type() == CV_8UC3);
        assert(sceneDepth.type() == CV_16U);
        assert(sceneHSV.type() == CV_8UC3);
        assert(sceneGray.type() == CV_8UC1);

        // TODO speed could be improved by only computing part of scene which is in objectness bounding box
        // Compute quantized surface normals and orientation gradients
        quantizedOrientationGradients(sceneGray, sceneQuantizedAngles, sceneMagnitudes);

        // TODO better handling of max depth optimization
        float ratio = depthNormalizationFactor(criteria->info.maxDepth * scale, criteria->depthDeviationFun);

        // TODO parseTemplate camera params and use proper fx and fy and distances etc
        quantizedNormals(sceneDepth, sceneNormals, 1076.74064739f, 1075.17825536f,
                         static_cast<int>((criteria->info.maxDepth * scale) / ratio), 100);
    }

    void Classifier::detect(std::string trainedTemplatesListPath, std::string trainedPath, std::string scenePath) {
        // Init classifiers
        Objectness objectness(criteria);
        Hasher hasher(criteria);
        Matcher matcher(criteria);

        // Load trained template data
        load(trainedTemplatesListPath, trainedPath);
        std::ostringstream oss;

        for (int i = 1; i < 503; ++i) {
            Timer tTotal;

            // Load scene
            Timer tSceneLoading;
            oss.str("");
            oss << std::setfill('0') << std::setw(4) << i;
            loadScene(scenePath, oss.str());
            std::cout << "  |_ Scene loaded in: " << tSceneLoading.elapsed() << "s" << std::endl;

            /// Objectness detection
            assert(criteria->info.smallestTemplate.area() > 0);
            assert(criteria->info.minEdgels > 0);

            Timer tObjectness;
            objectness.objectness(sceneDepth, windows);
            std::cout << "  |_ Objectness detection took: " << tObjectness.elapsed() << "s" << std::endl;

            Visualizer::visualizeWindows(this->scene, windows, false, 1, "Locations detected");

            /// Verification and filtering of template candidates
            assert(!tables.empty());

            Timer tVerification;
            hasher.verifyCandidates(sceneDepth, sceneNormals, tables, windows);
            std::cout << "  |_ Hashing verification took: " << tVerification.elapsed() << "s" << std::endl;

//            Visualizer::visualizeHashing(scene, sceneDepth, tables, windows, criteria, true);
            Visualizer::visualizeWindows(this->scene, windows, false, 1, "Filtered locations");

            /// Match templates
            assert(!windows.empty());
            matcher.match(1.2f, sceneHSV, sceneDepth, sceneMagnitudes, sceneQuantizedAngles, sceneNormals, windows, matches);

            /// Show matched template results
            std::cout << std::endl << "Matches size: " << matches.size() << std::endl;
            Visualizer::visualizeMatches(scene, matches, "data/", 1);

            // Cleanup
            std::cout << "Classification took: " << tTotal.elapsed() << "s" << std::endl;
            windows.clear();
            matches.clear();
        }
    }
}