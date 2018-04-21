#include "classifier.h"
#include <boost/filesystem.hpp>
#include "../utils/timer.h"
#include "../utils/visualizer.h"
#include "../processing/processing.h"
#include "fine_pose.h"
#include "../core/result.h"

namespace tless {
    void Classifier::train(const std::string &tplsFolder, const std::vector<int> &indices) {
        Timer tTraining;
        std::vector<Template> objTpls;
        std::cout << "Training... " << std::endl;
        std::cout << "  |_ templates -> ";

        this->templates.clear();
        this->tables.clear();

        for (auto &id : indices) {
            std::string path = cv::format((tplsFolder + "%02d/").c_str(), id);

            // Parse each object by one and extract features for it
            parser.parseObject(path, objTpls);
            matcher.train(objTpls);

            // Save templates for later hash table generation
            this->templates.insert(this->templates.end(), objTpls.begin(), objTpls.end());
            std::cout << id << ", ";

            objTpls.clear();
        }

        // Train hash tables
        std::cout << std::endl << "  |_ hash tables -> ";
        hasher.train(this->templates, this->tables);
        std::cout << tables.size() << " hash tables generated" << std::endl;
        std::cout << "DONE!, training took: " << tTraining.elapsed() << " s" << std::endl << std::endl;

        // Save obj ids
        this->objIds.insert(this->objIds.end(), indices.begin(), indices.end());

        // Checks
        assert(!this->tables.empty());
        assert(this->tables.size() == criteria->tablesCount);
        assert(!this->templates.empty());
    }

    void Classifier::save(const std::string &trainedFolder, const std::string &classifierFileName,
                          const std::string &tplsFileFormat) {
        // Create directories if they don't exist
        boost::filesystem::create_directories(trainedFolder);

        cv::FileStorage fs;
        const std::string classifierPath = trainedFolder + classifierFileName;
        int objId = -1;

        std::cout << "Saving results... " << std::endl;
        assert(!this->templates.empty());

        for (auto &tpl: this->templates) {
            // Create new file for each new object
            if (objId < 0 || objId != tpl.objId) {
                if (fs.isOpened()) {
                    fs << "]";
                    fs.release();
                }

                // Save obj id as current one
                objId = tpl.objId;

                // Open new file
                std::string tplPath = cv::format((trainedFolder + tplsFileFormat).c_str(), objId);
                fs.open(tplPath, cv::FileStorage::WRITE);
                fs << "templates" << "[";

                std::cout << "  |_ " << objId << " -> " << tplPath << std::endl;
            }

            fs << tpl;
        }

        fs.release();

        // Save classifier info and hashTables
        fs.open(classifierPath, cv::FileStorage::WRITE);

        // Save classifier info
        fs << "criteria" << *this->criteria;
        std::cout << "  |_ classifier info -> " << classifierPath << std::endl;

        // Save info about parsed objects
        fs << "objIds" << this->objIds;

        // Persist hashTables
        assert(!this->tables.empty());

        fs << "tables" << "[";
        for (auto &table : this->tables) {
            fs << table;
        }
        fs << "]";

        std::cout << "  |_ hash tables -> " << classifierPath << std::endl << std::endl;
        fs.release();
    }

    void Classifier::load(const std::string &trainedFolder, const std::string &classifierFileName,
                          const std::string &tplsFileFormat) {
        Timer tLoading;
        std::cout << "Loading trained templates... " << std::endl;

        templates.clear();
        tables.clear();
        std::string criteriaPath = trainedFolder + classifierFileName;

        // Load criteria
        cv::FileStorage fsc(criteriaPath, cv::FileStorage::READ);
        fsc["criteria"] >> criteria;
        fsc["objIds"] >> this->objIds;
        std::cout << "  |_ loaded criteria -> " << criteriaPath << std::endl;
        std::cout << "  |_ templates -> ";

        // Load templates
        for (auto &objId : this->objIds) {
            cv::FileStorage fs(cv::format((trainedFolder + tplsFileFormat).c_str(), objId), cv::FileStorage::READ);
            cv::FileNode tplsNode = fs["templates"];

            // Loop through templates
            for (auto &&t : tplsNode) {
                Template nTpl;
                t >> nTpl;
                templates.push_back(nTpl);
            }

            fs.release();
            std::cout << objId << ", ";
        }

        // Load hash tables
        cv::FileNode htNode = fsc["tables"];
        for (auto &&table : htNode) {
            this->tables.emplace_back(HashTable::load(table, templates));
        }

        fsc.release();
        std::cout << std::endl << "  |_ loaded hash tables (" << tables.size() << ")" << std::endl;
        std::cout << "DONE!, took: " << tLoading.elapsed() << " s" << std::endl << std::endl;
        std::cout << "Criteria..." << std::endl;
        std::cout << *criteria << std::endl << std::endl;
    }

    void Classifier::detect(const std::string &scenesFolder, std::vector<int> sceneIndices, const std::string &resultsFolder,
                            int startScene, int endScene, const std::string &resultsFileFormat) {
        assert(criteria->info.smallestTemplate.area() > 0);
        assert(criteria->info.minEdgels > 0);

        std::vector<std::vector<double>> timers;
        std::vector<std::vector<Match>> results;
        std::vector<Window> windows;
        std::vector<Match> matches;

        // Init fine pose and vizualizer
        Visualizer viz(criteria);
        FinePose finePose(criteria, shadersFolder, modelsFolder, modelsFileFormat, objIds);

        // Define contsants
        const int pyrLevels = criteria->pyrLvlsDown + criteria->pyrLvlsUp;
        const auto minEdgels = static_cast<const int>(criteria->info.minEdgels * criteria->objectnessFactor);
        const auto minDepthMag = static_cast<const int>(criteria->objectnessDiameterThreshold * criteria->info.smallestDiameter * criteria->info.depthScaleFactor);

        // Timing
        Timer tTotal;
        double ttSceneLoading, ttObjectness, ttVerification, ttMatching, ttNMS;
        std::cout << "Matching started..." << std::endl << std::endl;

        for (auto &sceneId : sceneIndices) {
            for (int i = startScene; i < endScene; ++i) {
                // Reset timers
                ttObjectness = ttVerification = ttMatching = 0;
                tTotal.reset();

                // Load scene
                Timer tSceneLoading;
                std::string scenePath = cv::format((scenesFolder + "%02d/").c_str(), sceneId);
                Scene scene = parser.parseScene(scenePath, i, criteria->pyrScaleFactor, criteria->pyrLvlsDown, criteria->pyrLvlsUp);
                ttSceneLoading = tSceneLoading.elapsed();

                // Verification for current level of image pyramid
                for (int l = 0; l <= pyrLevels; ++l) {
                    /// Objectness detection
                    Timer tObjectness;
                    objectness(scene.pyramid[l].srcDepth, windows, criteria->info.smallestTemplate, criteria->windowStep,
                               scene.pyramid[l].scale, criteria->info.minDepth, criteria->info.maxDepth, minDepthMag, minEdgels);
                    ttObjectness += tObjectness.elapsed();
                    viz.objectness(scene.pyramid[l], windows);

                    if (windows.empty()) {
                        continue;
                    }

                    /// Verification and filtering of template candidates
                    Timer tVerification;
                    hasher.verifyCandidates(scene.pyramid[l].srcDepth, scene.pyramid[l].srcNormals, tables, windows);
                    ttVerification += tVerification.elapsed();
                    viz.windowsCandidates(scene.pyramid[l], windows);

                    /// Match templates
                    Timer tMatching;
                    matcher.match(scene.pyramid[l], windows, matches);
                    ttMatching += tMatching.elapsed();
                    windows.clear();
                }

                // Apply non-maxima suppression
                viz.preNonMaxima(scene.pyramid[criteria->pyrLvlsDown], matches, 0);
                Timer tNMS;
                nms(matches, criteria->overlapFactor);
                ttNMS = tNMS.elapsed();

                // Print results
                std::cout << std::endl << "Classification took: " << tTotal.elapsed() << "s" << std::endl;
                std::cout << "  |_ Scene " << (i + 1) << "/" <<  (endScene) << " took: " << ttSceneLoading << "s" << std::endl;
                std::cout << "  |_ Objectness detection took: " << ttObjectness << "s" << std::endl;
                std::cout << "  |_ Hashing verification took: " << ttVerification << "s" << std::endl;
                std::cout << "  |_ Template matching took: " << ttMatching << "s" << std::endl;
                std::cout << "  |_ NMS took: " << ttNMS << "s" << std::endl;
                std::cout << "  |_ Matches: " << matches.size() << std::endl;

                // Save times each section took
                timers.push_back({ttSceneLoading, ttObjectness, ttVerification, ttMatching, ttNMS});

                // Vizualize results and clear current matches
                viz.matches(scene.pyramid[criteria->pyrLvlsDown], matches, 1);

                // Apply fine pose estimation
//                finePose.estimate(matches, scene);

                results.emplace_back(std::move(matches));
            }

            // Save results
            saveResults(sceneId, results, resultsFolder, resultsFileFormat, timers, startScene);
            results.clear();
        }
    }

    void Classifier::saveResults(int sceneId, const std::vector<std::vector<Match>> &results, const std::string &resultsFolder,
                                     const std::string &resultsFileFormat, const std::vector<std::vector<double>> &timers, int startIndex) {
        // Crete directories
        boost::filesystem::create_directories(resultsFolder);

        // Open file for saving
        cv::FileStorage fs(cv::format((resultsFolder + resultsFileFormat).c_str(), sceneId), cv::FileStorage::WRITE);
        fs << "scenes" << "[";

        for (int i = 0; i < results.size(); ++i) {
            fs << "{";
            fs << "index" << i + startIndex;
            fs << "timers" << "{";
            fs << "scene" << timers[i][0];
            fs << "objectness" << timers[i][1];
            fs << "hashing" << timers[i][2];
            fs << "matching" << timers[i][3];
            fs << "nms" << timers[i][4];
            fs << "}";
            fs << "matches" << "[";
            for (auto &match : results[i]) {
                fs << Result(match);
            }
            fs << "]";
            fs << "}";
        }

        // Close file
        fs << "]";
        fs.release();
    }

    const std::string &Classifier::getShadersFolder() const {
        return shadersFolder;
    }

    void Classifier::setShadersFolder(const std::string &shadersFolder) {
        Classifier::shadersFolder = shadersFolder;
    }

    const std::string &Classifier::getModelsFolder() const {
        return modelsFolder;
    }

    void Classifier::setModelsFolder(const std::string &modelsFolder) {
        Classifier::modelsFolder = modelsFolder;
    }

    const std::string &Classifier::getModelFileFormat() const {
        return modelsFileFormat;
    }

    void Classifier::setModelFileFormat(const std::string &modelFileFormat) {
        Classifier::modelsFileFormat = modelFileFormat;
    }
}