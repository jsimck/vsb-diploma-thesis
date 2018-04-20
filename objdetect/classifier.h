#ifndef VSB_SEMESTRAL_PROJECT_CLASSIFIER_H
#define VSB_SEMESTRAL_PROJECT_CLASSIFIER_H

#include <memory>
#include "../core/match.h"
#include "../core/hash_table.h"
#include "../utils/parser.h"
#include "hasher.h"
#include "../core/window.h"
#include "matcher.h"
#include "../core/classifier_criteria.h"

namespace tless {
    /**
     * @brief Main class of the whole project which handles all training and classification.
     */
    class Classifier {
    private:
        std::string shadersFolder = "data/shaders/";
        std::string modelsFolder = "data/models/";
        std::string modelsFileFormat = "obj_%02d.ply";

        cv::Ptr<ClassifierCriteria> criteria;
        std::vector<int> objIds;
        std::vector<Template> templates;
        std::vector<HashTable> tables;

        Parser parser;
        Hasher hasher;
        Matcher matcher;

        /**
         * @brief Saves matched results to yml file for further evaluation.
         *
         * @param[in] sceneId           Current scene ID
         * @param[in] results           Array of found matches
         * @param[in] resultsFolder     Path to results folder (created if doesn't exist)
         * @param[in] resultsFileFormat File format for the results file
         */
        void saveResults(int sceneId, const std::vector<std::vector<Match>> &results, const std::string &resultsFolder,
                                 const std::string &resultsFileFormat, int startIndex);

    public:
        explicit Classifier(cv::Ptr<ClassifierCriteria> criteria) :
                criteria(criteria), parser(criteria), hasher(criteria), matcher(criteria) {}

        /**
         * @brief Runs object detection algorithm on given set of scene (identified by indicies) with templates that were trained beforehand.
         *
         * @param[in] scenesFolder      Base path to scenes folder (this folder should contain scenes 01, 02, ...)
         * @param[in] sceneIndices      Scene indicies identifying scenes we want to run detection on
         * @param[in] resultsFolder     Folder containing all results files
         * @param[in] resultsFileFormat File format of the results file
         */
        void detect(const std::string &scenesFolder, std::vector<int> sceneIndices, const std::string &resultsFolder, int startScene,
                       int endScene, const std::string &resultsFileFormat = "results_%02d.yml.gz");

        /**
         * @brief Trains hastables and extract template features for objects defined in indicies parameter.
         *
         * @param[in] tplsFolder Path to templates folder containing object folders (01, 02, ...)
         * @param[in] indices    Indicies of object to extract feature and train hash tables for
         */
        void train(const std::string &tplsFolder, const std::vector<int> &indices);

        /**
         * @brief Save trained classifier and templates.
         *
         * @param[in] trainedFolder      Trained data output folder (created if doesn't exits)
         * @param[in] classifierFileName Trained classifier file name
         * @param[in] tplsFileFormat     File name format for trained templates
         */
        void save(const std::string &trainedFolder, const std::string &classifierFileName = "classifier.yml.gz",
                  const std::string &tplsFileFormat = "template_%02d.yml.gz");

        /**
         * @brief Loads trained templates and hash tables into classifier.
         *
         * @param[in] trainedFolder      Trained data output folder containing trained templates and classifier
         * @param[in] classifierFileName Trained classifier file name
         * @param[in] tplsFileFormat     Trained templates file format
         */
        void load(const std::string &trainedFolder, const std::string &classifierFileName = "classifier.yml.gz",
                  const std::string &tplsFileFormat = "template_%02d.yml.gz");

        void setShadersFolder(const std::string &shadersFolder);
        void setModelsFolder(const std::string &modelsFolder);
        void setModelFileFormat(const std::string &modelFileFormat);

        const std::string &getShadersFolder() const;
        const std::string &getModelsFolder() const;
        const std::string &getModelFileFormat() const;
    };
}

#endif
