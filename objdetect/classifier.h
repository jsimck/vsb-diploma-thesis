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
     * class Classifier
     *
     * Main class which runs all other classifiers and template parsers in the correct order.
     * In this class it's also possible to fine-tune the resulted parameters of each verification stage
     * which can in the end produce different results. These params can be adapted to processed templates
     * and scenes.
     */
    class Classifier {
    private:
        cv::Ptr<ClassifierCriteria> criteria;
        std::vector<Template> templates;
        std::vector<HashTable> tables;
        std::vector<Window> windows;
        std::vector<Match> matches;

        // Methods
        void load(const std::string &trainedTemplatesListPath, const std::string &trainedPath);

        void saveResults(const std::vector<std::vector<Match>> &results, const std::string &fileName);

    public:
        // Constructors
        explicit Classifier(cv::Ptr<ClassifierCriteria> criteria) : criteria(criteria) {}

        // Methods
        void train(const std::string &templatesListPath, const std::string &resultPath, std::vector<uint> indices = {});
        void detect(const std::string &trainedTemplatesListPath, const std::string &trainedPath, const std::string &shadersFolder,
                            const std::string &meshesListPath, const std::string &scenePath, const std::string &resultsFileName);
    };
}

#endif
