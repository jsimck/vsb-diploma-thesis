#ifndef VSB_SEMESTRAL_PROJECT_CLASSIFIER_H
#define VSB_SEMESTRAL_PROJECT_CLASSIFIER_H

#include <memory>
#include "../core/match.h"
#include "../core/hash_table.h"
#include "../utils/parser.h"
#include "hasher.h"
#include "objectness.h"
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

    public:
        // Constructors
        Classifier(cv::Ptr<ClassifierCriteria> criteria) : criteria(criteria) {}

        // Methods
        void train(std::string templatesListPath, std::string resultPath, std::vector<uint> indices = {});
        void detect(std::string trainedTemplatesListPath, std::string trainedPath, std::string scenePath);
    };
}

#endif
