#ifndef VSB_SEMESTRAL_PROJECT_PARSER_H
#define VSB_SEMESTRAL_PROJECT_PARSER_H

#include <string>
#include <utility>
#include <memory>
#include "../core/template.h"
#include "../core/classifier_criteria.h"

namespace tless {
    /**
     * @brief  Utility class to parse templates downloaded from http://cmp.felk.cvut.cz/t-less/
     */
    class Parser {
    private:
        std::vector<float> diameters;
        cv::Ptr<ClassifierCriteria> criteria;

        Template parseGt(uint index, const std::string &path, cv::FileNode &gtNode);
        void parseInfo(Template &t, cv::FileNode &infoNode);
        void parseModelsInfo(const std::string &modelsPath);
        void extractCriteria(Template &t);

    public:
        std::vector<uint> indices;
        uint tplCount, modelCount;

        Parser(cv::Ptr<ClassifierCriteria> criteria, uint tplCount = 1296, uint modelCount = 30)
                : criteria(criteria), tplCount(tplCount), modelCount(modelCount) {}

        void parse(std::string basePath, std::string modelsPath, std::vector<Template> &templates);
    };
}

#endif
