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

        Template parseTemplateGt(uint index, const std::string &path, cv::FileNode &gtNode);
        void parseTemplateInfo(Template &t, cv::FileNode &infoNode);

    public:
        explicit Parser(cv::Ptr<ClassifierCriteria> criteria) : criteria(criteria) {}

        /**
         * @brief Parses templates for one object in given path.
         *
         * Function expects rgb/, depth/ folders and gt.yml and info.yml
         * files in root folder defined by path param.
         *
         * @param[in]  path       Path to object folder
         * @param[in]  modelsPath Path to models folder (contains *.ply and info.yml files)
         * @param[out] templates  Output vector containing all parsed templates
         * @param[in]  indices    Optional parameter to parse only templates with defined indicies
         */
        void parseTemplate(const std::string &path, const std::string &modelsPath, std::vector<Template> &templates, std::vector<uint> indices = {});
    };
}

#endif
