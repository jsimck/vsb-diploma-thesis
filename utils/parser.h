#ifndef VSB_SEMESTRAL_PROJECT_PARSER_H
#define VSB_SEMESTRAL_PROJECT_PARSER_H

#include <string>
#include <utility>
#include <memory>
#include "../core/template.h"
#include "../core/classifier_criteria.h"
#include "../core/scene.h"

namespace tless {
    /**
     * @brief  Utility class to parse templates downloaded from http://cmp.felk.cvut.cz/t-less/
     */
    class Parser {
    private:
        std::vector<float> diameters;
        cv::Ptr<ClassifierCriteria> criteria;

        /**
         * @brief Parses template images, gt.yml and creates base Template object
         *
         * @param[in] index    Current template index
         * @param[in] basePath Base path to templates folder
         * @param[in] gtNode   Template specific gt.yml file node
         * @return             Parsed template object
         */
        Template parseTemplateGt(uint index, const std::string &basePath, cv::FileNode &gtNode);

        /**
         * @brief Parses camera params, computes normals, gradients and extracts criteria from each template
         *
         * @param[in] t        Existing template object to parse features and camera params for
         * @param[in] infoNode Template specific info.yml file node
         */
        void parseTemplateInfo(Template &t, cv::FileNode &infoNode);

    public:
        explicit Parser(cv::Ptr<ClassifierCriteria> criteria) : criteria(criteria) {}

        /**
         * @brief Parses templates for one object in given path.
         *
         * Function expects rgb/, depth/ folders and gt.yml and info.yml
         * files in root folder defined by path param.
         *
         * @param[in]  basePath   Path to object folder
         * @param[in]  modelsPath Path to models folder (contains *.ply and info.yml files)
         * @param[out] templates  Output vector containing all parsed templates
         * @param[in]  indices    Optional parameter to parse only templates with defined indicies
         */
        void parseTemplate(const std::string &basePath, const std::string &modelsPath, std::vector<Template> &templates, std::vector<uint> indices = {});

        /**
         * @brief Parses scene info, images, computes quantized normals and gradients
         *
         * @param[in] basePath Base path to scene folder with info.yml and rgb, depth folders
         * @param[in] index    Current index of a scene image
         * @param[in] scale    Current scale of image scale pyramid
         * @return             Parsed scene object
         */
        Scene parseScene(const std::string &basePath, int index, float scale);
    };
}

#endif
