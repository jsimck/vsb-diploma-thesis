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

        /**
         * @brief Parses template images, gt.yml and creates base Template object.
         *
         * @param[in] index        Current template index
         * @param[in] basePath     Base path to templates folder
         * @param[in] gtNode       Template specific gt.yml file node
         * @param[in] infoNode     Template specific info.yml file node
         * @param[in,out] criteria Optional criteria parameter, if not null, template criteria are
         *                         extracted + all quantized normals and gradient images are generated
         * @return                 Parsed template object
         */
        Template parseTemplate(uint index, const std::string &basePath, cv::FileNode &gtNode, cv::FileNode &infoNode, cv::Ptr<ClassifierCriteria> criteria);

        /**
         * @brief Parses criteria for given template + generates quantized normals and gradients images.
         *
         * @param[in,out] t        Template to extract criteria for
         * @param[in,out] criteria Criteria objects which holds all currently extracted data
         */
        void parseCriteria(Template &t, cv::Ptr<ClassifierCriteria> criteria);

    public:
        Parser() = default;

        /**
         * @brief Parses templates for one object in given path.
         *
         * Function expects rgb/, depth/ folders and gt.yml and info.yml
         * files in root folder defined by path param.
         *
         * @param[in]     basePath   Path to object folder
         * @param[in]     modelsPath Path to models folder (contains *.ply and info.yml files)
         * @param[out]    templates  Output vector containing all parsed templates
         * @param[in,out] criteria   Optional criteria parameter, if provided, template criteria gets extracted
         *                           during parsing stage + all quantized normals and gradient images are generated
         * @param[in]     indices    Optional parameter to parse only templates with defined indices
         */
        void parseObject(const std::string &basePath, const std::string &modelsPath, std::vector<Template> &templates,
                         cv::Ptr<ClassifierCriteria> criteria = cv::Ptr<ClassifierCriteria>(), std::vector<uint> indices = {});

        /**
         * @brief Parses scene info, images, computes quantized normals and gradients.
         *
         * @param[in]     basePath Base path to scene folder with info.yml and rgb, depth folders
         * @param[in]     index    Current index of a scene image
         * @param[in]     scale    Current scale of image scale pyramid
         * @param[in,out] criteria Optional criteria parameter, if provided, scene quantized normals and gradient images are generated
         * @return                 Parsed scene object
         */
        Scene parseScene(const std::string &basePath, int index, float scale, cv::Ptr<ClassifierCriteria> criteria = cv::Ptr<ClassifierCriteria>());
    };
}

#endif
