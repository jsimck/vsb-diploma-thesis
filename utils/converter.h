#ifndef VSB_SEMESTRAL_PROJECT_CONVERTOR_H
#define VSB_SEMESTRAL_PROJECT_CONVERTOR_H

#include <string>
#include <opencv2/core/hal/interface.h>
#include <vector>
#include <opencv2/core/types.hpp>
#include "../core/template.h"

namespace tless {
    /**
     * @brief Utility class which parses, converts and resizes templates downloaded from http://cmp.felk.cvut.cz/t-less/
     * to desired output size + recalculates depth values based on the resize ratio.
     */
    class Converter {
    private:
        std::vector<float> diameters;

        /**
         * @brief Resizes rgb and depth images of given template + recalculates depth values based on resize ratio.
         *
         * @param[in,out] tpl    Template to resize image for (with rgb and depth images loaded)
         * @param[in] outputPath Output path, where new resized template images should be saved
         * @param[in] outputSize Size to which the template should be scaled to fit
         */
        void resizeAndSave(Template &tpl, const std::string &outputPath, int outputSize);

        /**
         * @brief Parses template images, gt.yml and creates base Template object.
         *
         * @param[in] index        Current template index
         * @param[in] basePath     Base path to templates folder
         * @param[in] gtNode       Template specific gt.yml file node
         * @param[in] infoNode     Template specific info.yml file node
         * @return                 Parsed template object
         */
        Template parseTemplate(uint index, const std::string &basePath, cv::FileNode &gtNode, cv::FileNode &infoNode);
    public:
        Converter() = default;

        /**
         * @brief Parses templates downloaded from http://cmp.felk.cvut.cz/t-less/ and resizes them.
         *
         * Templates are parsed and resized to given outputSize. Some additional meta details are also extracted/generated and then saved to
         * accompanying info.yml file for further reference.
         *
         * @param[in] objectsListPath List of object folders to parse
         * @param[in] modelsInfoPath  Path to the models/info.yml containing objects diameters and other info
         * @param[in] outputPath      Base output path to which the converted templates are saved
         * @param[in] outputSize      Desired size of the final template to which the templates should be scaled to fit
         */
        void convert(const std::string &objectsListPath, const std::string &modelsInfoPath, std::string outputPath, int outputSize);
    };
}

#endif