#ifndef VSB_SEMESTRAL_PROJECT_CONVERTOR_H
#define VSB_SEMESTRAL_PROJECT_CONVERTOR_H

#include <string>
#include <opencv2/core/hal/interface.h>
#include <vector>
#include <opencv2/core/types.hpp>
#include <MacTypes.h>
#include "../core/template.h"

namespace tless {
    /**
     * @brief Utility class which parses, converts and resizes templates downloaded from http://cmp.felk.cvut.cz/t-less/.
     *
     * Conversion is doe to desired output size + recalculates depth values based on the resize ratio.
     */
    class Converter {
    private:
        const uchar minGray = 20; //!< Minimal gray color value of image to consider containing object
        const int offset = 2; //!< Offset for object bounding boxes, when cropping we enlarge the objBB by this value in both directions to not cut any edges
        std::vector<float> diameters;

        /**
         * @brief Validates depth by looking in the pixel neighbourhood and invalidating all pixels that have larger difference than maxDiff.
         *
         * This function helps to eliminate invalid pixel that were likely caused by a sensor issue, e.g. very small or very large values.
         *
         * @param[in] depth   Depth value to validate
         * @param[in] src     Input 16-bit depth image
         * @param[in] p       Location of the suspicious pixel
         * @param[in] maxDiff Maximum allowed difference between pixel and it's neighbours (usually obj diameter)
         * @param[in] ksize   Kernel size (odd number, area around pixel to search in)
         * @return            True/false whether the pixel is valid or invalid
         */
        bool validateDepth(ushort depth, const cv::Mat &src, const cv::Point &p, int maxDiff, int ksize);

        /**
         * @brief Resizes rgb and depth images of given template + recalculates depth values based on resize ratio.
         *
         * Intristic camera parameters are also modified based on the resize ratio.
         *
         * @param[in,out] t          Template to resize image for (with rgb and depth images loaded)
         * @param[in]     outputPath Output path, where new resized template images should be saved
         * @param[in]     outputSize Size to which the template should be scaled to fit
         */
        void resizeAndSave(Template &t, const std::string &outputPath, int outputSize);

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