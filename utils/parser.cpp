#include "parser.h"
#include "../processing/processing.h"
#include "../objdetect/matcher.h"
#include "../objdetect/hasher.h"

namespace tless {
    void Parser::parseTemplate(const std::string &path, const std::string &modelsPath, std::vector<Template> &templates, std::vector<uint> indices) {
        cv::FileStorage fs;

        // Parse object diameters if empty
        if (diameters.empty()) {
            fs.open(modelsPath + "info.yml", cv::FileStorage::READ);
            assert(fs.isOpened());

            // Parse only diameters for now
            for (uint i = 1;; i++) {
                float diameter;
                std::string index = "model_" + std::to_string(i);
                cv::FileNode modelNode = fs[index];

                // Break if obj is empty (last model)
                if (modelNode.empty()) break;

                // Parse diameter
                modelNode["diameter"] >> diameter;
                diameters.push_back(diameter);
            }

            fs.release();
        }


        // Load obj_gt
        fs.open(path + "gt.yml", cv::FileStorage::READ);
        assert(fs.isOpened());

        // Parse obj_gt
        for (uint i = 0;; i++) {
            auto tplIndex = static_cast<uint>((!indices.empty()) ? indices[i] : i);
            std::string index = "tpl_" + std::to_string(tplIndex);
            cv::FileNode objGt = fs[index];

            // Break if obj is empty (last template)
            if (objGt.empty()) break;

            // Parse template gt file
            templates.push_back(parseTemplateGt(tplIndex, path, objGt));
        }

        fs.release();


        // Load obj_info
        fs.open(path + "info.yml", cv::FileStorage::READ);
        assert(fs.isOpened());

        // Parse obj_info
        for (uint i = 0;; i++) {
            auto tplIndex = static_cast<uint>((!indices.empty()) ? indices[i] : i);
            std::string index = "tpl_" + std::to_string(tplIndex);
            cv::FileNode objInfo = fs[index];

            // Break if obj is empty (last template)
            if (objInfo.empty()) break;

            // Parse template info file
            parseTemplateInfo(templates[i], objInfo);
        }

        fs.release();
    }

    Template Parser::parseTemplateGt(uint index, const std::string &path, cv::FileNode &gtNode) {
        int id;
        std::vector<float> vCamRm2c, vCamTm2c;
        std::vector<int> vObjBB;

        gtNode["obj_id"] >> id;
        gtNode["obj_bb"] >> vObjBB;
        gtNode["cam_R_m2c"] >> vCamRm2c;
        gtNode["cam_t_m2c"] >> vCamTm2c;

        assert(!vObjBB.empty());
        assert(!vCamRm2c.empty());
        assert(!vCamTm2c.empty());

        // Create filename from index
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << index;
        std::string fileName = ss.str();

        // Load source images
        cv::Mat src = cv::imread(path + "/rgb/" + fileName + ".png", CV_LOAD_IMAGE_COLOR);
        cv::Mat srcDepth = cv::imread(path + "/depth/" + fileName + ".png", CV_LOAD_IMAGE_UNCHANGED);

        assert(src.type() == CV_8UC3);
        assert(srcDepth.type() == CV_16U);

        cv::Mat srcHSV;
        cv::cvtColor(src, srcHSV, CV_BGR2HSV);
        cv::cvtColor(src, src, CV_BGR2GRAY);
        src.convertTo(src, CV_32F, 1.0f / 255.0f);

        // Generate quantized orientations
        cv::Mat gradients, magnitudes;
        quantizedOrientationGradients(src, gradients, magnitudes);

        // Create template
        Template t;
        t.id = index + (2000 * id);
        t.fileName = std::move(fileName);
        t.diameter = diameters[id];
        t.srcGray = std::move(src);
        t.srcHSV = std::move(srcHSV);
        t.srcDepth = std::move(srcDepth);
        t.srcGradients = std::move(gradients);
        t.objBB = cv::Rect(vObjBB[0], vObjBB[1], vObjBB[2], vObjBB[3]);
        t.camera = Camera(
            cv::Mat(3, 3, CV_32FC1, vCamRm2c.data()),
            cv::Mat(3, 1, CV_32FC1, vCamTm2c.data())
        );

        // Extract criteria
        if (t.objBB.area() < criteria->info.smallestTemplate.area()) {
            criteria->info.smallestTemplate = t.objBB.size();
        }

        if (t.objBB.width > criteria->info.largestArea.width) {
            criteria->info.largestArea.width = t.objBB.width;
        }

        if (t.objBB.height > criteria->info.largestArea.height) {
            criteria->info.largestArea.height = t.objBB.height;
        }

        return t;
    }

    void Parser::parseTemplateInfo(Template &t, cv::FileNode &infoNode) {
        std::vector<float> vCamK;

        infoNode["cam_K"] >> vCamK;
        infoNode["elev"] >> t.camera.elev;
        infoNode["mode"] >> t.camera.mode;

        t.camera.azimuth = 5 * ((t.id % 2000) % 72); // Training templates are sampled in 5 step azimuth
        t.camera.K = cv::Mat(3, 3, CV_32FC1, vCamK.data()).clone();

        // Find max/min depth and max local depth for depth quantization
        ushort localMax = t.srcDepth.at<ushort>(t.objBB.tl());
        ushort localMin = static_cast<ushort>(-1);
        const int areaOffset = 5;

        // Offset bounding box so we cover edges of object
        for (int y = t.objBB.tl().y - areaOffset; y < t.objBB.br().y + areaOffset; y++) {
            for (int x = t.objBB.tl().x - areaOffset; x < t.objBB.br().x + areaOffset; x++) {
                ushort val = t.srcDepth.at<ushort>(y, x);

                // Extract local max (val shouldn't also be bigger than 3 times local max, there shouldn't be so much swinging)
                if (val > localMax && val < 3 * localMax) {
                    localMax = val;
                }

                // Extract local min
                if (val < localMin && val > 0) {
                    localMin = val;
                }

                // Extract criteria
                if (val > criteria->info.maxDepth && val < 3 * localMax) {
                    criteria->info.maxDepth = val;
                }

                if (val < criteria->info.minDepth && val > 0) {
                    criteria->info.minDepth = val;
                }
            }
        }

        // Normalize local max and min depths to define error corrected range
        localMax /= depthNormalizationFactor(localMax, criteria->depthDeviationFun);
        localMin *= depthNormalizationFactor(localMax, criteria->depthDeviationFun);

        // Extract min edgels
        cv::Mat integral;
        depthEdgelsIntegral(t.srcDepth, integral, localMin, localMax);

        // Cover little bit larger area than object bounding box, to count edges
        cv::Point A(t.objBB.tl().x - areaOffset, t.objBB.tl().y - areaOffset);
        cv::Point B(t.objBB.br().x + areaOffset, t.objBB.tl().y - areaOffset);
        cv::Point C(t.objBB.tl().x - areaOffset, t.objBB.br().y + areaOffset);
        cv::Point D(t.objBB.br().x + areaOffset, t.objBB.br().y + areaOffset);

        // Get edgel count inside obj bounding box
        int edgels = integral.at<int>(D) - integral.at<int>(B) - integral.at<int>(C) + integral.at<int>(A);

        if (edgels < criteria->info.minEdgels) {
            criteria->info.minEdgels = edgels;
        }

        // Compute normals
        quantizedNormals(t.srcDepth, t.srcNormals, t.camera.fx(), t.camera.fy(), localMax, criteria->maxDepthDiff);
    }
}