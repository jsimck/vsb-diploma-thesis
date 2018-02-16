#include "parser.h"
#include "../processing/processing.h"
#include "../objdetect/matcher.h"
#include "../objdetect/hasher.h"

namespace tless {
    void Parser::parseObject(const std::string &basePath, std::vector<Template> &templates, std::vector<uint> indices) {
        // Load object info.yml.gz at the root of each object folder
        cv::FileStorage fsInfo(basePath + "info.yml.gz", cv::FileStorage::READ);
        cv::FileNode tplNodes = fsInfo["templates"];
        const size_t nodesSize = indices.empty() ? tplNodes.size() : indices.size();

        // Loop through each template node and parse info + images
        for (uint i = 0; i < nodesSize; i++) {
            uint index = indices.empty() ? i : indices[i];

            // Load template info
            Template tpl;
            tplNodes[index] >> tpl;

            // Load template images and generate gradients, normals, hsv, gray images
            parseTemplate(tpl, basePath);
            templates.push_back(tpl);
        }

        fsInfo.release();
    }

    void Parser::parseTemplate(Template &tpl, const std::string &basePath) {
        // Load source images
        cv::Mat srcRGB = cv::imread(basePath + "rgb/" + tpl.fileName + ".png", CV_LOAD_IMAGE_COLOR);
        cv::Mat srcDepth = cv::imread(basePath + "depth/" + tpl.fileName + ".png", CV_LOAD_IMAGE_UNCHANGED);
        assert(srcRGB.type() == CV_8UC3);
        assert(srcDepth.type() == CV_16U);

        cv::Mat srcHSV, srcGray;
        cv::cvtColor(srcRGB, srcHSV, CV_BGR2HSV);
        cv::cvtColor(srcRGB, srcGray, CV_BGR2GRAY);

        // Generate quantized orientations
        cv::Mat gradients, magnitudes;
        quantizedOrientationGradients(srcGray, gradients, magnitudes);

        // Save images to template object
        tpl.srcRGB = std::move(srcRGB);
        tpl.srcGray = std::move(srcGray);
        tpl.srcHSV = std::move(srcHSV);
        tpl.srcDepth = std::move(srcDepth);
        tpl.srcGradients = std::move(gradients);

        // Parse criteria from template and extract normals and gradient images
        parseCriteriaAndNormals(tpl);
    }

    void Parser::parseCriteriaAndNormals(Template &tpl) {
        // Parse largest area
        if (tpl.objBB.area() < criteria->info.smallestTemplate.area()) {
            criteria->info.smallestTemplate = tpl.objBB.size();
        }

        if (tpl.objBB.width > criteria->info.largestArea.width) {
            criteria->info.largestArea.width = tpl.objBB.width;
        }

        if (tpl.objBB.height > criteria->info.largestArea.height) {
            criteria->info.largestArea.height = tpl.objBB.height;
        }

        // Find max/min depth and max local depth for depth quantization
        ushort localMax = tpl.srcDepth.at<ushort>(tpl.objBB.tl());
        auto localMin = static_cast<ushort>(-1);
        const int areaOffset = 0;

        // Offset bounding box so we cover edges of object
        for (int y = tpl.objBB.tl().y - areaOffset; y < tpl.objBB.br().y + areaOffset; y++) {
            for (int x = tpl.objBB.tl().x - areaOffset; x < tpl.objBB.br().x + areaOffset; x++) {
                ushort val = tpl.srcDepth.at<ushort>(y, x);

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

        // TODO - Better minMag value definition here and in objectness handling
        // Extract min edgels
        cv::Mat integral;
        depthEdgelsIntegral(tpl.srcDepth, integral, localMin, localMax);

        // Cover little bit larger area than object bounding box, to count edges
        cv::Point A(tpl.objBB.tl().x - areaOffset, tpl.objBB.tl().y - areaOffset);
        cv::Point B(tpl.objBB.br().x + areaOffset, tpl.objBB.tl().y - areaOffset);
        cv::Point C(tpl.objBB.tl().x - areaOffset, tpl.objBB.br().y + areaOffset);
        cv::Point D(tpl.objBB.br().x + areaOffset, tpl.objBB.br().y + areaOffset);

        // Get edgel count inside obj bounding box
        int edgels = integral.at<int>(D) - integral.at<int>(B) - integral.at<int>(C) + integral.at<int>(A);

        if (edgels < criteria->info.minEdgels) {
            criteria->info.minEdgels = edgels;
        }

        // Compute normals
        quantizedNormals(tpl.srcDepth, tpl.srcNormals, tpl.camera.fx(), tpl.camera.fy(), localMax, criteria->maxDepthDiff);
    }

    Scene Parser::parseScene(const std::string &basePath, int index, float scale) {
        Scene scene;
        std::ostringstream oss;
        oss << std::setw(4) << std::setfill('0') << index;
        oss << ".png";

        // Load depth, hsv, gray images
        scene.scale = scale;
        scene.srcRGB = cv::imread(basePath + "rgb/" + oss.str(), CV_LOAD_IMAGE_COLOR);
        scene.srcDepth = cv::imread(basePath + "depth/" + oss.str(), CV_LOAD_IMAGE_UNCHANGED);

        // Resize based on scale
        if (scale != 1.0f) {
            cv::resize(scene.srcRGB, scene.srcRGB, cv::Size(), scale, scale);
            cv::resize(scene.srcDepth, scene.srcDepth, cv::Size(), scale, scale);
        }

        // Convert to gray and hsv
        cv::cvtColor(scene.srcRGB, scene.srcGray, CV_BGR2GRAY);
        cv::cvtColor(scene.srcRGB, scene.srcHSV, CV_BGR2HSV);

        // Load camera
        std::string infoIndex = "scene_" + std::to_string(index);
        cv::FileStorage fs(basePath + "info.yml", cv::FileStorage::READ);
        cv::FileNode infoNode = fs[infoIndex];

        Camera camera;
        std::vector<float> vCamK, vCamRw2c, vCamTw2c;

        infoNode["cam_K"] >> vCamK;
        infoNode["cam_R_w2c"] >> vCamRw2c;
        infoNode["cam_t_w2c"] >> vCamTw2c;
        infoNode["elev"] >> camera.elev;
        infoNode["mode"] >> camera.mode;

        camera.K = cv::Mat(3, 3, CV_32FC1, vCamK.data());
        camera.R = cv::Mat(3, 3, CV_32FC1, vCamRw2c.data());
        camera.t = cv::Mat(3, 1, CV_32FC1, vCamTw2c.data());
        scene.camera = std::move(camera);

        // Generate quantized normals and orientations
        if (criteria != nullptr) {
            float ratio = depthNormalizationFactor(criteria->info.maxDepth * scale, criteria->depthDeviationFun);
            quantizedOrientationGradients(scene.srcGray, scene.gradients, scene.magnitudes);
            quantizedNormals(scene.srcDepth, scene.normals, scene.camera.fx(), scene.camera.fy(),
                             static_cast<int>((criteria->info.maxDepth * scale) / ratio), criteria->maxDepthDiff);
        }

        return scene;
    }
}