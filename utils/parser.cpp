#include "parser.h"
#include "../processing/processing.h"
#include "../objdetect/matcher.h"
#include "../objdetect/hasher.h"
#include "../core/classifier_criteria.h"

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

            // Load template images and generate gradients, normals, hue, gray images
            parseTemplate(tpl, basePath);
            templates.push_back(tpl);
        }

        fsInfo.release();
    }

    void Parser::parseTemplate(Template &t, const std::string &basePath) {
        // Load source images
        cv::Mat srcRGB = cv::imread(basePath + "rgb/" + t.fileName + ".png", CV_LOAD_IMAGE_COLOR);
        cv::Mat srcDepth = cv::imread(basePath + "depth/" + t.fileName + ".png", CV_LOAD_IMAGE_UNCHANGED);
        assert(srcRGB.type() == CV_8UC3);
        assert(srcDepth.type() == CV_16U);

        cv::Mat srcHSV, srcHue, srcGray;
        cv::cvtColor(srcRGB, srcHSV, CV_BGR2HSV);
        cv::cvtColor(srcRGB, srcGray, CV_BGR2GRAY);

        // Normalize HSV
        normalizeHSV(srcHSV, srcHue);

        // Generate quantized orientations
        cv::Mat gradients;
        quantizedGradients(srcRGB, gradients, criteria->minMagnitude);

        // Save images to template object
        t.srcRGB = std::move(srcRGB);
        t.srcGray = std::move(srcGray);
        t.srcHue = std::move(srcHue);
        t.srcDepth = std::move(srcDepth);
        t.srcGradients = std::move(gradients);

        // Parse criteria from template and extract normals and gradient images
        parseCriteriaAndNormals(t);
    }

    void Parser::parseCriteriaAndNormals(Template &t) {
        // Parse largest area and smallest areas
        if (t.objBB.area() < criteria->info.smallestTemplate.area()) { criteria->info.smallestTemplate = t.objBB.size(); }
        if (t.objBB.width > criteria->info.largestArea.width) { criteria->info.largestArea.width = t.objBB.width; }
        if (t.objBB.height > criteria->info.largestArea.height) { criteria->info.largestArea.height = t.objBB.height; }

        // Extract criteria depth extremes from local extremes
        if (t.maxDepth > criteria->info.maxDepth) { criteria->info.maxDepth = t.maxDepth; }
        if (t.minDepth < criteria->info.minDepth) { criteria->info.minDepth = t.minDepth; }

        // Extract smallest diameter
        if (t.diameter < criteria->info.smallestDiameter) { criteria->info.smallestDiameter = t.diameter; }

        // Normalize local max and min depths to define error corrected range
        int localMax = static_cast<int>(t.maxDepth / depthNormalizationFactor(t.maxDepth, criteria->depthDeviationFun));
        int localMin = static_cast<int>(t.minDepth * depthNormalizationFactor(t.minDepth, criteria->depthDeviationFun));

        // Extract min edgels
        cv::Mat integral, edgels;
        depthEdgels(t.srcDepth, edgels, localMin, localMax, static_cast<int>(criteria->objectnessDiameterThreshold * t.diameter * criteria->info.depthScaleFactor));
        cv::integral(edgels, integral, CV_32S);

        // Get objBB corners for sum area table calculation
        cv::Point A(t.objBB.tl().x, t.objBB.tl().y);
        cv::Point B(t.objBB.br().x, t.objBB.tl().y);
        cv::Point C(t.objBB.tl().x, t.objBB.br().y);
        cv::Point D(t.objBB.br().x, t.objBB.br().y);

        // Get edgel count inside obj bounding box
        int edgelsCount = integral.at<int>(D) - integral.at<int>(B) - integral.at<int>(C) + integral.at<int>(A);
        if (edgelsCount < criteria->info.minEdgels) {
            criteria->info.minEdgels = edgelsCount;
        }

        // Compute normals
        quantizedNormals(t.srcDepth, t.srcNormals, t.camera.fx(), t.camera.fy(), localMax, static_cast<int>(criteria->maxDepthDiff / t.resizeRatio));
    }

    Scene Parser::parseScene(const std::string &basePath, int index, float scale) {
        Scene scene;
        std::ostringstream oss;
        oss << std::setw(4) << std::setfill('0') << index;
        oss << ".png";

        // Load depth, hue, gray images
        scene.id = static_cast<uint>(index);
        scene.scale = scale;
        scene.srcRGB = cv::imread(basePath + "rgb/" + oss.str(), CV_LOAD_IMAGE_COLOR);
        scene.srcDepth = cv::imread(basePath + "depth/" + oss.str(), CV_LOAD_IMAGE_UNCHANGED);

        // Resize based on scale
        if (scale != 1.0f) {
            cv::resize(scene.srcRGB, scene.srcRGB, cv::Size(), scale, scale);
            cv::resize(scene.srcDepth, scene.srcDepth, cv::Size(), scale, scale);

            // Recalculate depth values
            scene.srcDepth = scene.srcDepth / scale;
        }

        // Convert to gray and hsv
        cv::Mat hsv;
        cv::cvtColor(scene.srcRGB, scene.srcGray, CV_BGR2GRAY);
        cv::cvtColor(scene.srcRGB, hsv, CV_BGR2HSV);

        // Normalize HSV
        normalizeHSV(hsv, scene.srcHue);

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

        // Update intristic camera params based on current scale
        camera.K.at<float>(0, 0) *= scale;
        camera.K.at<float>(0, 2) *= scale;
        camera.K.at<float>(1, 1) *= scale;
        camera.K.at<float>(1, 2) *= scale;
        scene.camera = std::move(camera);

        // Generate quantized normals and orientations
        float ratio = depthNormalizationFactor(criteria->info.maxDepth, criteria->depthDeviationFun);
        quantizedGradients(scene.srcRGB, scene.srcGradients, criteria->minMagnitude);
        quantizedNormals(scene.srcDepth, scene.srcNormals, scene.camera.fx(), scene.camera.fy(),
                         static_cast<int>(criteria->info.maxDepth / ratio), static_cast<int>(criteria->maxDepthDiff / scale));

        return scene;
    }
}