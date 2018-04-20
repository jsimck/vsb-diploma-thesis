#include "parser.h"
#include "../processing/processing.h"
#include "../objdetect/matcher.h"
#include "../objdetect/hasher.h"
#include "../core/classifier_criteria.h"
#include "timer.h"
#include "../processing/computation.h"

namespace tless {
    void Parser::parseObject(const std::string &basePath, std::vector<Template> &templates) {
        // Load object info.yml.gz at the root of each object folder
        cv::FileStorage fsInfo(basePath + "info.yml.gz", cv::FileStorage::READ);
        cv::FileNode tplNodes = fsInfo["templates"];
        objEdgels.clear();

        // Loop through each template node and parse info + images
        for (const auto &tplNode : tplNodes) {
            // Load template info
            Template tpl;
            tplNode >> tpl;

            // Load template images and generate gradients, normals, hue, gray images
            parseTemplate(tpl, basePath);
            templates.push_back(tpl);
        }

        // TODO validate sigma rule for other scenes
        // Calculate sd and mean to remove outliers
        removeOutliers<int>(objEdgels, 2);

        // Save min edgels
        std::stable_sort(objEdgels.begin(), objEdgels.end());
        if (objEdgels[0] < criteria->info.minEdgels && objEdgels[0] > 0) {
            criteria->info.minEdgels = objEdgels[0];
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
        quantizedGradients(srcGray, gradients, criteria->minMagnitude);

        // Save images to template object
        t.srcRGB = std::move(srcRGB);
        t.srcGray = std::move(srcGray);
        t.srcHue = std::move(srcHue);
        t.srcDepth = std::move(srcDepth);
        t.srcGradients = std::move(gradients);

        // Smooth out depth image
        cv::medianBlur(t.srcDepth, t.srcDepth, 5);

        // Parse criteria from template and extract normals and gradient images
        parseCriteriaAndNormals(t);
    }

    void Parser::parseCriteriaAndNormals(Template &t) {
        // Parse largest area and smallest areas
        if (t.objBB.area() < criteria->info.smallestTemplate.area()) { criteria->info.smallestTemplate = t.objBB.size(); }
        if (t.objBB.width > criteria->info.largestArea.width) { criteria->info.largestArea.width = t.objBB.width; }
        if (t.objBB.height > criteria->info.largestArea.height) { criteria->info.largestArea.height = t.objBB.height; }

        // Extract criteria depth extremes from local extremes + slightly increase min and max depth intervals
        if (t.maxDepth > criteria->info.maxDepth) { criteria->info.maxDepth = static_cast<ushort>(t.maxDepth + (t.maxDepth * 0.1f)); }
        if (t.minDepth < criteria->info.minDepth) { criteria->info.minDepth = static_cast<ushort>(t.minDepth - (t.minDepth * 0.1f)); }

        // Extract smallest diameter
        if (t.diameter < criteria->info.smallestDiameter) { criteria->info.smallestDiameter = t.diameter; }

        // Extract min edgels
        cv::Mat integral, edgels;
        depthEdgels(t.srcDepth, edgels, t.minDepth - 1000, t.maxDepth + 1000, static_cast<int>(criteria->objectnessDiameterThreshold * t.diameter * criteria->info.depthScaleFactor));
        cv::integral(edgels, integral, CV_32S);

        // Get objBB corners for sum area table calculation
        cv::Point A(t.objBB.tl().x, t.objBB.tl().y);
        cv::Point B(t.objBB.br().x, t.objBB.tl().y);
        cv::Point C(t.objBB.tl().x, t.objBB.br().y);
        cv::Point D(t.objBB.br().x, t.objBB.br().y);

        // Get edgel count inside obj bounding box
        int edgelsCount = integral.at<int>(D) - integral.at<int>(B) - integral.at<int>(C) + integral.at<int>(A);
        objEdgels.push_back(edgelsCount);

        // Compute normals
        cv::Mat normals3D;
        quantizedNormals(t.srcDepth, t.srcNormals, normals3D, t.camera.fx(), t.camera.fy(), t.maxDepth, static_cast<int>(criteria->maxDepthDiff / t.resizeRatio));
    }

    Scene Parser::parseScene(const std::string &basePath, int index, float scaleFactor, int levelsUp, int levelsDown) {
        Scene scene;
        std::ostringstream oss;
        oss << std::setw(4) << std::setfill('0') << index;
        oss << ".png";

        // Load Scene images
        scene.id = static_cast<uint>(index);
        cv::Mat srcRGB = cv::imread(basePath + "rgb/" + oss.str(), CV_LOAD_IMAGE_COLOR);
        cv::Mat srcDepth = cv::imread(basePath + "depth/" + oss.str(), CV_LOAD_IMAGE_UNCHANGED);

        // Load scene info
        std::string infoIndex = "scene_" + std::to_string(index);
        cv::FileStorage fs(basePath + "info.yml", cv::FileStorage::READ);
        cv::FileNode infoNode = fs[infoIndex];

        // Parse yml file
        int mode, elev;
        std::vector<float> vCamK, vCamRw2c, vCamTw2c;

        infoNode["cam_K"] >> vCamK;
        infoNode["cam_R_w2c"] >> vCamRw2c;
        infoNode["cam_t_w2c"] >> vCamTw2c;
        infoNode["elev"] >> elev;
        infoNode["mode"] >> mode;

        cv::Mat K = cv::Mat(3, 3, CV_32FC1, vCamK.data());
        cv::Mat R = cv::Mat(3, 3, CV_32FC1, vCamRw2c.data());
        cv::Mat t = cv::Mat(3, 1, CV_32FC1, vCamTw2c.data());
        fs.release();

        // Create gray and hsv images
        cv::Mat srcHSV, srcHue, srcGray;
        cv::cvtColor(srcRGB, srcGray, CV_BGR2GRAY);
        cv::cvtColor(srcRGB, srcHSV, CV_BGR2HSV);
        normalizeHSV(srcHSV, srcHue);

        // Reserve size for scene pyramid
        const int pyrSize = levelsDown + levelsUp + 1;
        scene.pyramid.resize(pyrSize);

        #pragma omp parallel for
        for (int i = 0; i < pyrSize; ++i) {
            float scale = 1.0f;
            if (i < levelsDown) {
                scale = 1.0f / std::pow(scaleFactor, levelsDown - i);
            } else if (i > levelsDown) {
                scale =  std::pow(scaleFactor, i - levelsDown);
            }

            scene.pyramid[i] = createPyramid(scale, srcRGB, srcDepth, srcGray, srcHue, K, R, t);
        }

        return scene;
    }

    ScenePyramid Parser::createPyramid(float scale, const cv::Mat &rgb, const cv::Mat &depth, const cv::Mat &gray, const cv::Mat &hue,
                                           const cv::Mat &K, const cv::Mat &R, const cv::Mat &t) {
        // Create camera
        Camera camera;
        camera.K = K.clone();
        camera.R = R.clone();
        camera.t = t.clone();

        // Recalculate K matrix based on scale
        camera.K.at<float>(0, 0) *= scale;
        camera.K.at<float>(0, 2) *= scale;
        camera.K.at<float>(1, 1) *= scale;
        camera.K.at<float>(1, 2) *= scale;

        // Create scene pyramid
        ScenePyramid pyramid(scale);
        pyramid.camera = std::move(camera);

        if (scale != 1.0f) {
            cv::resize(rgb, pyramid.srcRGB, cv::Size(), scale, scale, CV_INTER_CUBIC);
            cv::resize(depth, pyramid.srcDepth, cv::Size(), scale, scale, CV_INTER_AREA);
            cv::resize(gray, pyramid.srcGray, cv::Size(), scale, scale, CV_INTER_CUBIC);
            cv::resize(hue, pyramid.srcHue, cv::Size(), scale, scale, CV_INTER_CUBIC);

            // Recalculate depth values
            pyramid.srcDepth /= scale;
        } else {
            pyramid.srcRGB = rgb.clone();
            pyramid.srcDepth = depth.clone();
            pyramid.srcGray = gray.clone();
            pyramid.srcHue = hue.clone();
        }

        // Smooth out depth image
        cv::medianBlur(pyramid.srcDepth, pyramid.srcDepth, 5);

        // Generate quantized normals and orientations
        quantizedGradients(pyramid.srcGray, pyramid.srcGradients, criteria->minMagnitude);
        quantizedNormals(pyramid.srcDepth, pyramid.srcNormals, pyramid.srcNormals3D, pyramid.camera.fx(), pyramid.camera.fy(),
                         static_cast<int>(criteria->info.maxDepth), static_cast<int>(criteria->maxDepthDiff / scale));

        // Spread features
        spread(pyramid.srcNormals, pyramid.spreadNormals, criteria->patchOffset * 2 + 1);
        spread(pyramid.srcGradients, pyramid.spreadGradients, criteria->patchOffset * 2 + 1);

        return pyramid;
    }
}