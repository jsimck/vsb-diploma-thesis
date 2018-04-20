#include "converter.h"
#include "parser.h"
#include <cassert>
#include <fstream>
#include <boost/filesystem.hpp>

namespace tless {
    bool Converter::validateDepth(ushort depth, const cv::Mat &src, const cv::Point &p, int maxDiff, int ksize) {
        assert(src.type() == CV_16UC1);
        const int offset = ksize / 2;

        for (int y = p.x - offset; y < p.x + offset; y++) {
            for (int x = p.x - offset; x < p.x + offset; x++) {
                ushort px = src.at<ushort>(y, x);
                if (std::abs(px - depth) > maxDiff) {
                    return false;
                }
            }
        }

        return true;
    }

    void Converter::resizeAndSave(Template &t, const std::string &outputPath, int outputSize) {
        const cv::Size resizeSize(outputSize, outputSize);

        // Offset bounding box to preserve edges in resulted image
        cv::Rect offsetBB(t.objBB.x - this->offset, t.objBB.y - this->offset, t.objBB.width + (this->offset * 2), t.objBB.height + (this->offset * 2));

        // Shift bounding box so object is in it's center + make sure bounding box is always equal to resizeSize
        if (offsetBB.width < offsetBB.height) {
            int x = (offsetBB.height - offsetBB.width) / 2;
            offsetBB.x -= x;
            offsetBB.width = offsetBB.height;

            // Offset has to be larger than 0
            offsetBB.x = offsetBB.x < 0 ? 0 : offsetBB.x;
        } else {
            int y = (offsetBB.width - offsetBB.height) / 2;
            offsetBB.y -= y;
            offsetBB.height = offsetBB.width;

            // Offset has to be larger than 0
            offsetBB.y = offsetBB.y < 0 ? 0 : offsetBB.y;
        }

        // Shift to fit source image size
        if (offsetBB.x + offsetBB.width > t.srcRGB.cols) {
            offsetBB.x = t.srcRGB.cols - offsetBB.width;
        } else if (offsetBB.y + offsetBB.height > t.srcRGB.rows) {
            offsetBB.y = t.srcRGB.rows - offsetBB.height;
        }

        // Resize images by given ratio and recalculate depth values
        t.resizeRatio = outputSize / static_cast<float>(offsetBB.width);
        t.objBB = cv::Rect(0, 0, outputSize, outputSize); // update new objBB
        assert(t.resizeRatio > 0);

        cv::Mat resizedRGB = t.srcRGB(offsetBB);
        cv::Mat resizedDepth = t.srcDepth(offsetBB);

        int interpolation = (t.resizeRatio > 1.0f) ? CV_INTER_LANCZOS4 : CV_INTER_AREA;
        cv::resize(resizedRGB, resizedRGB, resizeSize, interpolation);
        cv::resize(resizedDepth, resizedDepth, resizeSize, interpolation);
        resizedDepth = resizedDepth / t.resizeRatio;

        // Adjust local depth extremes based on ratio
        t.minDepth /= t.resizeRatio;
        t.maxDepth /= t.resizeRatio;

        // Validate that conversion didn't change image format
        assert(resizedRGB.type() == CV_8UC3);
        assert(resizedDepth.type() == CV_16UC1);
        assert(resizedRGB.cols == outputSize);
        assert(resizedRGB.rows == outputSize);
        assert(resizedDepth.cols == outputSize);
        assert(resizedDepth.rows == outputSize);

        // Write results
        cv::imwrite(outputPath + "rgb/" + t.fileName + ".png", resizedRGB);
        cv::imwrite(outputPath + "depth/" + t.fileName + ".png", resizedDepth);

        // Extract object area
        cv::Mat resizedGray;
        cv::cvtColor(resizedRGB, resizedGray, CV_BGR2GRAY);

        for (int y = 0; y < resizedGray.rows; y++) {
            for (int x = 0; x < resizedGray.cols; x++) {
                if (resizedGray.at<uchar>(y, x) > this->minGray) {
                    t.objArea++;
                }
            }
        }

        // Normalize object area
        t.objArea /= resizeSize.area();

        // Adjust intristic camera parameters based on resize ratio
        t.camera.K.at<float>(0, 0) *= t.resizeRatio;
        t.camera.K.at<float>(0, 2) *= t.resizeRatio;
        t.camera.K.at<float>(1, 1) *= t.resizeRatio;
        t.camera.K.at<float>(1, 2) *= t.resizeRatio;
        t.camera.K.at<float>(0, 2) += (t.objBB.x - offsetBB.x) * t.resizeRatio;
        t.camera.K.at<float>(1, 2) += (t.objBB.y - offsetBB.y) * t.resizeRatio;
    }

    Template Converter::parseTemplate(uint index, const std::string &basePath, cv::FileNode &gtNode, cv::FileNode &infoNode) {
        int id, elev, mode, azimuth;
        std::vector<float> vCamRm2c, vCamTm2c, vCamK;
        std::vector<int> vObjBB;

        // Parse data from .yml files
        gtNode["obj_id"] >> id;
        gtNode["obj_bb"] >> vObjBB;
        gtNode["cam_R_m2c"] >> vCamRm2c;
        gtNode["cam_t_m2c"] >> vCamTm2c;
        infoNode["cam_K"] >> vCamK;
        infoNode["elev"] >> elev;
        infoNode["mode"] >> mode;
        azimuth = 5 * (index % 72); // Training templates are sampled in 5 step azimuth

        assert(!vObjBB.empty());
        assert(!vCamRm2c.empty());
        assert(!vCamTm2c.empty());

        // Create filename from index
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << index;
        std::string fileName = ss.str();

        // Load source images
        cv::Mat srcRGB = cv::imread(basePath + "rgb/" + fileName + ".png", CV_LOAD_IMAGE_COLOR);
        cv::Mat srcDepth = cv::imread(basePath + "depth/" + fileName + ".png", CV_LOAD_IMAGE_UNCHANGED);
        assert(srcRGB.type() == CV_8UC3);
        assert(srcDepth.type() == CV_16U);

        cv::Mat srcGray;
        cv::cvtColor(srcRGB, srcGray, CV_BGR2GRAY);

        // Create template
        Template t;
        t.objId = id;
        t.id = index + (10000 * id);
        t.fileName = std::move(fileName);
        t.diameter = diameters[id - 1];
        t.srcRGB = std::move(srcRGB);
        t.srcGray = std::move(srcGray);
        t.srcDepth = std::move(srcDepth);
        t.objBB = cv::Rect(vObjBB[0], vObjBB[1], vObjBB[2], vObjBB[3]);
        t.camera.R = cv::Mat(3, 3, CV_32FC1, vCamRm2c.data()).clone();
        t.camera.t = cv::Mat(3, 1, CV_32FC1, vCamTm2c.data()).clone();
        t.camera.K = cv::Mat(3, 3, CV_32FC1, vCamK.data()).clone();
        t.camera.elev = elev;
        t.camera.azimuth = azimuth;
        t.camera.mode = mode;

        // Extract local depth extremes, search in object bounding box
        auto depthDiameter = static_cast<int>(t.diameter * 10);

        for (int y = t.objBB.tl().y - this->offset; y < t.objBB.br().y + this->offset; y++) {
            for (int x = t.objBB.tl().x - this->offset; x < t.objBB.br().x + this->offset; x++) {
                if (t.srcGray.at<uchar>(y, x) > this->minGray) {
                    ushort depth = t.srcDepth.at<ushort>(y, x);

                    // Extract local max
                    if (depth > t.maxDepth && validateDepth(depth, t.srcDepth, cv::Point(x, y), depthDiameter, 5)) {
                        t.maxDepth = depth;
                    }

                    // Extract local min
                    if (depth < t.minDepth && depth > 0) {
                        t.minDepth = depth;
                    }
                }
            }
        }

        return t;
    }

    void Converter::convert(const std::string &templatesFolder, const std::string &modelsFolder,
                            const std::vector<int> &indices, std::string outputFolder, int outputSize) {
        std::cout << "Converting..." << std::endl;

        // Parse object diameters if empty
        if (diameters.empty()) {
            cv::FileStorage fs(modelsFolder + "info.yml", cv::FileStorage::READ);
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

        std::cout << "  |_ models info.yml parsed" << std::endl;

        // Init parser and common
        std::vector<Template> templates;
        std::string basePath;

        for (auto &objId : indices) {
            // Load obj_gt and info for each object
            basePath = cv::format((templatesFolder + "%02d/").c_str(), objId);
            cv::FileStorage fsInfo(basePath + "info.yml", cv::FileStorage::READ);
            cv::FileStorage fsGt(basePath + "gt.yml", cv::FileStorage::READ);

            // Parse templates
            for (uint i = 0;; i++) {
                std::string index = "tpl_" + std::to_string(i);
                cv::FileNode gtNode = fsGt[index];
                cv::FileNode infoNode = fsInfo[index];

                // Break if obj is empty (last template)
                if (gtNode.empty() || infoNode.empty()) break;

                // Parse template
                templates.push_back(parseTemplate(i, basePath, gtNode, infoNode));
            }

            fsInfo.release();
            fsGt.release();

            // Resize templates and save obj info
            std::string objectOutputPath = cv::format((outputFolder + "%02d/").c_str(), templates[0].objId);

            // Create directories
            boost::filesystem::create_directories(objectOutputPath);
            boost::filesystem::create_directories(objectOutputPath + "rgb");
            boost::filesystem::create_directories(objectOutputPath + "depth");

            // Save info and template files
            cv::FileStorage fsObjInfo(objectOutputPath + "info.yml.gz", cv::FileStorage::WRITE);

            // Resize source images and save object info
            fsObjInfo << "templates" << "[";
            for (auto &t : templates) {
                resizeAndSave(t, objectOutputPath, outputSize);
                fsObjInfo << t;
            }
            fsObjInfo << "]";

            fsObjInfo.release();
            templates.clear();

            std::cout << "  |_ model ID:" << templates[0].objId << " converted, results saved to -> " << objectOutputPath << std::endl;
        }

        std::cout << "DONE!" << std::endl << std::endl;
    }
}