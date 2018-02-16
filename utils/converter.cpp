#include "converter.h"
#include "parser.h"
#include <cassert>
#include <fstream>

namespace tless {
    void Converter::resizeAndSave(Template &tpl, const std::string &outputPath, int outputSize) {
        const cv::Size resizeSize(outputSize, outputSize);
        const int offset = 2;

        // Offset bounding box to preserve edges in resulted image
        cv::Rect offsetBB(tpl.objBB.x - offset, tpl.objBB.y - offset, tpl.objBB.width + (offset * 2), tpl.objBB.height + (offset * 2));

        // Shift bounding box so object is in it's center + make sure bounding box is always equal to resizeSize
        if (offsetBB.width < offsetBB.height) {
            int x = (offsetBB.height - offsetBB.width) / 2;
            offsetBB.x -= x;
            offsetBB.width = offsetBB.height;
        } else {
            int y = (offsetBB.width - offsetBB.height) / 2;
            offsetBB.y -= y;
            offsetBB.height = offsetBB.width;
        }

        // Resize images by given ratio and recalculate depth values
        tpl.resizeRatio = outputSize / static_cast<float>(offsetBB.width);
        tpl.objBB = cv::Rect(0, 0, outputSize, outputSize); // update new objBB
        assert(tpl.resizeRatio > 0);

        cv::Mat resizedRGB = tpl.srcRGB(offsetBB);
        cv::Mat resizedDepth = tpl.srcDepth(offsetBB);

        int interpolation = (tpl.resizeRatio > 1.0f) ? CV_INTER_LANCZOS4 : CV_INTER_AREA;
        cv::resize(resizedRGB, resizedRGB, resizeSize, interpolation);
        cv::resize(resizedDepth, resizedDepth, resizeSize, interpolation);
        resizedDepth = resizedDepth / tpl.resizeRatio;

        // Validate that conversion didn't change image format
        assert(resizedRGB.type() == CV_8UC3);
        assert(resizedDepth.type() == CV_16UC1);
        assert(resizedRGB.cols == outputSize);
        assert(resizedRGB.rows == outputSize);
        assert(resizedDepth.cols == outputSize);
        assert(resizedDepth.rows == outputSize);

        // Write results
        cv::imwrite(outputPath + "rgb/" + tpl.fileName + ".png", resizedRGB);
        cv::imwrite(outputPath + "depth/" + tpl.fileName + ".png", resizedDepth);
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

        // Create template
        Template t;
        t.id = index + (2000 * id);
        t.fileName = std::move(fileName);
        t.diameter = diameters[id];
        t.srcRGB = std::move(srcRGB);
        t.srcDepth = std::move(srcDepth);
        t.objBB = cv::Rect(vObjBB[0], vObjBB[1], vObjBB[2], vObjBB[3]);
        t.camera = Camera(
            cv::Mat(3, 3, CV_32FC1, vCamRm2c.data()),
            cv::Mat(3, 1, CV_32FC1, vCamTm2c.data()),
            cv::Mat(3, 3, CV_32FC1, vCamK.data()),
            elev, azimuth, mode
        );

        return t;
    }

    void Converter::convert(const std::string &objectsListPath, const std::string &modelsInfoPath, std::string outputPath, int outputSize) {
        std::cout << "Converting..." << std::endl;

        // Parse object diameters if empty
        if (diameters.empty()) {
            cv::FileStorage fs(modelsInfoPath + "info.yml", cv::FileStorage::READ);
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

        // Parse each object in the objects list
        std::ifstream ifs(objectsListPath);
        assert(ifs.is_open());

        // Init parser and common
        std::vector<Template> templates;
        std::ostringstream oss;
        std::string basePath;

        while (ifs >> basePath) {
            // Load obj_gt and info for each object
            cv::FileStorage fsInfo(basePath + "info.yml", cv::FileStorage::READ);
            cv::FileStorage fsGt(basePath + "gt.yml", cv::FileStorage::READ);
            assert(fsInfo.isOpened());
            assert(fsGt.isOpened());

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
            oss << outputPath << std::setw(2) << std::setfill('0') << (templates[0].id / 2000) << "/";
            std::string modelOutputPath = oss.str();
            cv::FileStorage fsObjInfo(modelOutputPath + "info.yml.gz", cv::FileStorage::WRITE);

            // Resize source images and save object info
            fsObjInfo << "templates" << "[";
            for (auto &t : templates) {
                resizeAndSave(t, modelOutputPath, outputSize);
                fsObjInfo << t;
            }
            fsObjInfo << "]";

            oss.str("");
            fsObjInfo.release();
            templates.clear();

            std::cout << "  |_ model ID:" << (templates[0].id / 2000) << " converted, results saved to -> " << modelOutputPath << std::endl;
        }

        ifs.close();
        std::cout << "DONE!" << std::endl << std::endl;
    }
};