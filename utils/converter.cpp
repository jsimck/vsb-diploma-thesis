#include "converter.h"
#include "parser.h"
#include <cassert>
#include <fstream>

void tless::Converter::resizeAndSave(std::vector<tless::Template> &objectTemplates, const std::string &outputPath, int outputSize) {
    const cv::Size resizeSize(outputSize, outputSize);
    const int offset = 2;
    int index = 0;

    // Create base path
    std::ostringstream oss;
    oss << outputPath << std::setw(2) << std::setfill('0') << (objectTemplates[0].id / 2000) << "/";
    std::string basePath = oss.str();
    oss.str("");

//    cv::FileStorage fsInfo(basePath + "info.yml", cv::FileStorage::WRITE);
//    cv::FileStorage fsGt(basePath + "gt.yml", cv::FileStorage::WRITE);
//    fsInfo << "templates" << "[";

    for (auto &t : objectTemplates) {
        // Offset bounding box to preserve edges in resulted image
        cv::Rect offsetBB(t.objBB.x - offset, t.objBB.y - offset, t.objBB.width + (offset * 2), t.objBB.height + (offset * 2));

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
        float resizeRatio = outputSize / static_cast<float>(offsetBB.width);
        cv::Mat resizedRGB = t.srcRGB(offsetBB);
        cv::Mat resizedDepth = t.srcDepth(offsetBB);

        cv::resize(resizedRGB, resizedRGB, resizeSize, CV_INTER_CUBIC);
        cv::resize(resizedDepth, resizedDepth, resizeSize, CV_INTER_CUBIC);
        resizedDepth = resizedDepth / resizeRatio;

        // Checks
        assert(resizedRGB.type() == CV_8UC3);
        assert(resizedDepth.type() == CV_16UC1);
        assert(resizeRatio > 0);
        assert(resizedRGB.cols == outputSize);
        assert(resizedRGB.rows == outputSize);
        assert(resizedDepth.cols == outputSize);
        assert(resizedDepth.rows == outputSize);

        // Form output path and write results
        cv::imwrite(basePath + "rgb/" + t.fileName  + ".png", resizedRGB);
        cv::imwrite(basePath + "depth/" + t.fileName  + ".png", resizedDepth);

//        fsInfo << t;
//        index++;
    }

//    fsInfo << "]";
//    fsInfo.release();
//    fsGt.release();
}

void tless::Converter::convert(std::string templatesListPath, std::string outputPath, std::string modelsPath, int outputSize) {
    std::ifstream ifs(templatesListPath);
    assert(ifs.is_open());

    // Init parser and common
    Parser parser;
    std::vector<Template> objectTemplates;
    std::ostringstream oss;
    std::string path;

    while (ifs >> path) {
        // Parse each object by one and save it
        parser.parseObject(path, modelsPath, objectTemplates);

        // Resize object templates and recalculate their depth values
        resizeAndSave(objectTemplates, outputPath, outputSize);

        objectTemplates.clear();
    }

    ifs.close();
}
