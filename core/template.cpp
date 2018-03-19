#include "template.h"

namespace tless {
    bool Template::operator==(const Template &rhs) const {
        return id == rhs.id;
    }

    bool Template::operator!=(const Template &rhs) const {
        return !(rhs == *this);
    }

    std::ostream &operator<<(std::ostream &os, const Template &t) {
        os << "Template ID: " << t.id << std::endl
           << "objId: " << t.objId << std::endl
           << "fileName: " << t.fileName << std::endl
           << "diameter: " << t.diameter << std::endl
           << "srcGray (size): " << t.srcGray.size() << std::endl
           << "srcRGB (size): " << t.srcRGB.size() << std::endl
           << "srcDepth (size): " << t.srcDepth.size() << std::endl
           << "srcGradients (size): " << t.srcGradients.size() << std::endl
           << "srcNormals (size): " << t.srcNormals.size() << std::endl
           << "resizeRatio: " << t.resizeRatio << std::endl
           << "objBB: " << t.objBB << std::endl
           << "objArea: " << t.objArea << std::endl
           << "minDepth: " << t.minDepth << std::endl
           << "maxDepth: " << t.maxDepth << std::endl
           << "camera: " << t.camera << std::endl
           << "features - srcGradients size: " << t.features.gradients.size() << std::endl
           << "features - srcNormals size: " << t.features.normals.size() << std::endl
           << "features - depths size: " << t.features.depths.size() << std::endl
           << "features - hue size: " << t.features.hue.size() << std::endl;

        return os;
    }

    void operator>>(const cv::FileNode &node, Template &t) {
        int id, objId;
        node["objId"] >> objId;
        node["id"] >> id;
        t.objId = static_cast<uint>(objId);
        t.id = static_cast<uint>(id);
        t.fileName = (std::string) node["fileName"];
        node["diameter"] >> t.diameter;
        node["edgePoints"] >> t.edgePoints;
        node["stablePoints"] >> t.stablePoints;
        node["depthMedian"] >> t.features.depthMedian;
        node["srcGradients"] >> t.features.gradients;
        node["srcNormals"] >> t.features.normals;
        node["depths"] >> t.features.depths;
        node["hue"] >> t.features.hue;
        node["objBB"] >> t.objBB;
        node["objArea"] >> t.objArea;
        node["minDepth"] >> t.minDepth;
        node["maxDepth"] >> t.maxDepth;
        node["resizeRatio"] >> t.resizeRatio;
        node["camera"] >> t.camera;
    }

    cv::FileStorage &operator<<(cv::FileStorage &fs, const Template &t) {
        fs << "{";
        fs << "objId" << static_cast<int>(t.objId);
        fs << "id" << static_cast<int>(t.id);
        fs << "fileName" << t.fileName;
        fs << "diameter" << t.diameter;

        if (t.edgePoints.size() > 0 && t.stablePoints.size() > 0) {
            fs << "edgePoints" << t.edgePoints;
            fs << "stablePoints" << t.stablePoints;
            fs << "depthMedian" << t.features.depthMedian;
            fs << "srcGradients" << t.features.gradients;
            fs << "srcNormals" << t.features.normals;
            fs << "depths" << t.features.depths;
            fs << "hue" << t.features.hue;
        }

        fs << "resizeRatio" << t.resizeRatio;
        fs << "objBB" << t.objBB;
        fs << "objArea" << t.objArea;
        fs << "minDepth" << t.minDepth;
        fs << "maxDepth" << t.maxDepth;
        fs << "camera" << t.camera;
        fs << "}";

        return fs;
    }
}