#include "template.h"

void Template::vote() {
    votes++;
}

void Template::resetVotes() {
    votes = 0;
}

void Template::applyROI() {
    // Apply roi to both sources
    srcGray = srcGray(objBB);
    srcHSV = srcHSV(objBB);
    srcDepth = srcDepth(objBB);
}

void Template::resetROI() {
    // Locate ROI
    cv::Point offset;
    cv::Size size;
    srcGray.locateROI(size, offset);

    // Set to original [disable ROI]
    srcGray.adjustROI(offset.y, size.height, offset.x, size.width);
    srcDepth.adjustROI(offset.y, size.height, offset.x, size.width);
    srcHSV.adjustROI(offset.y, size.height, offset.x, size.width);
}

void Template::save(cv::FileStorage &fs) {
    fs << "{";
    fs << "id" << id;
    fs << "fileName" << fileName;
    fs << "angles" << angles;
    fs << "edgePoints" << edgePoints;
    fs << "stablePoints" << stablePoints;
    fs << "gradients" << features.gradients;
    fs << "normals" << features.normals;
    fs << "depths" << features.depths;
    fs << "colors" << features.colors;
    fs << "objBB" << objBB;
    fs << "camK" << camK;
    fs << "camRm2c" << camRm2c;
    fs << "camTm2c" << camTm2c;
    fs << "elev" << elev;
    fs << "mode" << mode;
    fs << "}";
}

Template Template::load(cv::FileNode node) {
    Template t;

    node["id"] >> t.id;
    t.fileName = (std::string) node["fileName"];
    node["angles"] >> t.angles;
    node["edgePoints"] >> t.edgePoints;
    node["stablePoints"] >> t.stablePoints;
    node["gradients"] >> t.features.gradients;
    node["normals"] >> t.features.normals;
    node["depths"] >> t.features.depths;
    node["colors"] >> t.features.colors;
    node["objBB"] >> t.objBB;
    node["camK"] >> t.camK;
    node["camRm2c"] >> t.camRm2c;
    node["camTm2c"] >> t.camTm2c;
    node["elev"] >> t.elev;
    node["mode"] >> t.mode;

    return t;
}

bool Template::operator==(const Template &rhs) const {
    return id == rhs.id &&
           fileName == rhs.fileName &&
           elev == rhs.elev &&
           mode == rhs.mode;
}

bool Template::operator!=(const Template &rhs) const {
    return !(rhs == *this);
}

std::ostream &operator<<(std::ostream &os, const Template &t) {
    os << "Template ID: " << t.id << std::endl
       << "fileName: " << t.fileName << std::endl
       << "srcGray (size): " << t.srcGray.size()  << std::endl
       << "srcDepth (size): " << t.srcDepth.size() << std::endl
       << "objBB: " << t.objBB  << std::endl
       << "camK: " << t.camK  << std::endl
       << "camRm2c: " << t.camRm2c << std::endl
       << "camTm2c: " << t.camTm2c  << std::endl
       << "elev: " << t.elev  << std::endl
       << "mode: " << t.mode << std::endl
       << "votes: " << t.votes << std::endl
       << "gradients size: " << t.features.gradients.size() << std::endl
       << "normals size: " << t.features.normals.size() << std::endl
       << "depths size: " << t.features.depths.size() << std::endl
       << "colors size: " << t.features.colors.size() << std::endl;

    return os;
}