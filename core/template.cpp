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

void Template::visualize(cv::Mat &result) {
    result = srcHSV.clone();
    cv::cvtColor(srcHSV, result, CV_HSV2BGR);

    // Draw edge points
    if (!edgePoints.empty()) {
        for (auto &point : edgePoints) {
            cv::circle(result, point, 1, cv::Scalar(0, 0, 255), -1);
        }
    }

    // Draw stable points
    if (!stablePoints.empty()) {
        for (auto &point : stablePoints) {
            cv::circle(result, point, 1, cv::Scalar(255, 0, 0), -1);
        }
    }

    // Draw bounding box
    cv::rectangle(result, objBB.tl(), objBB.br(), cv::Scalar(255 ,255, 255), 1);

    // Put text data
    std::ostringstream oss;
    oss << "votes: " << votes;
    cv::putText(result, oss.str(), objBB.tl() + cv::Point(objBB.width + 5, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255 ,255, 255), 1, CV_AA);
    oss.str("");
    oss << "mode: " << mode;
    cv::putText(result, oss.str(), objBB.tl() + cv::Point(objBB.width + 5, 28), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255 ,255, 255), 1, CV_AA);
    oss.str("");
    oss << "elev: " << elev;
    cv::putText(result, oss.str(), objBB.tl() + cv::Point(objBB.width + 5, 46), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255 ,255, 255), 1, CV_AA);
    oss.str("");
    oss << "gradients: " << features.gradients.size();
    cv::putText(result, oss.str(), objBB.tl() + cv::Point(objBB.width + 5, 64), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255 ,255, 255), 1, CV_AA);
    oss.str("");
    oss << "normals: " << features.normals.size();
    cv::putText(result, oss.str(), objBB.tl() + cv::Point(objBB.width + 5, 82), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255 ,255, 255), 1, CV_AA);
    oss.str("");
    oss << "depths: " << features.depths.size();
    cv::putText(result, oss.str(), objBB.tl() + cv::Point(objBB.width + 5, 100), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255 ,255, 255), 1, CV_AA);
    oss.str("");
    oss << "colors: " << features.colors.size();
    cv::putText(result, oss.str(), objBB.tl() + cv::Point(objBB.width + 5, 118), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255 ,255, 255), 1, CV_AA);
}
