#include "template.h"

Template::Template(std::string fileName, cv::Rect bounds, cv::Mat &src, cv::Mat &srcDepth) {
    this->fileName = fileName;
    this->bounds = bounds;
    this->src = src;
    this->srcDepth = srcDepth;
}

void Template::print() {
    std::cout << "Filename: " << this->fileName << " Bounds:" << this->bounds << std::endl;
}
