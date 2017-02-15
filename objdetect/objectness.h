#ifndef VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
#define VSB_SEMESTRAL_PROJECT_OBJECTNESS_H


#include <string>
#include "../core/template.h"

void generateBINGTrainingSet(std::string destPath, std::vector<Template> &templates);
void computeBING(std::string trainingPath, cv::Mat &scene, std::vector<cv::Vec4i> &resultBB);

cv::Vec4i edgeBasedObjectness(cv::Mat &scene, cv::Mat &sceneDepth, std::vector<Template> &templates);
void filterSobel(cv::Mat &src, cv::Mat &dst);

#endif //VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
