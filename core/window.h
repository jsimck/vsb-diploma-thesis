#ifndef VSB_SEMESTRAL_PROJECT_WINDOW_H
#define VSB_SEMESTRAL_PROJECT_WINDOW_H

#include <opencv2/core/types.hpp>
#include "template.h"

struct Window {
public:
	cv::Point p;
	cv::Size size;
	std::vector<Template*> candidates;

	// Constructors
	Window(cv::Point p, cv::Size size) : p(p), size(size) {}
	Window(cv::Point p, cv::Size size, std::vector<Template*> candidates) : p(p), size(size), candidates(candidates) {}

	// Methods
	cv::Point tl();
	cv::Point tr();
	cv::Point bl();
	cv::Point br();
	bool hasCandidates();

	// Friends
	friend std::ostream& operator<<(std::ostream &os, const Window &w);
};

#endif //VSB_SEMESTRAL_PROJECT_WINDOW_H
