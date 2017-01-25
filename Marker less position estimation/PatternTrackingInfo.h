#ifndef LOADED_ALL
#include <opencv2\opencv.hpp>
#include <iostream>
#define LOADED_ALL
#endif

struct PatternTrackingInfo {
	PatternTrackingInfo();
	cv::Mat homography;
	std::vector<cv::Point2f> point2d;
};
