#ifndef LOADED_ALL
#include <opencv2\opencv.hpp>
#include <iostream>
#define LOADED_ALL
#endif

#ifndef PATTERN_INFO
#define PATTERN_INFO
class PatternTrackingInfo {
public:
	cv::Mat homography;
	std::vector<cv::Point2f> point2d;
};
#endif