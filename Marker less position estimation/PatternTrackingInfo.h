#ifndef LOADED_ALL
#include <opencv2\opencv.hpp>
#include <iostream>
#define LOADED_ALL
#endif

//contains homography and the points after they had underwent perspective control

struct PatternTrackingInfo {
	PatternTrackingInfo();
	cv::Mat homography;
	std::vector<cv::Point2f> point2d;
};
