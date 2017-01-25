#ifndef LOADED_ALL
#include <opencv2\opencv.hpp>
#define LOADED_ALL
#endif

#ifndef PATTERN
#define PATTERN
#include "CameraCalibrator.h"

struct Pattern {//holds the image, list of features and extracted descriptors and 2d 3d correspondents of the intial pattern positions
	Pattern(const cv::Mat&, const cv::Mat&, const cv::Size&, const std::vector<cv::KeyPoint>&, const cv::Mat&, const std::vector<cv::Point2f>&, const std::vector<cv::Point3f>&);
	cv::Size size;
	cv::Mat frame;
	cv::Mat data;//gray
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	std::vector<cv::Point2f> point2d;
	std::vector<cv::Point3f> point3d;
};

#endif