#ifndef LOADED_ALL
#include <opencv2\opencv.hpp>
#define LOADED_ALL
#endif

#include "CameraCalibrator.h"
//The pattern class
//holds the image, list of features and extracted descriptors and 2d 3d correspondents of the intial pattern positions(of the training image)

struct Pattern {
	Pattern();//the default constructor
	cv::Size size;//the size of the image
	cv::Mat frame;
	cv::Mat data;//gray pattern image
	std::vector<cv::KeyPoint> keypoints;//keypoints of data
	cv::Mat descriptors;//descriptors of data
	std::vector<cv::Point2f> point2d;//the 2d points representing the corners of data
	std::vector<cv::Point3f> point3d;//the 3d points representing the real world points corresponding to the points on the corners of data, set Z=0
};
