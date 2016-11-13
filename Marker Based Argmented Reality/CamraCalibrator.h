#ifndef OPEN_CV_
#include <opencv2\opencv.hpp>
#endif

class CameraCalibrator {
	std::vector<std::vector<cv::Point3f>> objectPoints;
	std::vector<std::vector<cv::Point2f>> imagePoints;
};