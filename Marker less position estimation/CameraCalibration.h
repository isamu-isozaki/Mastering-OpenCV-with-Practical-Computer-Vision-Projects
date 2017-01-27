#ifndef LOAD_ALL
#define LOAD_ALL
#include <opencv2\opencv.hpp>
#endif

//holds the intrinsic and distortion matrix

struct CameraCalibration {
	CameraCalibration();
	CameraCalibration(const cv::Mat&, const cv::Mat&);
	cv::Mat m_intrinsic;//the intrinsic matrix
	cv::Mat m_distortion;//the distortion matrix
};