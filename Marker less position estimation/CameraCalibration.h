#ifndef LOAD_ALL
#define LOAD_ALL
#include <opencv2\opencv.hpp>
#endif

class CameraCalibration {
public:
	CameraCalibration();
	CameraCalibration(float fx, float fy, float cx, float cy);//(cx, cy) is principal point
	CameraCalibration(float fx, float fy, float cx, float cy, float distCoeff[4]);
	CameraCalibration(const cv::Mat&, const cv::Mat&);
	cv::Mat m_intrinsic;
	cv::Mat m_distortion;
};