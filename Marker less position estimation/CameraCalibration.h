#ifndef LOAD_ALL
#define LOAD_ALL
#include <opencv2\opencv.hpp>
#endif

//#ifndef CAMERA_CALIBRATION
//#define CAMERA_CALIBRATION
struct CameraCalibration {
	CameraCalibration();
	CameraCalibration(const cv::Mat&, const cv::Mat&);
	cv::Mat m_intrinsic;
	cv::Mat m_distortion;
};
//#endif