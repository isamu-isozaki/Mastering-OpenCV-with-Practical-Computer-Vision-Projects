#ifndef LOADED_ALL
#include <opencv2\opencv.hpp>
#include <iostream>
#define LOADED_ALL
#endif

//#ifndef CAMERA_CALIBRATION
//#define CAMERA_CALIBRATION
#include "CameraCalibration.h"

CameraCalibration::CameraCalibration(const cv::Mat& intrinsic, const cv::Mat& distortion):m_intrinsic(intrinsic), m_distortion(distortion) {

}

//#endif