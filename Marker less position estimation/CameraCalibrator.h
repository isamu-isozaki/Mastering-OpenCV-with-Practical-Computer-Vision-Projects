#ifndef LOAD_ALL
#define LOAD_ALL
#include <opencv2\opencv.hpp>
#endif


//#ifndef CAMERA_CALIBRATOR
//#define CAMERA_CALIBRATOR
#include "CameraCalibration.h"
class CameraCalibrator{
public:
	CameraCalibrator();
	std::vector<std::vector<cv::Point3f>> objPoints;
	std::vector<std::vector<cv::Point2f>> imgPoints;
	CameraCalibration calibration;
	
	int findImg_and_ObjPoints(const std::vector<std::string>&, const cv::Size&);
	double caliberate(const cv::Size&, const std::vector<std::vector<cv::Point3f>>&, const std::vector<std::vector<cv::Point2f>>&, CameraCalibration&);
	void findPoints_caliberate(const std::vector<std::string>&, CameraCalibration&);

private:
	std::vector<cv::Mat> inputImgs;
	cv::Size boardSize = cv::Size(4, 4);
	bool drawDetection = true;
	int success = 0;
	double reprojectionError = 0.0f;
};
//#endif