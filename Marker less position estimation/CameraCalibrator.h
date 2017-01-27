#ifndef LOAD_ALL
#define LOAD_ALL
#include <opencv2\opencv.hpp>
#endif

#include "CameraCalibration.h"
//a class to find the camera calibration(the intrinsic and distortion matrix)
class CameraCalibrator{
public:
	CameraCalibrator();
	std::vector<std::vector<cv::Point3f>> objPoints;//real world points
	std::vector<std::vector<cv::Point2f>> imgPoints;//corresponding points on the image
	CameraCalibration calibration;//output of the calibrator
	
	void findPoints_caliberate(const std::vector<std::string>&, CameraCalibration&);//this function depends on the functions below, and does the work of both of them
private:
	std::vector<cv::Mat> inputImgs;//stores the images given by argument for vector<string>
	cv::Size boardSize = cv::Size(4, 4);//size of chessboard, used for camera calibration
	int success = 0;//num of successful detections of img corners
	double reprojectionError = 0.0f;//returned by cv::cameraCalibrator upon computing the intrinsic and distortion matrix
	bool drawDetection = true;//upon calibration process whether to draw it or not

	int findImg_and_ObjPoints(const std::vector<std::string>&, const cv::Size&);//given the location of the images and the size of the board, compute the real world coordinates and the image coordinates
	double caliberate(const cv::Size&, const std::vector<std::vector<cv::Point3f>>&, const std::vector<std::vector<cv::Point2f>>&, CameraCalibration&);//obtain the matrices given the image points and the object points
};
