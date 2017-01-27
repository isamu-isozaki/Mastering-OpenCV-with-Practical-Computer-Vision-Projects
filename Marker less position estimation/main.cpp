#ifndef LOADED_ALL
#include <opencv2\opencv.hpp>
#include <iostream>
#define LOADED_ALL
#endif

#include "PatternDetector.h"


int main(int argc, char** argv) {
	PatternDetector patternDetector;
	PatternTrackingInfo info;
	CameraCalibrator calibrator;
	CameraCalibration calibration;
	cv::Mat markerPose;


	std::vector<std::string> fileList;
	for (int i = 0; i < 20; i++){
		fileList.push_back("./calibrationImg/dst" + std::to_string(i) + ".jpg");
	}//get the images for the calibration

	calibrator.findPoints_caliberate(fileList, calibration);//get calibration params

	std::cout << "Intrinsic Matrix" << calibration.m_intrinsic << std::endl;
	std::cout << "Distortion Vector" << calibration.m_distortion << std::endl;



	patternDetector.findPattern("pattern.png", "dst.jpg", info, true, true);//detect the American flag in dst.jpg
	patternDetector.estimatePosition(patternDetector.m_pattern, calibration, markerPose, true);//get the position of dst.jpg

	std::cout << "homography matrix" << patternDetector.info.homography << std::endl;//currently disfunctional
	std::cout << "extrinsic matrix" << patternDetector.m_extrinsic << std::endl;
	std::cout << "marker pose" << patternDetector.m_markerPose << std::endl;//inverse of extrinsic matrix
	//marker pose now has the marker position
}