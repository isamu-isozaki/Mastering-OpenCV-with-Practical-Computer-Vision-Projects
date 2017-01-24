#ifndef LOADED_ALL
#include <opencv2\opencv.hpp>
#include <iostream>
#define LOADED_ALL
#endif

#ifndef PATTERN
#include "PatternDetector.h"
#define PATTERN
#endif


int main(int argc, char** argv) {
	PatternDetector patternDetector;
	PatternTrackingInfo info;
	CameraCalibrator calibrator;
	CameraCalibration calibration;
	cv::Mat markerPose;

	std::vector<std::string> fileList;

	calibrator.findPoints_caliberate(fileList, calibration);
	
	patternDetector.findPattern("pattern.png", "dstImg.jpg", info, true, true);
	patternDetector.estimatePosition(patternDetector.m_pattern, calibration, markerPose, true);
	//marker pose now has the marker position
}