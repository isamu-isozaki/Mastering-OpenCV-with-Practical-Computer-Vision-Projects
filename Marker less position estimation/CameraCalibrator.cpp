#ifndef LOADED_ALL
#include <opencv2\opencv.hpp>
#include <iostream>
#define LOADED_ALL
#endif

#include "CameraCalibrator.h"

CameraCalibrator::CameraCalibrator() {

}

int CameraCalibrator::findImg_and_ObjPoints(const std::vector<std::string>& fileNames, const cv::Size& boardSize) {//find points
	objPoints.clear();
	imgPoints.clear();
	inputImgs.clear();

	std::vector<cv::Point3f> objCorners;
	std::vector<cv::Point2f> imgCorners;

	for (int i = 0; i < boardSize.height; i++){
		for (int j = 0; j < boardSize.width; j++){
			objCorners.push_back(cv::Point3f(i, j, 0.0f));//(0,0,0) all the way to (boardSize.height, boardSize.width, 0)
		}
	}

	cv::Mat img;
	cv::namedWindow("showDetection");
	int success = 0;
	for (size_t i = 0; i < fileNames.size(); i++) {
		img = cv::imread(fileNames[i], 0);
		bool found = cv::findChessboardCorners(img, boardSize, imgCorners);
		cv::cornerSubPix(img, imgCorners, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.1));
		if (imgCorners.size() == boardSize.area()){
			objPoints.push_back(objCorners);
			imgPoints.push_back(imgCorners);
			inputImgs.push_back(img);
			if (drawDetection) {
				cv::drawChessboardCorners(img, boardSize, imgCorners, found);
				cv::imshow("showDetection", img);
				cv::waitKey(0);
			}
			success++;
		}
	}
	return success;
}

double CameraCalibrator::caliberate(const cv::Size& boardSize, const std::vector<std::vector<cv::Point3f>>& objPoints, const std::vector<std::vector<cv::Point2f>>& imgPoints, CameraCalibration& caliberation) {
	std::vector<cv::Mat> rvec, tvec;
	return cv::calibrateCamera(objPoints, imgPoints, boardSize, caliberation.m_intrinsic, caliberation.m_distortion, rvec, tvec);//get calibration
}

void CameraCalibrator::findPoints_caliberate(const std::vector<std::string>& fileNames, CameraCalibration& calibration){
	findImg_and_ObjPoints(fileNames, CameraCalibrator::boardSize);

	caliberate(CameraCalibrator::boardSize, CameraCalibrator::objPoints, CameraCalibrator::imgPoints, calibration);
}