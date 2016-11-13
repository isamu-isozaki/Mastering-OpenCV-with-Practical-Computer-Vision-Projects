#include "CameraCalibration.h"

CameraCalibration::CameraCalibration(const std::vector < std::string> fileNames , const cv::Size &imageSize): imgSize(imageSize){
	for (int i = 0; i < fileNames.size(); i++) {
		cv::Mat fileImage = cv::imread(fileNames[i], 0);
		findMarkers()
	}
	
	//cv::calibrateCamera(objectPoints, imagePoints, imageSize, m_intrinsic, m_distorsion, m_rotation, m_translation);


}