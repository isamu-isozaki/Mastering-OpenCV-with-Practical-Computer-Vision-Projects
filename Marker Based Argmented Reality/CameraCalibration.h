#include "GeometryTransformation.h"
#ifndef OPEN_CV_
#include <opencv2\opencv.hpp>
#endif

class CameraCalibration
{
public:
	CameraCalibration();
	CameraCalibration(float fx, float fy, float cx, float cy);
	CameraCalibration(float fx, float fy, float cx, float cy, float distortionCoeff[4]);
	void getMatrix34(float cparam[3][4]) const;
	const cv::Mat& getIntrinsic() const;
	const cv::Mat&getEucridean() const;
	const cv::Mat& getDistortion() const;
private:
	cv::Mat m_intrinsic;
	cv::Mat m_rotation;
	cv::Mat m_translation;
	cv::Mat m_distorsion;
	std::vector<std::vector<cv::Point3f>> objPoints;
	std::vector<std::vector<cv::Point2f>> imgPoints;
	cv::Size imgSize;
};

