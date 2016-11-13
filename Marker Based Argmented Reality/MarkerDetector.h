#ifndef OPEN_CV_
#include <opencv2\opencv.hpp>
#endif

#ifndef CAMERA_CALIBRATION
#include "CameraCalibration.h"
#endif

#ifndef BGRA
#include "BGRAVideoFrame.h"
#endif

#ifndef GEOMETRY_TRANSFORMATION
#include "GeometryTransformation.h"
#endif

#ifndef MARKER
#include "Marker.h"
#endif

class MarkerDetector
{
public:
	MarkerDetector(std::vector<std::string> filenames, const cv::Size &imageSize);
	void processFrame(const BGRAVideoFrame& frame);
	const std::vector<Transformation>& getTransformations() const;
protected:
	bool findMarkers(const BGRAVideoFrame& frame, std::vector<Marker>& detectedMarkers);
	void prepareImage(const cv::Mat& bgraMat, cv::Mat& grayscale);
	void performThreshold(const cv::Mat& gratscale, cv::Mat& thresholdImg);
	void findCountours(const cv::Mat& thresholdImg, std::vector<std::vector<cv::Point>>& contours, int minContourPointsAllowed);
	void findMarkerCandidates(const std::vector<std::vector<cv::Point> >& contours, std::vector<Marker>& detectedMarkers);
	void detectMarkers(const cv::Mat& grayscale, std::vector<Marker>& detectedMarkers);
	void etimatePosition(std::vector<Marker>& detectedMarkers);
private:
	float m_minContourLengthAllowed;
	cv::Size markerSize;
	cv::Mat m_grayscale;
	cv::Mat m_threshold;
	cv::Mat m_contours;
	std::vector<std::vector<cv::Point>> p_contours;
	CameraCalibration calib;
};

