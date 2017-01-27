#ifndef LOADED_ALL
#include <opencv2\opencv.hpp>
#define LOADED_ALL
#endif

//#ifndef PATTERN_DETECTOR
//#define PATTERN_DETECTOR
#include "Pattern.h"
#include "PatternTrackingInfo.h"

class PatternDetector{
public:
	PatternDetector();
	void extractFeatures(const cv::Mat&, std::vector<cv::KeyPoint>&, cv::Mat&);//obtain the keypoints and the descriptors given the image
	void train(const Pattern&);//train the matcher
	void getMatch(cv::Mat&, std::vector<cv::DMatch>&, const bool&);//From input image return matches depending on the type specified 0:match(), 1:knnMatch(), 2:radiusMatch
	bool refineMatches(const std::vector<cv::KeyPoint>&, const std::vector<cv::KeyPoint>&, const float&, std::vector<cv::DMatch>&, cv::Mat&, const bool&);//refine the matches by using homography and only returning inliners
	void prepareForFindPattern(const cv::Mat&);//Makes pattern, sets marker and trains the marker
	bool findPattern(const std::string&, const std::string&, PatternTrackingInfo&, const bool&, const bool&);//finds the pattern in the query image and stores the homography matrix and the transformed 2d image points in 
	void estimatePosition(const Pattern&, const CameraCalibration&, cv::Mat&, const bool&);//compute the marker pose and the extrinsic matrix given the pattern and the camera calibration
	
	Pattern m_pattern;
	PatternTrackingInfo info;
	cv::Mat m_markerPose;
	cv::Mat m_extrinsic;
private:
	
	cv::Mat m_grayImg;
	cv::Mat m_warpedImg;
	CameraCalibration calibration;

	cv::Ptr<cv::FeatureDetector> m_detector;
	cv::Ptr<cv::DescriptorExtractor> m_extractor;
	cv::Ptr<cv::DescriptorMatcher> m_matcher;

	std::vector<cv::KeyPoint> m_queryKeyPoints;
	cv::Mat m_queryDescriptors;
	std::vector<cv::DMatch> m_matches;
	cv::Mat m_roughHomography;
	cv::Mat m_refinedHomography;

	cv::Mat rMat;
	cv::Mat tvec;
	
	

	bool enableRatioTest = true;
	bool enableHomographyRefinement = true;
	float homographyReprojectionThreshold = 3;

	void getGray(const cv::Mat&, cv::Mat&);
	void setMatcher(const int&);//0:BFMatcher with crosscheck, 1:FlannBasedMatcher
	void makeTrainPattern(const cv::Mat& train, Pattern& pattern, const bool& apply);
};
