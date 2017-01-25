#ifndef LOADED_ALL
#include <opencv2\opencv.hpp>
#define LOADED_ALL
#endif

#ifndef PATTERN_DETECTOR
#define PATTERN_DETECTOR
#include "Pattern.h"
#include "PatternTrackingInfo.h"

class PatternDetector{
public:
	PatternDetector();
	void extractFeatures(const cv::Mat&, std::vector<cv::KeyPoint>&, cv::Mat&);
	void train(const Pattern&);//train matcher
	void getMatch(const cv::Mat&, std::vector<cv::DMatch>&, const bool&, const bool&);//From input image return matches depending on the type specified 0:match(), 1:knnMatch(), 2:radiusMatch
	bool refineMatches(const std::vector<cv::KeyPoint>&, const std::vector<cv::KeyPoint>&, const float&, std::vector<cv::DMatch>&, cv::Mat&, const bool&);
	void prepareForFindPattern(const cv::Mat& train);
	bool findPattern(const std::string&, const std::string&, PatternTrackingInfo&, const bool&, const bool&);
	void estimatePosition(const Pattern&, const CameraCalibration&, cv::Mat&, const bool&);
	
	Pattern m_pattern;
	cv::Mat m_markerPose;
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

	PatternTrackingInfo info;

	cv::Mat rMat;
	cv::Mat tvec;
	cv::Mat m_extrinsic;
	

	bool enableRatioTest = true;
	bool enableHomographyRefinement = true;
	float homographyReprojectionThreshold = 3;

	void getGray(const cv::Mat&, cv::Mat&);
	void setMatcher(const int&);//0:BFMatcher with crosscheck, 1:FlannBasedMatcher
	void makeTrainPattern(const cv::Mat& train, Pattern& pattern, const bool& apply);
};

#endif