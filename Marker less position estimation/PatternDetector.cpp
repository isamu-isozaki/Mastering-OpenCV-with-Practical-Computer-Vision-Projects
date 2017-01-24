#include "PatternDetector.h"

void PatternDetector::getGray(const cv::Mat& inputImg, cv::Mat& grayImg) {
	cv::cvtColor(inputImg, grayImg, cv::COLOR_RGB2GRAY);
}

void PatternDetector::setMatcher(const int& type) {
	switch (type){
	case 0://BFMatcher
		m_matcher = new cv::BFMatcher(cv::NORM_HAMMING, true);
		break;
	case 1:
		m_matcher = new cv::FlannBasedMatcher();
		break;
	}
}

void PatternDetector::extractFeatures(const cv::Mat& inputImg, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptors) {
	m_detector = cv::Ptr<cv::FeatureDetector>(new cv::OrbFeatureDetector());
	m_extractor = cv::Ptr<cv::DescriptorExtractor>(new cv::OrbDescriptorExtractor());

	m_detector->detect(inputImg, keyPoints);
	m_extractor->compute(inputImg, keyPoints, descriptors);
}

void PatternDetector::train(const Pattern& pattern) {
	m_matcher->clear();//clear the old training data
	std::vector<cv::Mat> descriptors(1);//with only one element
	//the vectors which are descriptors form a matrix
	descriptors[0] = pattern.descriptors.clone();
	m_matcher->add(descriptors);
	//after adding data, train the matcher
	m_matcher->train();
}

void PatternDetector::getMatch(const cv::Mat& queryImg, std::vector<cv::DMatch>& matches, const bool& show, const bool& apply) {
	std::vector < cv::KeyPoint> queryKeypoints;
	cv::Mat queryDescriptors;
	std::vector<std::vector<cv::DMatch> > kMatches;

	matches.clear();
	if (enableRatioTest) {
		const float maxRatio = 1.f / 1.5f;
		m_matcher->knnMatch(queryDescriptors, kMatches, 2);
		
		for (size_t i = 0; i < kMatches.size(); i++){
			const cv::DMatch& bestMatch = kMatches[i][0];
			const cv::DMatch& betterMatch = kMatches[i][1];

			const float distanceRatio = bestMatch.distance / betterMatch.distance;

			if (distanceRatio < maxRatio) {
				matches.push_back(bestMatch);
			}
		}
	}
	else{
		m_matcher->match(queryDescriptors, matches);
	}
	if (show){
		cv::Mat outImg;
		cv::namedWindow("drawMatchesWindow", CV_WINDOW_AUTOSIZE);
		cv::drawMatches(m_pattern.data, m_pattern.keypoints, queryImg, queryKeypoints, matches, outImg);
		cv::imshow("drawMatchesWindow", outImg);
		cv::waitKey(0);
		outImg.release();
		cv::destroyWindow("drawMatchesWindow");
	}
	if (apply) {
		PatternDetector::m_matches = matches;
	}
}

bool PatternDetector::refineMatches(const std::vector<cv::KeyPoint>& queryKeypoints, const std::vector<cv::KeyPoint>& trainKeypoints, const float& reprojectionThreshold, std::vector<cv::DMatch>& matches, cv::Mat& homography, const bool& apply){
	const int minNumMatchesAllowed = 8;

	if (matches.size() < minNumMatchesAllowed)
		return false;
	
	std::vector<cv::Point2f> srcPoints(matches.size());
	std::vector<cv::Point2f> dstPoints(matches.size());

	for (size_t i = 0; i < matches.size(); i++) {
		srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
		dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
	}

	std::vector<uchar> inlierMask(srcPoints.size());

	homography = cv::findHomography(srcPoints, dstPoints, inlierMask, CV_RANSAC, reprojectionThreshold);
	std::vector<cv::DMatch> inliers;
	for (size_t i = 0; i < inlierMask.size(); i++) {
		if (inlierMask[i])
			inliers.push_back(matches[i]);
	}

	matches.swap(inliers);
	return matches.size() > minNumMatchesAllowed;
	if (apply) {
		PatternDetector::m_matches = matches;
	}
}

void PatternDetector::makeTrainPattern(const cv::Mat& trainImg, Pattern& pattern, const bool& apply) {
	pattern.size = trainImg.size();
	pattern.frame = trainImg.clone();
	
	getGray(trainImg, pattern.data);

	pattern.point2d.resize(4);
	pattern.point3d.resize(4);

	const float w = trainImg.cols;
	const float h = trainImg.rows;

	const float maxSize = std::max(w, h);

	const float unitW = w / maxSize;
	const float unitH = h / maxSize;

	pattern.point2d[0] = cv::Point2f(0, 0);//considering from the center(uo, vo)
	pattern.point2d[1] = cv::Point2f(w, 0);
	pattern.point2d[2] = cv::Point2f(w, h);
	pattern.point2d[3] = cv::Point2f(0, h);

	pattern.point3d[0] = cv::Point3f(-unitW, -unitH, 0);//from the center(0, 0, 0)
	pattern.point3d[1] = cv::Point3f(unitW, -unitH, 0);
	pattern.point3d[2] = cv::Point3f(unitW, unitH, 0);
	pattern.point3d[3] = cv::Point3f(-unitW, unitH, 0);

	//Since (0,0,0) gets projected on to the image as (uo, vo)
	
	extractFeatures(pattern.data, pattern.keypoints, pattern.descriptors);	
	if (apply) {
		PatternDetector::m_pattern = pattern;
	}
}

void PatternDetector::prepareForFindPattern(const cv::Mat& trainImg) {
	makeTrainPattern(trainImg, m_pattern, true);
	
	setMatcher(0);

	train(m_pattern);
}

bool PatternDetector::findPattern(const std::string& trainFile , const std::string& queryFile, PatternTrackingInfo& info, const bool& prepare, const bool& apply) {
	if (prepare) {
		cv::Mat trainImg = cv::imread(trainFile);
		prepareForFindPattern(trainImg);
	}
	cv::Mat quertyImg = cv::imread(queryFile);
	
	getGray(queryImg, m_grayImg);
	
	extractFeatures(m_grayImg, m_queryKeyPoints, m_queryDescriptors);

	getMatch(m_queryDescriptors, m_matches, true, true);

	bool homographyFound = refineMatches(m_queryKeyPoints, m_pattern.keypoints, homographyReprojectionThreshold, m_matches, m_roughHomography, true);

	if (homographyFound) {
		info.homography = m_roughHomography;
		if (enableHomographyRefinement) {
			cv::warpPerspective(m_grayImg, m_warpedImg, m_roughHomography, m_pattern.size, cv::WARP_INVERSE_MAP | cv::INTER_CUBIC);

			std::vector<cv::KeyPoint> warpedKeypoints;
			std::vector<cv::DMatch> refinedMatches;

			extractFeatures(m_warpedImg, warpedKeypoints, m_queryDescriptors);

			getMatch(m_queryDescriptors, refinedMatches, true, true);

			homographyFound = refineMatches(warpedKeypoints, m_pattern.keypoints, homographyReprojectionThreshold, refinedMatches, m_refinedHomography, true);

			info.homography = m_roughHomography * m_refinedHomography;
		}

		cv::perspectiveTransform(m_pattern.point2d, info.point2d, info.homography);//output to info.point2d the four corners of the pattern after perspective transformation
		if (apply) {
			PatternDetector::info = info;
		}
	}



	return homographyFound;
}

void PatternDetector::estimatePosition(const Pattern& pattern, const CameraCalibration& calibration, cv::Mat& m_markerPose, const bool& apply) {
	cv::Mat_<float> rvec;
	cv::Mat_<float> tvec;
	cv::Mat rMat;
	cv::Mat m_extrinsic;

	cv::Mat raux, taux;
	cv::solvePnP(pattern.point3d, pattern.point2d, calibration.m_intrinsic, calibration.m_distortion, raux, taux);//camera loc

	raux.convertTo(rvec, CV_32F);
	taux.convertTo(tvec, CV_32F);

	rMat.resize(3, 3);
	cv::Rodrigues(rvec, rMat);//convert vector to matrix
	
	m_extrinsic.resize(3, 4);
	
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			m_extrinsic.at<cv::Point>(i, j) = rMat.at<cv::Point>(i, j);
		}
		m_extrinsic.at<cv::Point>(i, 3) = tvec.at<cv::Point>(i, 0);
	}
	cv::invert(m_extrinsic, m_markerPose);
	if (apply) {
		PatternDetector::calibration = calibration;
		PatternDetector::rMat = rMat;
		PatternDetector::tvec = tvec;
		PatternDetector::m_extrinsic = m_extrinsic;
		PatternDetector::m_markerPose = m_markerPose;
	}
}