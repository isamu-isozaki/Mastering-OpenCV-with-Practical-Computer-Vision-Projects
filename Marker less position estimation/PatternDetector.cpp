#include "PatternDetector.h"

PatternDetector::PatternDetector() {

}

void PatternDetector::getGray(const cv::Mat& inputImg, cv::Mat& grayImg) {
	cv::cvtColor(inputImg, grayImg, cv::COLOR_RGB2GRAY);//convert the color
}

void PatternDetector::setMatcher(const int& type) {
	switch (type){
	case 0://BFMatcher
		m_matcher = new cv::BFMatcher(cv::NORM_HAMMING, false);// note to future self cannot do crossmatch and ratio test
		break;
	case 1:
		m_matcher = new cv::FlannBasedMatcher();
		break;
	}
}

void PatternDetector::extractFeatures(const cv::Mat& inputImg, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptors) {
	m_detector = cv::Ptr<cv::FeatureDetector>(new cv::OrbFeatureDetector());
	m_extractor = cv::Ptr<cv::DescriptorExtractor>(new cv::OrbDescriptorExtractor());

	m_detector->detect(inputImg, keyPoints);//detect keypoints
	m_extractor->compute(inputImg, keyPoints, descriptors);//compute descriptors
}

void PatternDetector::train(const Pattern& pattern) {//train the matcher
	m_matcher->clear();//clear the old training data
	std::vector<cv::Mat> descriptors(1);//with only one element
	//the vectors which are descriptors form a matrix
	descriptors[0] = pattern.descriptors.clone();
	m_matcher->add(descriptors);
	//after adding data, train the matcher
	m_matcher->train();
}

void PatternDetector::getMatch(cv::Mat& queryDescriptors, std::vector<cv::DMatch>& matches, const bool& apply) {
	std::vector<std::vector<cv::DMatch> > kMatches;

	matches.clear();
	if (enableRatioTest) {//if ratio test is enabled
		const float maxRatio = 1.f / 1.2f;//tried with 1.f/1.5f yet it miserably failed as it only outputted one match
		m_matcher->knnMatch(queryDescriptors, kMatches, 2);
		
		for (size_t i = 0; i < kMatches.size(); i++){
			const cv::DMatch& bestMatch = kMatches[i][0];
			const cv::DMatch& betterMatch = kMatches[i][1];

			const float distanceRatio = bestMatch.distance / betterMatch.distance;

			if (distanceRatio < maxRatio) {//if the ratio is smaller
				matches.push_back(bestMatch);//add to the match
			}
		}
	}
	else{
		m_matcher->match(queryDescriptors, matches);
	}
	if (apply) {
		PatternDetector::m_matches = matches;
	}
}

bool PatternDetector::refineMatches(const std::vector<cv::KeyPoint>& queryKeypoints, const std::vector<cv::KeyPoint>& trainKeypoints, const float& reprojectionThreshold, std::vector<cv::DMatch>& matches, cv::Mat& homography, const bool& apply){
	const int minNumMatchesAllowed = 8;//minimum requirement to compute homography matrix

	if (matches.size() < minNumMatchesAllowed)
		return false;
	
	std::vector<cv::Point2f> srcPoints(matches.size());
	std::vector<cv::Point2f> dstPoints(matches.size());

	for (size_t i = 0; i < matches.size(); i++) {
		srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
		dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
	}

	std::vector<uchar> inlierMask(srcPoints.size());

	homography = cv::findHomography(srcPoints, dstPoints, inlierMask, CV_RANSAC, reprojectionThreshold);//the outlier mask are the ones that cannot be, homography wise

	std::vector<cv::DMatch> inliers;
	for (size_t i = 0; i < inlierMask.size(); i++) {
		if (inlierMask[i])//if it is an inliner
			inliers.push_back(matches[i]);
	}

	matches.swap(inliers);//swap the contents of matches with inliers
	std::cout << "homographt test's success:" << ((matches.size() > minNumMatchesAllowed)? "success": "failure") << std::endl;
	if (apply) {
		PatternDetector::m_matches = matches;
	}
	return matches.size() > minNumMatchesAllowed;
}

void PatternDetector::makeTrainPattern(const cv::Mat& trainImg, Pattern& pattern, const bool& apply) {
	pattern.size = trainImg.size();
	pattern.frame = trainImg.clone();
	
	getGray(trainImg, pattern.data);

	pattern.point2d.resize(4);
	pattern.point3d.resize(4);

	const float w = trainImg.cols;
	const float h = trainImg.rows;

	const float maxSize = std::max(w, h);//So that the half of each side will not be larger than one

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
		prepareForFindPattern(trainImg);//prepare the matcher
	}

	cv::Mat outImg;
	cv::namedWindow("drawMatchesWindow", CV_WINDOW_AUTOSIZE);

	cv::Mat queryImg = cv::imread(queryFile);

	getGray(queryImg, m_grayImg);//get gray image

	extractFeatures(m_grayImg, m_queryKeyPoints, m_queryDescriptors);//get keypoints and the descriptors

	getMatch(m_queryDescriptors, m_matches, true);//get the matches


	cv::drawMatches(m_grayImg, m_queryKeyPoints, m_pattern.data, m_pattern.keypoints, m_matches, outImg);//draw the matches
	cv::imshow("drawMatchesWindow", outImg);
	cv::waitKey(0);
	
	bool homographyFound = refineMatches(m_queryKeyPoints, m_pattern.keypoints, homographyReprojectionThreshold, m_matches, m_roughHomography, true);//if homography was found(homography matrix is in m_roughHomography

	if (homographyFound) {
		info.homography = m_roughHomography;
		if (enableHomographyRefinement) {
			cv::warpPerspective(m_grayImg, m_warpedImg, m_roughHomography, m_pattern.size, cv::WARP_INVERSE_MAP | cv::INTER_CUBIC);

			std::vector<cv::KeyPoint> warpedKeypoints;
			std::vector<cv::DMatch> refinedMatches;

			extractFeatures(m_warpedImg, warpedKeypoints, m_queryDescriptors);

			getMatch(m_queryDescriptors, refinedMatches, true);//get the matches where the pattern and the destination image has the same perspective transformation

			homographyFound = refineMatches(warpedKeypoints, m_pattern.keypoints, homographyReprojectionThreshold, refinedMatches, m_refinedHomography, true);//only return inliers

			info.homography = m_roughHomography * m_refinedHomography;//get homography matrix
		    cv::drawMatches(m_grayImg, m_queryKeyPoints, m_pattern.data, m_pattern.keypoints, refinedMatches, outImg);//draw
			cv::imshow("drawMatchesWindow", outImg);
			cv::waitKey(0);
			cv::warpPerspective(m_grayImg, m_warpedImg, info.homography, m_pattern.size, cv::WARP_INVERSE_MAP | cv::INTER_CUBIC);//as is evident here the homography matrix is messed up
			cv::imshow("drawMatchesWindow", m_warpedImg);
			cv::waitKey(0);
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
	cv::solvePnP(pattern.point3d, pattern.point2d, calibration.m_intrinsic, calibration.m_distortion, raux, taux);//get camera pose
	
	raux.convertTo(rvec, CV_32F);
	taux.convertTo(tvec, CV_32F);
	
	rMat = cv::Mat(cv::Size(3,3), CV_32F);
	cv::Rodrigues(rvec, rMat);//convert vector to matrix
	
	cv::Mat tped_tvec;
	cv::transpose(tvec, tped_tvec);//so that we can push it to the extrinsic matrix
	cv::Mat tped_rMat;
	cv::transpose(rMat, tped_rMat);//so that we can push it to the extrinsic matrix

		
	m_extrinsic.push_back(tped_rMat);

	m_extrinsic.push_back(tped_tvec);

	cv::transpose(m_extrinsic, m_extrinsic);//to make it return to its original state

	float bottomRow[4] = { 0, 0, 0, 1 };

	cv::Mat m_bRow = cv::Mat(1, 4, CV_32F, &bottomRow);//so that m_extrinsic is invertable

	m_extrinsic.push_back(m_bRow);
	cv::invert(m_extrinsic, m_markerPose);//get the marker pose
	if (apply) { 
		PatternDetector::calibration = calibration;
		PatternDetector::rMat = rMat;
		PatternDetector::tvec = tvec;
		PatternDetector::m_extrinsic = m_extrinsic;
		PatternDetector::m_markerPose = m_markerPose;
	}
}
