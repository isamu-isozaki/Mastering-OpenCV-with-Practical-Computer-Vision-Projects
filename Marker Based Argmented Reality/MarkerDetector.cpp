#include "MarkerDetector.h"
#include "Marker.h"

MarkerDetector::MarkerDetector(std::vector<std::string> filenames, const cv::Size &imageSize) {
	BGRAVideoFrame img;
	for (int i = 0; i < filenames.size(); i++){
		img.BGRA = cv::imread(filenames[i]);
		std::vector <Marker> imgCorners;
		if (!findMarkers(img, imgCorners))
			continue;

	}
}

void MarkerDetector::prepareImage(const cv::Mat& bgraMat, cv::Mat& grayscale) {
	cv::cvtColor(bgraMat, grayscale, CV_BGRA2GRAY);
}

void MarkerDetector::performThreshold(const cv::Mat& grayscale, cv::Mat& threshold) {
	cv::adaptiveThreshold(grayscale, threshold, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 7, 7);//change the final two arguments
}

void MarkerDetector::findCountours(const cv::Mat& thresholdImg, std::vector<std::vector<cv::Point>>& contours, int minContourPointsAllowed){
	std::vector<std::vector<cv::Point>> allContours;
	cv::findContours(thresholdImg, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);//list of all polygons reprenting each counter
	contours.clear();
	for (size_t i = 0; i < allContours.size(); i++) {
		int contourSize = allContours[i].size();
		if (contourSize > minContourPointsAllowed){
			contours.push_back(allContours[i]);
		}
	}
}

void MarkerDetector::findMarkerCandidates(const std::vector<std::vector<cv::Point> > & contours, std::vector<Marker>& detectedMarkers){
	std::vector<cv::Point> approxCurve;
	std::vector<Marker> possibleMarkers;

	for (size_t i = 0; i < contours.size(); i++){
		double eps = contours[i].size() * 0.05;
		cv::approxPolyDP(contours[i], approxCurve, eps, true);//eradicate unnecessary points
		if (approxCurve.size() != 4)
			continue;
		if (!cv::isContourConvex(approxCurve))
			continue;

		float minDist = std::numeric_limits<float>::max();
		for (int i = 0; i < 4; i++) {
			cv::Point side = approxCurve[i] - approxCurve[(i + 1) % 4];
			float squaredSideLength = side.dot(side);
			minDist = std::min(minDist, squaredSideLength);
		}
		if (minDist < m_minContourLengthAllowed)
			continue;
		Marker m;
		for (int i = 0; i < 4; i++) 
			m.points.push_back(cv::Point2f(approxCurve[i].x, approxCurve[i].y));
		cv::Point v1 = m.points[1] - m.points[0];
		cv::Point v2 = m.points[2] - m.points[0];
		double o = (v1.x * v2.y) - (v1.y * v2.x);
		if (o < 0.0) {
			std::swap(m.points[1], m.points[3]);
		}
		possibleMarkers.push_back(m);
	}
	std::vector<std::pair<int, int>> tooNearCandidates;
	for (size_t i = 0; i < possibleMarkers.size(); i++) {
		const Marker& ml = possibleMarkers[i];
		for (size_t j = 0; j < possibleMarkers.size(); j++) {
			const Marker& ml2 = possibleMarkers[j];
			float distSquared = 0;
			for (int c; c < 4; c++){
				cv::Point v = ml.points[c] - ml2.points[c];
				distSquared += v.dot(v);
			}
			distSquared /= 4;
			if (distSquared < 100) {
				tooNearCandidates.push_back(std::pair<int, int>(i, j));
			}
		}
	}
	std::vector<bool> removalMask(possibleMarkers.size(), false);
	for (size_t i = 0; i < tooNearCandidates.size(); i++) {
		float p1 = perimeter(possibleMarkers[tooNearCandidates[i].first].points);
		float p2 = perimeter(possibleMarkers[tooNearCandidates[i].second].points);
		size_t removalIndex = 0;
		if (p1 < p2)
			removalIndex = tooNearCandidates[i].first;
		else
			removalIndex = tooNearCandidates[i].second;
		removalMask[removalIndex] = true;
	}
	detectedMarkers.clear();
	for (size_t i = 0; i < possibleMarkers.size(); i++) {
		if (!removalMask[i]) {
			detectedMarkers.push_back(possibleMarkers[i]);
		}
	}
};

void MarkerDetector::detectMarker(const cv::Mat& grayscale, std::vector<Marker>& detectedMarkers) {
	cv::Mat canonicalMarker;
	std::vector<cv::Point2f> m_markerCorners2d;
	std::vector<Marker> goodMarkers = {};
	m_markerCorners2d.push_back(cv::Point2f(0, 0));
	m_markerCorners2d.push_back(cv::Point2f(0, markerSize.height));
	m_markerCorners2d.push_back(cv::Point2f(markerSize.width, markerSize.height));
	m_markerCorners2d.push_back(cv::Point2f(markerSize.width, 0));
	int cellSize = grayscale.rows / 7;
	for (size_t i = 0; i < detectedMarkers.size(); i++){
		Marker& marker = detectedMarkers[i];
		cv::Mat M = cv::getPerspectiveTransform(marker.points, m_markerCorners2d);
		cv::warpPerspective(grayscale, canonicalMarker, M, markerSize);
		cv::threshold(canonicalMarker, canonicalMarker, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		cv::Mat bitMatrix = cv::Mat::zeros(5, 5, CV_8UC1);
		for (int y = 0; y < 5; y++) {
			for (int x = 0; x < 5; x++) {
				int cellX = (x + 1) * cellSize;
				int cellY = (y + 1) * cellSize;
				cv::Mat cell = grayscale(cv::Rect(cellX, cellY, cellSize, cellSize));
				
				int nz = cv::countNonZero(cell);
				if (nz > (cellSize * cellSize) / 2)
					bitMatrix.at<uchar>(y, x) = 1;
			}
		}
		std::pair<int, int> minDist = hammDistMarker(bitMatrix);
		if (minDist.first == 0){//check just in case
			std::rotate(marker.points.begin(), marker.points.begin() + 4 - minDist.second, marker.points.end());
			goodMarkers.push_back(marker);
		}
	}
	std::vector<cv::Point2f> preciseCorners(4 * goodMarkers.size());

	for (size_t i = 0; i < goodMarkers.size(); i++) {
		Marker& marker = goodMarkers[i];
		for (int c = 0; c < 4; c++) {
			preciseCorners[i * 4 + c] = marker.points[c];
		}

	}
	cv::cornerSubPix(grayscale, preciseCorners, cvSize(5, 5), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER, 30, 0.1));

	for (size_t i = 0; i < goodMarkers.size(); i++) {
		Marker& marker = goodMarkers[i];
		for (int c = 0; c < 4; c++) {
			marker.points[c] = preciseCorners[i * 4 + c];
		}
	}
}

bool MarkerDetector::findMarkers(const BGRAVideoFrame& frame, std::vector<Marker>& detectedMarkers) {
	cv::Mat bgra = frame.BGRA;

	prepareImage(bgra, m_grayscale);
	performThreshold(m_grayscale, m_threshold);
	findCountours(m_threshold, p_contours, m_grayscale.cols / 5);
	findMarkerCandidates(p_contours, detectedMarkers);
	detectMarkers(m_grayscale, detectedMarkers);
	if (detectedMarkers.size())
		return true;
	else
		return false;
}

void MarkerDetector::etimatePosition(std::vector<Marker>& detectedMarkers) {

}

void processFrame(const BGRAVideoFrame& frame) {

}