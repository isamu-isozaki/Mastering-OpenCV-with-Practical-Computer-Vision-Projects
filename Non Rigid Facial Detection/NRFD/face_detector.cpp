#include "face_detector.h"


face_detector::face_detector()
{
}

std::vector<cv::Point2f> face_detector::detect(const cv::Mat &im, const float scaleFactor, const int minNeighbours, const cv::Size minSize){
	cv::Mat gray;
	im.convertTo(gray, CV_8UC1);
	if (gray.channels() == 3) cv::cvtColor(gray, gray, CV_BGR2GRAY);
	cv::Mat eqIm;
	cv::equalizeHist(gray, eqIm);//equalize all the histograms i.e. make all the num of pixels associated to a certain tone the same as with any other tone
	std::vector<cv::Rect> faces;//detect largest face in image
	detector.detectMultiScale(eqIm, faces, scaleFactor, minNeighbours, 0 | CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, minSize);
	if (faces.size() < 1) { return std::vector<cv::Point2f>(); }
	cv::Rect R = faces[0]; cv::Vec3f scale = detector_offset * R.width;
	int n = reference.rows / 2; std::vector<cv::Point2f> p(n);
	for (int i = 0; i < n; i++) {
		p[i].x = scale[2] * reference.fl(2 * i)/50 + R.x + 0.5 * R.width + scale[0];
		p[i].y = scale[2] * reference.fl(2 * i + 1)/50 + R.y + 0.5 * R.height + scale[1];//why 1/100
	}
	return p;
};

void face_detector::train(ft_data &data, std::string fname, const cv::Mat &ref, bool mirror, bool visi, float frac, float scaleFactor, int minNeighbours, const cv::Size &minSize) {
	detector.load(fname.c_str()); detector_fname = fname; 
	reference = ref.clone();//convinient choice for this is the normalized average shape
	std::vector<float> xoffset, yoffset, zoffset;
	for (int i = 0; i < data.n_images(); i++) {
		cv::Mat im = data.get_image(i, 0); if (im.empty()) continue;
		std::vector <cv::Point2f> p = data.get_points(i, false); int n = p.size();
		cv::Mat pt = pts_to_mat(p);
		cv::Mat eqIm;
		cv::equalizeHist(im, eqIm);//equalize all the histograms i.e. make all the num of pixels associated to a certain tone the same as with any other tone
		std::vector<cv::Rect> faces;//detect largest face in image
		detector.detectMultiScale(eqIm, faces, scaleFactor, minNeighbours, 0 | CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, minSize);
		if (faces.size() >= 1) {
			if (this->enough_bounding_points(p, faces[0], frac)) {
				cv::Point2f center = find_center_of_mass(pt);
				float w = faces[0].width;
				xoffset.push_back((center.x - (faces[0].x + 0.5 * faces[0].width)) / w);//offset between the center of mass and the actual center
				yoffset.push_back((center.y - (faces[0].y + 0.5 * faces[0].height)) / w);
				zoffset.push_back(this->calc_scale(pt) / w);//scale from reference per unit width
				if (visi) {
					cv::Mat I;
					cv::cvtColor(eqIm, I, CV_GRAY2RGB);
					for (int i = 0; i < n; i++) {
						cv::circle(I, p[i], 1, CV_RGB(0, 255, 0), 2, CV_AA);
					}
					cv::rectangle(I, faces[0].tl(), faces[0].br(), CV_RGB(255, 0, 0), 3);//tl -> top left, br -> bottom right
					cv::imshow("face detector training", I); cv::waitKey(10);
				}
			}
		}
		if (mirror) {
			cv::Mat im = data.get_image(i, 1); if (im.empty()) continue;
			std::vector <cv::Point2f> p = data.get_points(i, true); int n = p.size();
			cv::Mat pt = pts_to_mat(p);
			cv::Mat eqIm;
			cv::equalizeHist(im, eqIm);//equalize all the histograms i.e. make all the num of pixels associated to a certain tone the same as with any other tone
			std::vector<cv::Rect> faces;//detect largest face in image
			detector.detectMultiScale(eqIm, faces, scaleFactor, minNeighbours, 0 | CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, minSize);
			if (faces.size() >= 1) {
				if (this->enough_bounding_points(p, faces[0], frac)) {
					cv::Point2f center = find_center_of_mass(pt);
					float w = faces[0].width;
					xoffset.push_back((center.x - (faces[0].x + 0.5 * faces[0].width)) / w);
					yoffset.push_back((center.y - (faces[0].y + 0.5 * faces[0].height)) / w);
					zoffset.push_back(this->calc_scale(pt) / w);
					if (visi) {
						cv::Mat I;
						cv::cvtColor(eqIm, I, CV_GRAY2RGB);
						for (int i = 0; i < n; i++) {
							cv::circle(I, p[i], 1, CV_RGB(0, 255, 0), 2, CV_AA);
						}
						cv::rectangle(I, faces[0].tl(), faces[0].br(), CV_RGB(255, 0, 0), 3);//tl -> top left, br -> bottom right
						cv::imshow("face detector training", I); cv::waitKey(10);
					}
				}
			}
		}
	}
	cv::Mat X = cv::Mat(xoffset), Xsort, Y = cv::Mat(yoffset), Ysort, Z = cv::Mat(zoffset), Zsort;
	cv::sort(X, Xsort, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
	int nx = Xsort.rows;
	cv::sort(Y, Ysort, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
	int ny = Ysort.rows;
	cv::sort(Z, Zsort, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
	int nz = Zsort.rows;
	detector_offset = cv::Vec3f(Xsort.fl(nx / 2), Ysort.fl(ny / 2), Zsort.fl(nz / 2));//set to the median of offsets
}

bool face_detector::enough_bounding_points(const std::vector<cv::Point2f> &pts, const cv::Rect &face, const float &frac){
	int m = 0, n = pts.size();
	for (int i = 0; i < n; i++) {
		if (face.x <= pts[i].x && pts[i].x <= face.x + face.width && face.y <= pts[i].y && pts[i].y <= face.y + face.width) m++;
	}
	if (float(m) / n >= frac)
		return true;
	else
		return false;
};

float face_detector::calc_scale(const cv::Mat &pt) {
	float ref_xmin = reference.fl(0), ref_xmax = reference.fl(0), pt_xmin = pt.fl(0), pt_xmax = pt.fl(0);
	int n = reference.rows / 2;
	for (int i = 0; i < n; i++) {//as reference and shape ought to have the same number of points
		ref_xmin = std::min(ref_xmin, reference.fl(2 * i));
		ref_xmax = std::max(ref_xmax, reference.fl(2 * i));
		pt_xmin = std::min(pt_xmin, pt.fl(2 * i));
		pt_xmax = std::max(pt_xmax, pt.fl(2 * i));
	}
	return (pt_xmax - pt_xmin) / (ref_xmax - ref_xmin);
}

void face_detector::write(cv::FileStorage& fs) const{
	fs << "Detector offset" << detector_offset << "Detector" << detector_fname << "reference" << reference;
}

void face_detector::read(const cv::FileStorage& fs) {
	fs["Detector offset"] >> detector_offset;
	fs["Detector"] >> detector_fname;
	//fs["reference"] >> reference;
	detector.load(detector_fname.c_str());
}

void write(cv::FileStorage& fs, const face_detector& detector){
	detector.write(fs);
}

void read(const cv::FileStorage& fs, face_detector& detector, const face_detector& d){
	detector.read(fs);
}