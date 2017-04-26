#include "Haar_cascades.h"

class face_detector//face detector for initialization
{
public:
	face_detector();
	std::string detector_fname;//file containing cascade classifier
	cv::Vec3f detector_offset;//contains 3 elements, (x, y) offset from the center of the face from the center of the detection's bounding box, and scaling factor to best fit the face in the image(learned in face_detector::train
	cv::Mat reference;//canonical model with no distortion
	cv::CascadeClassifier detector;//face detector

	std::vector<cv::Point2f> detect(//points describing detected face in image
		const cv::Mat &im,//image containing face
		const float scaleFactor = 1.1,//scale increment
		const int minNeighbors = 2,//minimum neighbor size
		const cv::Size minSize = cv::Size(30, 30));//minimum window size

	void train(
		ft_data &data,//training data
		const std::string fname,//cascade detector
		const cv::Mat &ref,//reference shape
		const bool mirror = true,//mirror data?
		const bool visi = true,//visualize data?
		const float frac = 0.8,//fraction of points in detection
		const float scaleFactor  = 1.1,//scale increment
		const int minNeighbours = 2,
		const cv::Size &minSize = cv::Size(30, 30));

	void write(cv::FileStorage& fs) const;

	void read(const cv::FileStorage& fs);
private:
	bool enough_bounding_points(const std::vector<cv::Point2f> &pts, const cv::Rect &face, const float &frac);
	float calc_scale(const cv::Mat &pt);
};

void write(cv::FileStorage& fs, const face_detector& detector);

void read(const cv::FileStorage& fs, face_detector& detector, const face_detector& d);

