#include "patch_model.h"

class patch_models
{
public:
	patch_models();
	cv::Mat reference;
	std::vector<patch_model> patches;

	void train(ft_data &data,//annotated image and shape data
		const std::vector < cv::Point2f> &ref,//reference shape(what?)
		const cv::Size psize,//desired patch size
		const cv::Size ssize,//training search window size
		const bool mirror = true,//for getPoints and getImage
		const float var = 10.0,//variance of annotation error
		const float lambda = 1e-6,//regularization weight
		const float mu_init = 1e-3,//initial step weight
		const int nsamples = 1000,//number of samples
		const float visi = true);//visualize training procedure?


	std::vector<cv::Point2f> calc_peaks(//location of peak sharpness/feature in image
		const cv::Mat &im,//image to detect feature in
		const std::vector < cv::Point2f> &points,//current estimate of shape
		const cv::Size ssize = cv::Size(44, 44));//search window size

	cv::Mat obtain_training(const cv::Mat& img, cv::Mat &pts, const cv::Size &wsize, const int& pt_num);
	cv::Mat ideal_response;
	void write(cv::FileStorage &fs) const;
	void read(const cv::FileStorage &fs);
private:
	cv::Mat calc_simil(const cv::Mat &pts);//calculate mean deducted
	cv::Mat inv_simil(const cv::Mat &simil);//invert simil so as the transformation is rather points to reference rather than reference to points
	std::vector<cv::Point2f> apply_simil(const cv::Mat &insimil, const std::vector<cv::Point2f> &pts);//make the points appear in the transformation space reference shape is in
};

void write(cv::FileStorage &fs, const patch_models &x);

void read(const cv::FileStorage &fs, patch_models &x, const patch_models &default = patch_models());

