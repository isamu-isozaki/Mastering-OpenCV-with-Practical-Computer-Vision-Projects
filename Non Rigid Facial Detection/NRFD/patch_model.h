#include "shape_model.h"

class patch_model
{
public:
	patch_model();
	cv::Mat P;//normalized patch

	cv::Mat calc_response(//response map
		const cv::Mat &im,//image patch of search region
		const bool sum2one = false);//normalize to sum to one

	void train(
		const cv::Mat &F,//response map
		const std::vector<cv::Mat> &images,//training img patches(from a single annotation?)
		const cv::Size &psize,//patch size
		const float& var = 10.0,//ideal response variance
		const float& lambda = 1e-6,//regularization weight
		const float mu_init = 1e-3,//initial step size
		const int nsamples = 1000,//number of samples
		const bool visi = true);//visualize process?
	cv::Size patch_size();
	void write(cv::FileStorage &fs) const;
	void read(const cv::FileNode &fs);//as we will never store a feature in it individually
private:
	cv::Mat convert_image(const cv::Mat& img);
	cv::Size psize;
};

void write(cv::FileStorage &fs, const patch_model &x);

void read(const cv::FileNode &fs, patch_model &x, const patch_model &default = patch_model());