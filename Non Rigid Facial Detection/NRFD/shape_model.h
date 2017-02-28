#include "ft_data.h"

class shape_model
{
public:
	shape_model();
	cv::Mat p;//parameter vector (kx1) CV_32F, matrix that contains the projection of points on a face onto the space of potential faces
	cv::Mat V;//linear subspace (2nxh) CV_32F matrix containing orthonormal subspace of rigid transformation in 0, 0 to 2 * n, 4, and non rigid transformation in 4, 0 to k + 4, 2 * n
	cv::Mat e;//parameter variance (kx1) CV_32F
	cv::Mat C;//connectivity (cx2) CV_32S

	void calc_params(//projects a set of points onto the space of plausible faces-provides an optional confident weight?
		const std::vector<cv::Point2f> &pts,//points to compute params
		const cv::Mat & weight = cv::Mat(),//weight/point (nx1) CV_32F
		const float& c_factor = 3.0);//clamping factor

	std::vector<cv::Point2f> calc_shape();//shape described by parameters, i.e. decode the parameter vector using the face model

	void train(//learns encoding model from a dataset of face shapes, each of which consists of the same number of points
		const std::vector<std::vector<cv::Point2f> > &p,//N-example shape
		const std::vector<std::vector<int>> &con = std::vector < std::vector<int>>(),//connectivity
		const float frac = 0.95,//fraction of variation to retain
		const int kmax = 10);//maximum number of modes to retain, both can be specialized to data at hand
private:
	int N, n;//num of cols, and half the number of rows respectively

	cv::Mat procrustes(
		const cv::Mat &X,//raw shape data as columns
		const int itol = 30,//maximum number of iterations
		const float ftol = 5.//convergence tolerance
		);
	cv::Mat rot_scale_align(const cv::Mat &src,//vector of source shape
		const cv::Mat &dst);//destination, canonical shape
	cv::Mat calc_rigid_basis(const cv::Mat& Y);//compute rigid subspace from procrusted matrix
	cv::Mat pts_to_mat(const std::vector<std::vector<cv::Point2f>> &p);//convert the training data into a matrix
	void clamp(const float &c = 3.0);
};

