#include "patch_model.h"

patch_model::patch_model()
{
}

cv::Mat patch_model::convert_image(const cv::Mat &img) {
	cv::Mat output;
	if(img.type() != CV_32F) img.convertTo(output, CV_32F, 1.0/255.0);// as it turned out to be always white
	else output = img;
	if (img.channels() == 3) {
		cv::cvtColor(output, output, CV_BGR2GRAY);
	}
	
	output += 1;//so as not to have any unfortunate occurance such as errors for log 0
	cv::log(output, output);//do a log as this makes the image more robust against difference in contrast and change in illumination conditions
	
	return output;
}

void patch_model::train(
	const cv::Mat &F,//ideal response map
	const std::vector<cv::Mat> &images,//training img patches
	const cv::Size &psize,//patch size
	const float& var,//ideal response variance
	const float& lambda,//regularization weight
	const float mu_init,//initial step size
	const int nsamples,//number of samples
	const bool visi)//visualize process?
{
	
	this->psize = psize;
	int N = images.size(), n = psize.width * psize.height;
	cv::Size wsize = images[0].size();

	int dx = wsize.width - psize.width;//deduct because each of the responses when put into a map, as the training and model starts at a point where the bottom is psize.width, psize.height, will result in
	//the size where those are deducting from the training image size
	int dy = wsize.height - psize.height;
	//dx, dy is the ssize, search size which is the size of the response map

	cv::Mat I(wsize.height, wsize.width, CV_32F);//training map
	cv::Mat dP(psize.height, psize.width, CV_32F);//the gradient
	cv::Mat O = cv::Mat::ones(psize.height, psize.width, CV_32F)/n;
	P = cv::Mat::zeros(psize.height, psize.width, CV_32F);

	std::cout << "conducting stochatic gradient descent" << std::endl;
	//stochatic gradient descent
	cv::RNG rn(cv::getTickCount());//random number generator
	double mu = mu_init, step = std::pow(1e-8 / mu_init, 1.0 / nsamples);//so as after nsamples iterations the stepsize will be 13-8, a value close to 0
	for (int sample = 0; sample < nsamples; sample++) {
		int i = rn.uniform(0, N); //random sample size index
		I = this->convert_image(images[i]); dP = 0.0;
		for (int y = 0; y < dy; y++) {
			for (int x = 0; x < dx; x++) {
				cv::Mat Wi = I(cv::Rect(x, y, psize.width, psize.height)).clone();
				Wi -= Wi.dot(O);//center it, i.e. deduct mean
				cv::normalize(Wi, Wi);//normalize
				dP += (F.fl(y, x) - P.dot(Wi) )* Wi;//the gradient
			}
		}
		P += mu*(dP - lambda * P);//lambda*P is there for a safe guard this from growing too large so as to generalize to unseen data -> yet why not deduct dP, why keep it?
		mu *= step;//make step smaller on every iteration
	}
	if (visi){
 		std::cout << "P\n" << P << std::endl;
	}
	std::cout << "obtained optimal patch" << std::endl;
	return;//all in all computes single patch in each annotation
}

cv::Mat patch_model::calc_response(const cv::Mat &Img, const bool sum2one){
	cv::Mat I = this->convert_image(Img);

	

	cv::Mat response;
	cv::matchTemplate(I, P, response, cv::TM_CCORR_NORMED);//output response via normalized cross correlation(dot product of I and P, yet normalized so as not to have any adnormal signals get in the way of cross correlation) where the response is generated in response
	//at the high values in response map -> better match
	if (sum2one) {
		cv::normalize(response, response, 0, 1, cv::NORM_MINMAX);
		response /= cv::sum(response)[0];//only the gray scale
	}
	return response;
}

cv::Size patch_model::patch_size() {
	return psize;
}
/*
void write(cv::FileStorage &fs, const patch_model &x) {
	x.write(fs);
}

void patch_model::write(cv::FileStorage& fs) const {
	assert(fs.isOpened());
	fs << P;
}
*/
void read(const cv::FileNode &fs, patch_model &x, const patch_model &default) {
	if (fs["Patch"].empty()) x = default;
	x.read(fs);
}

void patch_model::read(const cv::FileNode &fs) {
	fs >> P;
	psize = P.size();
}