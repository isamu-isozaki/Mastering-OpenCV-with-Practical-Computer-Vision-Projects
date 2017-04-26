#include "patch_models.h"

patch_models::patch_models(){}

void patch_models::train(ft_data &data,//annotated image and shape data
	const std::vector < cv::Point2f> &ref,//reference shape
	const cv::Size psize,//desired patch size
	const cv::Size ssize,//training search window size
	const bool mirror,//for getPoints and getImage
	const float var,//variance of annotation error
	const float lambda,//regularization weight
	const float mu_init,//initial step weight
	const int nsamples,//number of samples
	const float visi)//visualize training procedure?
{
	int n = ref.size();
	cv::Size wsize = psize + ssize;
	std::cout << "obtained training patch size" << std::endl;
	reference = pts_to_mat(ref);

	int dx = wsize.width - psize.width;//deduct because each of the responses when put into a map, as the training and model starts at a point where the bottom is psize.width, psize.height, will result in
	//the size where those are deducting from the training image size
	int dy = wsize.height - psize.height;
	//dx, dy is the ssize, search size which is the size of the response map
	cv::Mat F(dx, dy, CV_32F);//ideal response map
	for (int y = 0; y < dy; y++) {
		float vy = (dy - 1) / 2 - y;//center - spread, if odd no change after dy-1 if even the center will change by one(will be smaller)
		for (int x = 0; x < dx; x++) {
			float vx = (dx - 1) / 2 - x;
			F.fl(y, x) = std::exp(-0.5*(std::pow(vx, 2) + std::pow(vy, 2)) / var);//formula for 2d gaussian
		}
	}
	cv::normalize(F, F, 0, 1, cv::NORM_MINMAX);//normalize to [0:1] range
	std::cout << "obtained ideal response map" << std::endl;
	for (int i = 0; i < n; i++) {//for each image point
		std::cout << "obtaining data for training patch number: " << i << std::endl;
		std::vector<cv::Mat> patch_train_img;
		for (int j = 0; j < data.points.size(); j++) {//for each annotation
			cv::Mat img = data.get_image(j, 0);
			std::vector<cv::Point2f> annotation_pts = data.get_points(j, false);
			cv::Mat pts = pts_to_mat(annotation_pts);

			cv::Mat I = obtain_training(img, pts, wsize, i);
			patch_train_img.push_back(I);//for point j
		}
		if (mirror)
			for (int j = 0; j < data.points.size(); j++) {
				cv::Mat img = data.get_image(j, 1);
				std::vector<cv::Point2f> annotation_pts = data.get_points(j, true);
				cv::Mat pts = pts_to_mat(annotation_pts);

				cv::Mat I = obtain_training(img, pts, wsize, i);

				patch_train_img.push_back(I);//for point j
			}
		std::cout << "training patch number: " << i << std::endl;
		patch_model patch;
		
		patch.train(F, patch_train_img, psize, var, lambda, mu_init, nsamples, visi);

		patches.push_back(patch);
	}
}

cv::Mat patch_models::calc_simil(const cv::Mat& pts) {
	int n = pts.rows / 2;
	cv::Mat p = pts.clone();
 	//get centered data at 0, 0
	cv::Point2f center_of_mass = find_center_of_mass(pts);
	for (int i = 0; i < n; i++) {
		p.fl(2 * i) -= center_of_mass.x;
		p.fl(2 * i + 1) -= center_of_mass.y;
	}
	cv::Mat S = rot_scale_align(reference, p);
	return cv::Mat_<float>(2, 3) << S.fl(0, 0), S.fl(0, 1), center_of_mass.x, S.fl(1, 0), S.fl(1, 1), center_of_mass.y;
}

cv::Mat patch_models::inv_simil(const cv::Mat &S)
{
	cv::Mat Si(2, 3, CV_32F);
	float d = S.fl(0, 0)*S.fl(1, 1) - S.fl(1, 0)*S.fl(0, 1);
	Si.fl(0, 0) = S.fl(1, 1) / d; Si.fl(0, 1) = -S.fl(0, 1) / d;
	Si.fl(1, 1) = S.fl(0, 0) / d; Si.fl(1, 0) = -S.fl(1, 0) / d;
	cv::Mat Ri = Si(cv::Rect(0, 0, 2, 2));
	cv::Mat t = -Ri*S.col(2), St = Si.col(2); t.copyTo(St); return Si;
}

std::vector<cv::Point2f> patch_models::apply_simil(const cv::Mat& invsimil, const std::vector<cv::Point2f> &pts) {
	std::vector<cv::Point2f> points(pts.size());
	for (int i = 0; i < pts.size(); i++) {
		points[i].x = pts[i].x * invsimil.fl(0, 0) + pts[i].y * invsimil.fl(0, 1) + invsimil.fl(0, 2);
		points[i].y = pts[i].x * invsimil.fl(1, 0) + pts[i].y * invsimil.fl(1, 1) + invsimil.fl(1, 2);
	}
	return points;
}

std::vector<cv::Point2f> patch_models::calc_peaks(//location of peak sharpness/feature in image
	const cv::Mat &im,//image to detect feature in
	const std::vector < cv::Point2f> &points,//current estimate of shape
	const cv::Size ssize)//search window size
{
	int n = points.size(); assert(n == int(patches.size()));
	cv::Mat pt = pts_to_mat(points);
	cv::Mat S = this->calc_simil(pt);//reference -> pt matrix generated
	cv::Mat Si = this->inv_simil(S);//pt -> reference matrix generated

	std::vector<cv::Point2f> pts = this->apply_simil(Si, points);//points are transformed to be in the reference space
	for (int i = 0; i < n; i++) {
		cv::Size wsize = ssize + patches[i].patch_size();//get patch image size
		cv::Mat A(2, 3, CV_32F), I;
		A.fl(0, 0) = S.fl(0, 0); A.fl(0, 1) = S.fl(0, 1); A.fl(0, 2) = pt.fl(2 * i) - (A.fl(0, 0) * (wsize.width - 1) / 2 + A.fl(0, 1) * (wsize.height - 1) / 2);//trainsformation that will be applied to im
		A.fl(1, 0) = S.fl(1, 0); A.fl(1, 1) = S.fl(1, 1); A.fl(1, 2) = pt.fl(2 * i + 1) - (A.fl(1, 0) * (wsize.width - 1) / 2 + A.fl(1, 1) * (wsize.height - 1) / 2);//so as to have pt at the center of it at
		//estimated feature point
		cv::warpAffine(im, I, A, wsize, cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);//get inverse affine transform, basically transform reference to pt or im to pt's space,in doing so obtain patch by
		//taking wsize from it as the center will be at the feature location
		cv::Mat R = patches[i].calc_response(I, false);//get high value at feature point
		double maxVal;
		cv::Point maxLoc; cv::minMaxLoc(R, 0, &maxVal, 0, &maxLoc);//get the maximum value's location of response

		//std::cout << maxVal << std::endl;
		//cv::namedWindow("Win");
		//cv::imshow("Win", R);
		//cv::waitKey(0);
		
		//adjust position of feature point within the region of the feature point estimated
		if (maxVal > 0.4)
			pts[i] = cv::Point2f(pts[i].x + maxLoc.x - 0.5 * ssize.width, pts[i].y + maxLoc.y - 0.5 * ssize.height);
		//As the ssize is the size of the response map(wsize - psize) we will deduct half of it from max loc as pts[i] is situated at the center of the image
	} return this->apply_simil(S, pts);//return to the original space(im's space) as at the moment S got involved, we are situated in reference space
}

void write(cv::FileStorage &fs, const patch_models &x) {
	x.write(fs);
}

void patch_models::write(cv::FileStorage &fs) const{
	assert(fs.isOpened());
	fs << "Reference" << reference << "Patch models" << "[";
	for (patch_model model : patches) {
		fs << model.P;
	}
	fs << "]";
}

void read(const cv::FileStorage &fs, patch_models &x, const patch_models& default) {
	if (fs["Reference"].empty()) x = default;
	if (fs["Patch models"].empty()) x = default;
	std::cout << "adding data to patch_models" << std::endl;
	x.read(fs);
}

void patch_models::read(const cv::FileStorage &fs) {
	fs["Reference"] >> reference;

	patches.clear();
	cv::FileNode fsPatchModels = fs["Patch models"];
	cv::FileNodeIterator patchIterator = fsPatchModels.begin(), patchEnd = fsPatchModels.end();
	for (; patchIterator != patchEnd; patchIterator++) {
		patch_model model;
		model.read((*patchIterator));
		patches.push_back(model);
	}
}

cv::Mat patch_models::obtain_training(const cv::Mat &img, cv::Mat &pts, const cv::Size& wsize, const int& pt_num) {
	cv::Mat S = calc_simil(pts);
	cv::Mat A(2, 3, CV_32F);
	A.fl(0, 0) = S.fl(0, 0); A.fl(0, 1) = S.fl(0, 1); A.fl(0, 2) = pts.fl(2 * pt_num) - (A.fl(0, 0) * (wsize.width - 1) / 2 + A.fl(0, 1) * (wsize.height - 1) / 2);//As if we move to the hithero location, the point
	A.fl(1, 0) = S.fl(1, 0); A.fl(1, 1) = S.fl(1, 1); A.fl(1, 2) = pts.fl(2 * pt_num + 1) - (A.fl(1, 0) * (wsize.width - 1) / 2 + A.fl(1, 1) * (wsize.height - 1) / 2);//will be at the center of the image
	cv::Mat I;
	cv::warpAffine(img, I, A, wsize, cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
	return I;
}