#include "shape_model.h"

shape_model::shape_model(){}

cv::Mat shape_model::procrustes(const cv::Mat &X, const int itol, const float ftol){
	cv::Mat Co, P = X.clone();//copy
	for (int i = 0; i < N; i++) {
		cv::Mat& p = P.col(i);//i'th shape
		cv::Point2f center_of_mass = find_center_of_mass(p);
		for (int j = 0; j < n; j++) {
			p.fl(2 * j) -= center_of_mass.x; p.fl(2 * j + 1) -= center_of_mass.y;//make the model have the center at (0, 0)
		}
	}
	std::cout << "Acquired mean centered data" << std::endl;
	for (int iter = 0; iter < itol; iter++) {//all of P's shapes are now centerd
		cv::Mat canonical = P * cv::Mat::ones(N, 1, CV_32F) / N;//each shape is multiplied by 1/N and the vectors of all of them are added together-> mean x value and y value for each point is computed in canonical has the scale X x 1
		cv::normalize(canonical, canonical);//make the maximum one and the minimum 0 in for the points -> prevents shapes from going to 0

		if (iter > 0) { if (cv::norm(canonical, Co) < ftol) break; }//converged? If there is a lack of change break
		Co = canonical.clone();//A distinct copy
		for (int i = 0; i < N; i++) {
			cv::Mat R = rot_scale_align(P.col(i), canonical);//function that finds optimal rotation and scaling to align -> returns a 2x2 matrix
			for (int j = 0; j < n; j++) {
				float x = P.fl(2 * j, i), y = P.fl(2 * j + 1, i);//each point in the model is rotated and scaled so as all of the points approach the optimal model, with only rotation and scaling
				
				P.fl(2 * j, i) = R.fl(0, 0) * x + R.fl(0, 1) * y;//rotate and scale
				P.fl(2 * j + 1, i) = R.fl(1, 0) * x + R.fl(1, 1) * y;//rotate and scale
				
			}
		}//On every iteration P and hence, C is altered to lose more of the global transformation
		std::cout << cv::norm(canonical, Co) << std::endl;
	}
	std::cout << "Removed global(rigid) transformation" << std::endl;
	return P;//return procrusted(global transformation removed) aligned shapes
};

cv::Mat rot_scale_align(const cv::Mat &src, const cv::Mat &dst){
	float a = 0, b = 0, d = 0;
	int n = src.rows / 2;
	for (int i = 0; i < n; i++) {
		d += src.fl(2 * i)*src.fl(2 * i) + src.fl(2 * i + 1)* src.fl(2 * i + 1);//x^2 + y^2
		a += src.fl(2 * i) * dst.fl(2 * i) + src.fl(2 * i + 1) * dst.fl(2 * i + 1);//x*x' + y*y'
		b += src.fl(2 * i)* dst.fl(2 * i + 1) - src.fl(2 * i + 1) * dst.fl(2 * i);//x*y' - y*x'
		//where x', y' is the optimal model's coordinates
	}
	a /= d;	b /= d;
	return (cv::Mat_<float>(2, 2) << a, -b, b, a);//a is kcos and b is ksin coming from minimizing the equation: sigma ||[a -b; b a][xi; yi] -[cx; cy]||^2
}

void shape_model::train(const std::vector<std::vector<cv::Point2f>> &p, const std::vector<std::vector<int> > &conn, const float frac, const int kmax) {
	cv::Mat Shapes = pts_to_mat(p);
	N = Shapes.cols;
	n = Shapes.rows / 2;
	//align shape
	cv::Mat Y = this->procrustes(Shapes);
	//compute rigid transformation
	cv::Mat R = this->calc_rigid_basis(Y);//making subspace of global transformation
	
	cv::Mat dY = Y - R * R.t() * Y;//the error in Y's projection to R, orthogonal to R

	
	cv::SVD svd(dY * dY.t());//column space
	int m = std::min(std::min(kmax, N - 1), n - 1);

	float vsum = 0; 
	for (int i = 0; i < m; i++)
		vsum += svd.w.fl(i);//the ith singular value
	float v = 0; int k = 0;
	for (k = 0; k < m; k++){
		v += svd.w.fl(k);
		if (v / vsum >= frac) {//if it describes the true data fairly well
			k++;//add to k because we
			break;
		}
	}
	if (k > m) k = m;
	
	cv::Mat D = svd.u(cv::Rect(0, 0, k, 2 * n));//eigenvector of A*A^T, i.e. the column space's basis
	std::cout << "Computed PCA and retained the important parts" << std::endl;
	//here is where we extract a rectangle from the matrix, as (x, y, width, height)

	//store rigid and non rigid transformation
	V = cv::Mat(2 * n, 4 + k, CV_32F);

	cv::Mat Vr = V(cv::Rect(0, 0, 4, 2 * n)); R.copyTo(Vr);
	cv::Mat Vd = V(cv::Rect(4, 0, k, 2 * n)); D.copyTo(Vd);
	std::cout << "Created matrix V" << std::endl;
	//store connectivity
	//find height
	int height = 0;
	for (std::vector<int> connection : conn)
		height = std::max(height, (int)connection.size());

	//create matrix
	C = cv::Mat(height, conn.size(), CV_32F);

	for (int i = 0; i < conn.size(); i++) {
		for (int j = 0; j < conn[i].size(); j++) {
			C.fl(j, i) = (float)conn[i][j];//save it in row format
		}
		if (conn[i].size() < height) {
			int size = conn[i].size();
			for (; size < height; size++) {
				C.fl(size, i) = -1.;//convert the excess ones to -1s
			}
		}
	}
	std::cout << "Created matrix C" << std::endl;

	cv::Mat Q = V.t() * Y;//projection of Y(raw shape data) on to V
	for (int i = 0; i < N; i++) {
		float v = Q.fl(0, i); cv::Mat q = Q.col(i); q /= v;//devide the column by the value that appears to correspond to the global transformation(the scale) -> so as to not dominate the estimate
	}
	e.create(cv::Size(1, 4 + k), CV_32F); cv::multiply(Q, Q, Q);//Square every element in Q
	for (int i = 0; i < 4 + k; i++) {
		if (i < 4)
			e.fl(i) = -1;//no clamping for rigid coefficients
		else
			e.fl(i) = Q.row(i).dot(cv::Mat::ones(cv::Size(N, 1), CV_32F)) / (N - 1);//sum of variance in projection to the nonrigid transformation
	}
	std::cout << "Created matrix e" << std::endl;
}

cv::Mat pts_to_mat(const std::vector<std::vector<cv::Point2f>> &p) {
	cv::Mat conv_Mat = cv::Mat(cv::Size(p.size(), p[0].size() * 2), CV_32F);//width, height
	for (int i = 0; i < p.size(); i++){
		for (int j = 0; j < p[0].size(); j++) {
			conv_Mat.fl(2 * j, i) = p[i][j].x;
			conv_Mat.fl(2 * j + 1, i) = p[i][j].y;
		}
	}
	std::cout << "Converted input data to Mat" << std::endl;
	return conv_Mat;
}

cv::Mat pts_to_mat(const std::vector<cv::Point2f> &p) {
	cv::Mat conv_Mat = cv::Mat(cv::Size(1, p.size() * 2), CV_32F);
	for (int i = 0; i < p.size(); i++){
		conv_Mat.fl(2 * i) = p[i].x;
		conv_Mat.fl(2 * i + 1) = p[i].y;
	}
	return conv_Mat;
}

cv::Mat shape_model::calc_rigid_basis(const cv::Mat& Y) {
	cv::Mat mean_shape = Y * cv::Mat::ones(cv::Size(1, N), CV_32F)/N;

	cv::Mat R = cv::Mat(cv::Size(4, 2 * n), CV_32F);
	for (int j = 0; j < n; j++) {
		R.fl(2 * j, 0)	   = mean_shape.fl(2 * j);	   R.fl(2 * j, 1)     = -mean_shape.fl(2 * j + 1); R.fl(2 * j, 2) = 1.;     R.fl(2 * j, 3) = 0.;
		R.fl(2 * j + 1, 0) = mean_shape.fl(2 * j + 1); R.fl(2 * j + 1, 1) = mean_shape.fl(2 * j);  R.fl(2 * j + 1, 2) = 0.; R.fl(2 * j + 1, 3) = 1.;
		//constructing [x, -y, 1, 0; y, x, 0, 1]
	}

	for (int i = 0; i < 4; i++) {//thank you source code
		cv::Mat current_col = R.col(i);
		for (int j = 0; j < i; j++) {
			cv::Mat deducting_col = R.col(j); 
			current_col -= deducting_col * (deducting_col.t() * current_col);//Gram Sccmit, is deducting_col orthonormal? -> Yes, it is due to this proccess, 
		}
		cv::normalize(current_col, current_col);//to make it orthonormal
	}
	std::cout << "Computed rigid transformation" << std::endl;
	return R;
}

void shape_model::calc_params(const std::vector<cv::Point2f> &s_points, const cv::Mat &weight, const float& c_factor) {
	cv::Mat s = cv::Mat(cv::Size(1, 2 * n), CV_32F);
	s = pts_to_mat(s_points);

	p = V.t()*s;//As V.t() * V = I as both R and D are orthonormal -> from v.t()*(s-v*p) = 0
	this->clamp(c_factor);
}

void shape_model::clamp(const float &c) {
	double scale = p.fl(0);//First param is the scale
	for (int i = 0; i < e.rows; i++) {
		if (e.fl(i) < 0) continue;//ignore rigid component -> can also do if(i < 4)
		float v = c*sqrt(e.fl(i));//c * standard deviation
		if (std::fabs(p.fl(i) / scale) > v) {//preserve sign of coordinate,can compare to absolute as center is 0
			if (p.fl(i) > 0) p.fl(i) = v * scale;//positive threshold
			else p.fl(i) = -v *scale;//negative threshold
			//all of them had the scale removed so as not to have them influence the variance, hence it is timed back on it
			//scale = the amount one standard deviation corresponds to
		}
	}
}

std::vector<cv::Point2f> shape_model::calc_shape() {
	cv::Mat s = V * p;
 	std::vector<cv::Point2f> shape;

	for (int i = 0; i < n; i++) shape.push_back(cv::Point2f(s.fl(2 * i), s.fl(2 * i + 1)));
	return shape;
}

float shape_model::calc_scale(const cv::Mat &col, const int& pixels) {
	float xmin = col.fl(0), xmax = col.fl(0);
	for (int i = 0; i < n; i++) {
		xmin = std::min(xmin, col.fl(2 * i));//only the x coordinates
		xmax = std::max(xmax, col.fl(2 * i));
	} 
	return float(pixels) / (xmax - xmin);
}

void shape_model::draw_shape(cv::Mat &output, const std::vector < cv::Point2f> &q) {
	for (int i = 0; i < q.size(); i++) {
		cv::Point2f point = q[i];
		cv::putText(output, std::to_string(i), point, cv::FONT_HERSHEY_PLAIN, 0.5, cv::Scalar(0, 250, 0), 0.8);
		cv::Mat connection = C.col(i).clone();
		for (int k = 0; k < C.rows; k++){
			if (connection.fl(k) != -1)
				cv::line(output, point, q[connection.fl(k)], cv::Scalar(250, 250, 0), 0.8);
		}
	}
}

void shape_model::write(cv::FileStorage& fs) const{
	fs << "V" << V << "e" << e << "C" << C << "N" << N << "n" << n;
}

void write(cv::FileStorage &fs, const shape_model x) {
	x.write(fs);
}

void shape_model::read(const cv::FileStorage& fs) {
	fs["V"] >> V;
	fs["e"] >> e;
	fs["C"] >> C;
	fs["N"] >> N;
	fs["n"] >> n;
}

void read(const cv::FileStorage& fs, shape_model& x, const shape_model& default) {
	if (fs["V"].empty()) x = default;
	if (fs["e"].empty()) x = default;
	if (fs["C"].empty()) x = default;
	if (fs["N"].empty()) x = default;
	if (fs["n"].empty()) x = default;
	x.read(fs);
}

cv::Point2f find_center_of_mass(const cv::Mat& pts) {
	float mx = 0, my = 0;//compute the center of mass for both x and y
	int n = pts.rows / 2;
	for (int j = 0; j < n; j++) {
		mx += pts.fl(2 * j);//the x coordinates will be at the even indices
		my += pts.fl(2 * j + 1);//the y coordinates will be at the odd indices
	}
	mx /= n;//find the x coordinate that is at the middle of the shape
	my /= n;//find the y coordinate that is at the middle of the shape
	return cv::Point2f(mx, my);
}