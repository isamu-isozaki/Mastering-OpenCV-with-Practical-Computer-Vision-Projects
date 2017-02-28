#include "shape_model.h"

#define fl at<float>

shape_model::shape_model(){}

cv::Mat shape_model::procrustes(const cv::Mat &X, const int itol = 30, const float ftol = 5.){
	cv::Mat Co, P = X.clone();//copy
	for (int i = 0; i < N; i++) {
		cv::Mat& p = P.col(i);//i'th shape
		float mx = 0, my = 0;//compute the center of mass for both x and y
		for (int j = 0; j < n; j++) {
			mx += p.fl(2 * j);//the x coordinates will be at the even indices
			my += p.fl(2 * j + 1);//the y coordinates will be at the odd indices
		}
		mx /= n;//find the x coordinate that is at the middle of the shape
		my /= n;//find the y coordinate that is at the middle of the shape
		for (int j = 0; j < n; j++) {
			p.fl(2 * j) -= mx; p.fl(2 * j + 1) -= my;//make the model have the center at (0, 0)
		}
	}
	for (int iter = 0; iter < itol; iter++) {//all of P's shapes are now centerd
		cv::Mat C = P * cv::Mat::ones(N, 1, CV_32F) / N;//each shape is multiplied by 1/N and the vectors of all of them are added together-> mean x value and y value for each point is computed in C has the scale X x 1
		cv::normalize(C, C);//make the maximum one and the minimum 0 in for the points -> prevents shapes from going to 0
		
		if (iter > 0) { if (cv::norm(C, Co) < ftol) break; }//converged
		for (int i = 0; i < N; i++) {
			cv::Mat R = this->rot_scale_align(P.col(i), C);//function that finds optimal rotation and scaling to align -> returns a 2x2 matrix
			for (int j = 0; j < n; j++) {
				float x = P.fl(2 * j, i), y = P.fl(2 * j + 1, i);//each point in the model is rotated and scaled so as all of the points approach the optimal model, with only rotation and scaling
				P.fl(2 * j, i) = R.fl(0, 0) * x + R.fl(0, 1) * y;//rotate and scale
				p.fl(2 * j + 1, i) = R.fl(1, 0) * x + R.fl(1, 1) * y;//rotate and scale
			}
		}
	}
	return P;//return procrusted(global transformation removed) aligned shapes
};

cv::Mat shape_model::rot_scale_align(const cv::Mat &src, const cv::Mat &dst){
	float a = 0, b = 0, d = 0;
	for (int i = 0; i < n; i++) {
		d += src.fl(2 * i)*src.fl(2 * i) + src.fl(2 * i + 1)* src.fl(2 * i + 1);//x^2 + y^2
		a += src.fl(2 * i) * dst.fl(2 * i) + src.fl(2 * i + 1) * dst.fl(2 * i + 1);//x*x' + y*y'
		b += src.fl(2 * i)* dst.fl(2 * i + 1) - src.fl(2 * i + 1) * dst.fl(2 * i);//x*y' - y*x'
		//where x', y' is the optimal model's coordinates
	}
	a /= d;	b /= d;
	return (cv::Mat_<float>(2, 2) << a, -b, b, a);//a is kcos and b is ksin coming from minimizing the equation: sigma ||[a -b; b a][xi; yi] -[cx; cy]||^2
}

void shape_model::train(const std::vector<std::vector<cv::Point2f>> &p, const std::vector<std::vector<int> > &conn = std::vector<std::vector<int> >(), const float frac = 0.95, const int kmax = 10) {
	cv::Mat Shapes = this->pts_to_mat(p);
	N = Shapes.cols;
	n = Shapes.rows / 2;


	cv::Mat Y = this->procrustes(Shapes);
	cv::Mat R = this->calc_rigid_basis(Y);//
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
	//here is where we extract a rectangle from the matrix, as (x, y, width, height)

	//store rigid and non rigid transformation
	V = cv::Mat(2 * n, 4 + k, CV_32F);

	cv::Mat Vr = V(cv::Rect(0, 0, 4, 2 * n)); R.copyTo(Vr);
	cv::Mat Vd = V(cv::Rect(4, 0, k, 2 * n)); D.copyTo(Vd);

	//store connectivity
	//find height
	int height = 0;
	for (std::vector<int> connection : conn)
		height = std::max(height, (int)connection.size());

	//create matrix
	C = cv::Mat(cv::Size(conn.size(), height), CV_32F);

	for (int i = 0; i < conn.size(); i++) {
		for (int j = 0; j < conn[i].size(); j++) {
			C.fl(i, j) = (float)conn[i][j];//save it in row format
		}
		if (conn[i].size() < height) {
			int size = conn[i].size();
			for (; size < height; size++) {
				C.fl(i, size) = -1.;//convert the excess ones to -1s
			}
		}
	}

	cv::Mat Q = V.t() * Y;//projection of Y(raw shape data onto V
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

}

cv::Mat shape_model::pts_to_mat(const std::vector<std::vector<cv::Point2f>> &p) {
	cv::Mat conv_Mat = cv::Mat(cv::Size(p.size(), p[0].size() * 2), CV_32F);//width, height
	for (int i = 0; i < p.size(); i++){
		for (int j = 0; j < p[0].size(); j++) {
			conv_Mat.fl(2 * j, i) = p[i][j].x;
			conv_Mat.fl(2 * j + 1, i) = p[i][j + 1].y;
		}
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
	return R;
}

void shape_model::calc_params(const std::vector<cv::Point2f> &s_points, const cv::Mat &weight = cv::Mat(), const float& c_factor = 3.0) {
	cv::Mat s = cv::Mat(cv::Size(1, 2 * n), CV_32F);
	for (int j = 0; j < n; j++) {
		s.fl(2 * j) = s_points[j].x;
		s.fl(2 * j + 1) = s_points[j].y;
	}

	p = V.t()*s;//As V.t() * V = I as both R and D are orthonormal -> from v.t()*(s-v*p) = 0
}

void shape_model::clamp(const float &c = 3.0) {
	double scale = p.fl(0);//As p is the projection on to V -> here is where to x, y the rotation and scaling is applied(where x1 y1 x2 y2...) is multiplide to s's (x1 y1 x2 y2....) -> what about the semi rotation? Why only scale?
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