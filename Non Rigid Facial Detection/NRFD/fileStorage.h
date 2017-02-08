#ifndef OPEN_CV
#define OPEN_CV

#include <opencv2\opencv.hpp>

#endif

template <class T>
T load_ft(const char* fname) {//get the data fromfile
	T d;//default
	T x; cv::FileStorage f(fname, cv::FileStorage::READ); x.read(f); f.release(); return x;
}

template<class T>
void save_ft(const char* fname, T& x) {//save to file
	cv::FileStorage f(fname, cv::FileStorage::WRITE);
	x.write(f); f.release();
}