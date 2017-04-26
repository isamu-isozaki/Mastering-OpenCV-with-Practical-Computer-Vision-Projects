template <class T>
T load_ft(const char* fname) {//get the data fromfile
	std::cout << "reading from " + (std::string)fname << std::endl;
	T d;//default
	T x; cv::FileStorage f(fname, cv::FileStorage::READ); read(f, x, d); f.release(); return x;
}

template<class T>
void save_ft(const char* fname, T& x) {//save to file
	std::cout << "writing to " + (std::string)fname << std::endl;
	cv::FileStorage f(fname, cv::FileStorage::WRITE);
	write(f, x); f.release();
}