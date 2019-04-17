#ifndef __CD_NEURAL_NET_HPP_
#define __CD_NEURAL_NET_HPP_

#include <opencv2/dnn.hpp>



struct _bbox {
    int classId;
    float confidence;
    float x, y, w, h;
};



class CDNeuralNet{
private:
	cv::dnn::Net _net;

public:
	CDNeuralNet(std::string, std::string);
	void detect(cv::Mat);
	ssize_t get_output_boxes(struct _bbox *buf, size_t len);
};


#endif // __CD_NEURAL_NET_HPP_
