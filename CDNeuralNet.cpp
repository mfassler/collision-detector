
#include <mutex>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "CDNeuralNet.hpp"

std::mutex mtx_a;

int bboxIdx = 0;
int nboxes[2];
struct _bbox bboxes[2][25];

std::vector<cv::String> _outNames;

CDNeuralNet::CDNeuralNet(std::string modelPath, std::string configPath) {


	_net = cv::dnn::readNet(modelPath, configPath, "");
	_net.setPreferableBackend(0);
	_net.setPreferableTarget(0);

	_outNames = _net.getUnconnectedOutLayersNames();
}




void CDNeuralNet::detect(cv::Mat frame) {

	// Create a 4D blob from a frame.
	cv::Size _inpSize(512, 288); // size of the NN input layer

	// Input values are 0..255, but NN expects 0.0..1.0:
	float _scale = 1.0/255.0;  // Input values are 0-255, but NN expects 0.0-1.0
	cv::Scalar _mean = 0.0;

	bool _swapRB = true;  // OpenCV is BGR, but NN is RGB
	bool _cropping = false; // TODO:... hrmm, cropping isn't "letterbox"-style cropping...

	cv::Mat blob;
	cv::dnn::blobFromImage(frame, blob, _scale, _inpSize, _mean, _swapRB, _cropping);

	_net.setInput(blob);


	// This never seems to happen, and I don't know what it does:
	if (_net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
		printf(" ### This code is untested... ### \n");
		resize(frame, frame, _inpSize);
		cv::Mat imInfo = (cv::Mat_<float>(1, 3) << _inpSize.height, _inpSize.width, 1.6f);
		_net.setInput(imInfo, "im_info");
	}

	std::vector<cv::Mat> outs;
	_net.forward(outs, _outNames);


	static std::vector<int> outLayers = _net.getUnconnectedOutLayers();
	static std::string outLayerType = _net.getLayer(outLayers[0])->type;

	if (outLayerType == "DetectionOutput") {

		printf("This NN modeltype is not supported by this application.\n");
		printf("it is supported by OpenCV, tho.\n");

	} else if (outLayerType == "Region") {

		mtx_a.lock();
		bboxIdx++;
		if (bboxIdx > 1) {
			bboxIdx = 0;
		}
		int count = 0;

		for (size_t i=0; i<outs.size(); ++i) {
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
				cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				cv::Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > 0.2) {
					int classId = classIdPoint.x;
					//printf(" class: %d, conf: %0.2f\n", classId, confidence);
					bboxes[bboxIdx][count].classId = classId;
					bboxes[bboxIdx][count].confidence = confidence;
					bboxes[bboxIdx][count].x = data[0];
					bboxes[bboxIdx][count].y = data[1];
					bboxes[bboxIdx][count].w = data[2];
					bboxes[bboxIdx][count].h = data[3];
					count++;
				}
			}
		}
		nboxes[bboxIdx] = count;

		mtx_a.unlock();
	}
}


ssize_t CDNeuralNet::get_output_boxes(struct _bbox *buf, size_t len) {

	mtx_a.lock();

	int count = 0;
	for (int i=0; i<nboxes[bboxIdx]; ++i) {
		if (count > len) {
			break;
		}
		if (bboxes[bboxIdx][i].confidence > 0.4 && bboxes[bboxIdx][i].classId == 0) {
			buf[count].classId = bboxes[bboxIdx][i].classId;
			buf[count].confidence = bboxes[bboxIdx][i].confidence;
			buf[count].x = bboxes[bboxIdx][i].x;
			buf[count].y = bboxes[bboxIdx][i].y;
			buf[count].w = bboxes[bboxIdx][i].w;
			buf[count].h = bboxes[bboxIdx][i].h;
			count++;
		}
	}
	mtx_a.unlock();

	return count;
}


