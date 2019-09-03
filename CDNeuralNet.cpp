
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "CDNeuralNet.hpp"

std::mutex mtx_a;

int bboxIdx = 0;
int nboxes[2];

std::vector<int> classIds[2];
std::vector<float> confidences[2];
std::vector<cv::Rect> bboxes[2];

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

		classIds[bboxIdx].clear();
		confidences[bboxIdx].clear();
		bboxes[bboxIdx].clear();

		float confThreshold = 0.4;

		for (size_t i=0; i<outs.size(); ++i) {
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
				cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				cv::Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

				int classId = classIdPoint.x;
				if (confidence > confThreshold && classId==0) {

					classIds[bboxIdx].push_back(classId);
					confidences[bboxIdx].push_back(confidence);

					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					bboxes[bboxIdx].push_back(cv::Rect(left, top, width, height));
					count++;
				}
			}
		}

		float nmsThreshold = 0.45;

		std::vector<int> indices;
		cv::dnn::NMSBoxes(bboxes[bboxIdx], confidences[bboxIdx], confThreshold, nmsThreshold, indices);

		nboxes[bboxIdx] = indices.size();

		//if (indices.size() != count) {
		//	printf("nms.  old: %d, new: %d\n", count, indices.size());
		//}

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

		buf[count].classId = classIds[bboxIdx][i];
		buf[count].confidence = confidences[bboxIdx][i];
		buf[count].x = bboxes[bboxIdx][i].x;
		buf[count].y = bboxes[bboxIdx][i].y;
		buf[count].w = bboxes[bboxIdx][i].width;
		buf[count].h = bboxes[bboxIdx][i].height;
		count++;
	}
	mtx_a.unlock();

	return count;
}


