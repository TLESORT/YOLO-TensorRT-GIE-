#ifndef CAFFE_BOX_DRAW_H
#define CAFFE_BOX_DRAW_H

//#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <VideoGrabber/CVideoGrabber>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "NvInfer.h"
#include "NvCaffeParser.h"

#define NUM_CLASSES 20
#define NUM_CELLS 7
#define NUM_TOP_CLASSES 2
#define THRESHOLD 0.02
#define TEXT_SCALE 0.5

#define LOG_GIE "[GIE]  "




struct outputLayer
{
	std::string name;
	nvinfer1::Dims3 dims;
	uint32_t size;
	float* CPU;
	float* CUDA;
};

struct box{
    float x, y, w, h;
};

struct sortable_bbox{
    int index;
    int idx_class;
    float **probs;
};

extern const char *voc_names[NUM_CLASSES];
extern float colors[6][3];

class BoxDrawer
{
public:
	BoxDrawer(int nb);
	void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
	static int nms_comparator(const void *pa, const void *pb);
	void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, float w, float h, float thresh, float **probs, box *boxes, int only_objectness);
	void draw_detections(cv::Mat& im, int num, float thresh, box *boxes, float **probs, const char **names);

	
private:
	float overlap(float x1, float w1, float x2, float w2);
	float box_intersection(box a, box b);
	float box_union(box a, box b);
	float box_iou(box a, box b);
	float get_color(int c, int x, int max);
};
#endif


