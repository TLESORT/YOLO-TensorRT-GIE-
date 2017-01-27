#include "YOLODraw.h"
#include <opencv2/core.hpp>

#define BATCH_SIZE 1
#define BATCH_ITERS 1


float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
const char *voc_names[NUM_CLASSES] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

// marvellous Constructor
BoxDrawer::BoxDrawer(int nb){
	printf("\n");
}

int BoxDrawer::nms_comparator(const void *pa, const void *pb)
{
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.idx_class] - b.probs[b.index][b.idx_class];
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

float BoxDrawer::overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float BoxDrawer::box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float BoxDrawer::box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float BoxDrawer::box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

float BoxDrawer::get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}

void BoxDrawer::draw_detections(cv::Mat& im, int num, float thresh, box *boxes, float **probs, const char **names)
{
    int i;
    for(i = 0; i < num; ++i){
		float max = -1e10; int idx_class = -1;
		for (int j = 0; j < NUM_CLASSES; ++j) {
			if (probs[i][j] > max) {
				max = probs[i][j];
				idx_class = j;
			}
		}
        float prob = probs[i][idx_class];
        if(prob > thresh){
            int width = pow(prob, 1./2.)*10+1;
            //printf("%s: %.2f\n", names[idx_class], prob);
            int offset = idx_class*17 % NUM_CLASSES;
            float red = get_color(0,offset,NUM_CLASSES) * 255;
            float green = get_color(1,offset,NUM_CLASSES) * 255;
            float blue = get_color(2,offset,NUM_CLASSES) * 255;
            box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.cols;
            int right = (b.x+b.w/2.)*im.cols;
            int top   = (b.y-b.h/2.)*im.rows;
            int bot   = (b.y+b.h/2.)*im.rows;

            if(left < 0) left = 0;
            if(right > im.cols-1) right = im.cols-1;
            if(top < 0) top = 0;
            if(bot > im.rows-1) bot = im.rows-1;
			cv::Size szTxt = cv::getTextSize(names[idx_class], cv::FONT_HERSHEY_SIMPLEX, TEXT_SCALE, 1, NULL);
			cv::rectangle(im, cv::Rect(left, top-1-szTxt.height, szTxt.width, szTxt.height), cv::Scalar(red, green, blue), -1);
			//cv::putText(im, names[idx_class], cv::Point(left, top-2), cv::FONT_HERSHEY_SIMPLEX, TEXT_SCALE, cv::Scalar(255, 255, 255), 2);
			cv::putText(im, names[idx_class], cv::Point(left, top-2), cv::FONT_HERSHEY_SIMPLEX, TEXT_SCALE, cv::Scalar(0, 0, 0), 2);
			cv::rectangle(im, cv::Rect(left, top, right-left, bot-top), cv::Scalar(red, green, blue), 2);
        }
    }
}

void BoxDrawer::do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    sortable_bbox *s = (sortable_bbox*)calloc(total, sizeof(sortable_bbox));

    for(i = 0; i < total; ++i){
        s[i].index = i;       
        s[i].idx_class = 0;
        s[i].probs = probs;
    }

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            s[i].idx_class = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for(i = 0; i < total; ++i){
            if(probs[s[i].index][k] == 0) continue;
            box a = boxes[s[i].index];
            for(j = i+1; j < total; ++j){
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}

void BoxDrawer::convert_yolo_detections(float *predictions, int classes, int num, int square, int side, float w, float h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    for (i = 0; i < side*side; ++i){
        int row = i / side;
        int col = i % side;
        for(n = 0; n < num; ++n){
            int index = i*num + n;
            int p_index = side*side*classes + i*num + n;
            float scale = predictions[p_index];
//printf("side : %d , index : %d , p_index : %d , prediction : %f\n",side,index,p_index,scale);
            int box_index = side*side*(classes + num) + (i*num + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
			
			for(j = 0; j < classes; ++j){
				int class_index = i*classes;
				float prob = scale*predictions[class_index+j]; //

				probs[index][j] = (prob > thresh) ? prob : 0;
			}
			
			if(only_objectness){
				probs[index][0] = scale;
			}
        }
    }
}


