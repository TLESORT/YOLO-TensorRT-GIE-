


#include <opencv2/core.hpp>


#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <map>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <cstring>
#include "YOLODraw.h"


using namespace nvinfer1;
using namespace nvcaffeparser1;


// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 448;
static const int INPUT_W = 448;
static const int OUTPUT_SIZE = 1470;
const int BATCH_SIZE=1;

bool mEnableFP16=true;
bool mOverride16=false;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "result";



/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void WrapInputLayer(std::vector<std::vector<cv::Mat> >& input_channels,float* buffer) {

	float* input_data = buffer;

	for (int n = 0; n < 1; ++n) {
		input_channels.push_back(std::vector<cv::Mat>());
		for (int i = 0; i < 3; ++i) {
			cv::Mat channel(INPUT_H, INPUT_W, CV_32FC1, input_data);
			input_channels[n].push_back(channel);
			input_data += INPUT_H * INPUT_W;
		}
	}
}

#define CHECK(status)					\
{							\
    if (status != 0)				\
    {						\
        std::cout << "Cuda failure: " << status;\
		abort();				\
	}						\
}


// Logger for GIE info/warning/errors
class Logger : public ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;


void Preprocess(const cv::Mat& img, std::vector<std::vector<cv::Mat>> &input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	int num_channels_ = input_channels[0].size();
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Size input_geometry = cv::Size(input_channels[0][0].cols, input_channels[0][0].rows);

	cv::Mat sample_resized;
	/*preproc-resample */
	if (sample.size() != input_geometry)
		cv::resize(sample, sample_resized, input_geometry);
	else
		sample_resized = sample;
	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);
	/* END */
	/* preproc-normalize */
	cv::Mat sample_normalized(448, 448, CV_32FC3);
	bool _rescaleTo01=true;
	if (_rescaleTo01)
		sample_float = sample_float / 255.f;

	sample_float.convertTo(sample_normalized, CV_32FC3);

    	for (int n = 0; n < BATCH_SIZE; ++n) {
		cv::split(sample_normalized, input_channels[n]);
	}
}

int main(int argc, char** argv)
{


	cv::Mat frame=cv::imread("cat.jpg",CV_LOAD_IMAGE_UNCHANGED);
	//frame=cv::Mat::zeros(448, 448, CV_32FC3);

	void** mInputCPU= (void**)malloc(2*sizeof(void*));;
	cudaHostAlloc((void**)&mInputCPU[0],  3*INPUT_H*INPUT_W*sizeof(float), cudaHostAllocDefault);

	std::vector<std::vector<cv::Mat> > input_channels;
	WrapInputLayer(input_channels,(float*)mInputCPU[0]);
	Preprocess(frame, input_channels);

	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);
	const char* prototxt="yolo_small_modified.prototxt";
	const char* caffemodel="yolo_small.caffemodel";





	mEnableFP16 = (mOverride16 == true) ? false : builder->platformHasFastFp16();
	printf(LOG_GIE "platform %s FP16 support.\n", mEnableFP16 ? "has" : "does not have");
	printf(LOG_GIE "loading %s %s\n", prototxt, prototxt);

	nvinfer1::DataType modelDataType = mEnableFP16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; // create a 16-bit model if it's natively supported

	// parse the caffe model to populate the network, then set the outputs and create an engine
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser *parser = createCaffeParser();
	const IBlobNameToTensor *blobNameToTensor =
				parser->parse(prototxt,		// caffe deploy file
				caffemodel,		// caffe model file
				*network,		// network definition that the parser will populate
				modelDataType);


	assert(blobNameToTensor != nullptr);
	// the caffe file has no notion of outputs
	// so we need to manually say which tensors the engine should generate
	network->markOutput(*blobNameToTensor->find(OUTPUT_BLOB_NAME));
	// Build the engine
	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(16 << 20);//WORKSPACE_SIZE);
	
	// set up the network for paired-fp16 format
	if(mEnableFP16)
		builder->setHalf2Mode(true);

	// Eliminate the side-effect from the delay of GPU frequency boost
	builder->setMinFindIterations(3);
	builder->setAverageFindIterations(2);

	//build
	ICudaEngine *engine = builder->buildCudaEngine(*network);

	IExecutionContext *context = engine->createExecutionContext();

	// run inference
	float prob[OUTPUT_SIZE];
	//doInference(*context, (float*)mInputCPU[0], prob, 1);

	//const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine->getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME); 
	int   outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE *3* INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], mInputCPU[0], BATCH_SIZE *3* INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context->enqueue(BATCH_SIZE, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(prob, buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));


	// destroy the engine
	context->destroy();
	engine->destroy();

	int MAX_BATCH_SIZE=1;
	BoxDrawer boxDrawer(1);
		
	char ts[50];

	box *boxes = (box*)calloc(NUM_CELLS*NUM_CELLS*NUM_TOP_CLASSES, sizeof(box));
	float **probs = (float**)calloc(NUM_CELLS*NUM_CELLS*NUM_TOP_CLASSES, sizeof(float *));
	for(int j = 0; j < NUM_CELLS*NUM_CELLS*NUM_TOP_CLASSES; ++j) 
		probs[j] = (float*)calloc(NUM_CLASSES, sizeof(float *));


	boxDrawer.convert_yolo_detections(prob, NUM_CLASSES, NUM_TOP_CLASSES, 1, NUM_CELLS, 1, 1, THRESHOLD, probs, boxes, 0);
	boxDrawer.do_nms_sort(boxes, probs, NUM_CELLS*NUM_CELLS*NUM_TOP_CLASSES, NUM_CLASSES, (float)0.5);
	boxDrawer.draw_detections(frame, NUM_CELLS*NUM_CELLS*NUM_TOP_CLASSES, THRESHOLD, boxes, probs, voc_names);

	imwrite( "cat_detection_modified_16bits.jpg", frame );
	return 0;
}
