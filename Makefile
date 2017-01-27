CC=g++
CFLAGS=-O2 -std=c++0x -I. -I/usr/local/cuda/include -I/home/ubuntu/opencv-3.1.0/include


LIBDIRS= -L/usr/local/cuda/lib64 -L/home/ubuntu/opencv-3.1.0/lib

CVLIBS = -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_video -lopencv_videoio -lnvcaffe_parser -lnvinfer -lopencv_features2d -lopencv_imgcodecs -lopencv_objdetect
LDFLAGS=$(LIBDIRS) -lm -lstdc++ $(CVLIBS) -lcuda -lcublas -lcurand -lcudart -lboost_system -lVideoGrabber 
#-lgstreamer-1.0

DEPS = YOLODraw.h

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

all: YOLODetector

YOLODetector: YOLODetector.o YOLODraw.o
	gcc -o $@ $^ $(CFLAGS) $(LDFLAGS)


.PHONY: clean

clean:
	rm -f *.o YOLODetector
