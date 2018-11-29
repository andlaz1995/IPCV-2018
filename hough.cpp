#define _USE_MATH_DEFINES

#include <cmath>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

String cascade_name = "./dartcascade/cascade.xml";
CascadeClassifier cascade;
void detectAndDisplay( Mat frame );
  Mat frame_gray;
  Mat src_gray2;



void detectAndDisplay( Mat src ){
  std::vector<Rect> dartboards;
  Mat src_gray;

  // 1. Prepare Image by turning it into Grayscale and normalising lighting
  cvtColor( src, src_gray2, CV_BGR2GRAY );
  equalizeHist( src_gray2, src_gray2 );

  // 2. Perform Viola-Jones Object Detection
  cascade.detectMultiScale( src_gray2, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
  // QUESTION: do we need to adjust any of these params?

       // 3. Print number of Faces found
  std::cout << dartboards.size() << std::endl;

       // 4. Draw box around faces found
  for( int i = 0; i < dartboards.size(); i++ )
  {

    //rectangle(src, Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height), Scalar( 0, 255, 0 ), 2);
    // std::cout << dartboards[i].x << std::endl; // prints bounding box locations
    // std::cout << dartboards[i].y << std::endl;
    // printf("..");

  }
}
