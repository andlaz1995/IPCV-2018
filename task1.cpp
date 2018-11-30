/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - overlap.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

// calculates the Intersection Over Union of two bounding boxes
// this gives the ratio of how accurate the detection was
// generally, >0.5 is good
// area of union = area(A) + area(B) - area(intersection)
double Intersection_over_Union(Rect rect1, Rect rect2) {
	int area_intersect = (rect1 & rect2).area();
	int area_union = rect1.area() + rect2.area() - area_intersect;
	return ((double)area_intersect / area_union);
}


// draw bounding boxes for the given vector of rectangles
void drawBoxes(std::vector<Rect> rects, Mat frame, Scalar rectColor) {
	for( int i = 0; i < rects.size(); i++ )
	{
		rectangle(frame, Point(rects[i].x, rects[i].y), Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height), rectColor, 2);
	}
}

//draw real faces using coordinates given in detect and display and find whether they are true positive.
void fmeasure(double TP,double total_detect, double total_existing){
	double precision = (double)TP/ (double)total_detect;
	double recall = (double)TP/ (double) total_existing;
	double fmeasure = 2*((precision*recall)/(precision+recall));
	cout<< "true positive matches:"<< TP <<"\n";
	cout<< "total detected: " << total_detect<<"\n";
	cout<< "precision: " << double(precision)<<"\n";
	cout<< "recall: " << double(recall)<<"\n";
	cout<< "fmeasure: " << double(fmeasure)<<"\n";
}


void check_intersect(std::vector<Rect> rect_faces,std::vector<Rect> rect_real, Mat frame){
	drawBoxes(rect_real, frame, Scalar(200, 0, 255));
	int true_positive_matches =0 ;
	for( int i = 0; i < rect_real.size(); i++ ){
		for(int j=0; j<rect_faces.size();j++){
			if( bool intersects = (Intersection_over_Union(rect_real[i],rect_faces[j])> 0.5)){
				printf("%f\n", Intersection_over_Union(rect_real[i], rect_faces[j]));
				true_positive_matches++;
				}
			}
		}
		fmeasure(true_positive_matches,(double)rect_faces.size(),(double)rect_real.size());
	}
/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces, dart4, dart5, dart9, dart13, dart14, dart15;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

   // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

   // 4. Draw box around faces found
	drawBoxes(faces, frame, Scalar( 0, 255, 0 ));
// Rect(x,y,width,height)
	// declare rectangle bounding boxes
	Rect DART4(349,114,110,144); // blue
	dart4.push_back(DART4);

	Rect DART51(70,142,50,60);//11
	dart5.push_back(DART51);
	Rect DART52(55,256,50,60);//9
	dart5.push_back(DART52);
	Rect DART53(197,218,50,60);//4
	dart5.push_back(DART53);
	Rect DART54(256,179,50,60);//1
	dart5.push_back(DART54);
	Rect DART55(298,250,50,60);//6
	dart5.push_back(DART55);
	Rect DART56(381,192,50,60);//12
	dart5.push_back(DART56);
	Rect DART57(433,238,50,60);//5
	dart5.push_back(DART57);
	Rect DART58(519,179,50,60);// 2
	dart5.push_back(DART58);
	Rect DART59(564,254,50,60);//8
	dart5.push_back(DART59);
	Rect DART510(650,193,50,60);//3
	dart5.push_back(DART510);
	Rect DART511(683,253,50,60);//7
	dart5.push_back(DART511);

	Rect DART13(427,139,94,112);
	dart13.push_back(DART13);

	Rect DART141(474,235,74,82);//5
	dart14.push_back(DART141);
	Rect DART142(735,202,83,88);//4
	dart14.push_back(DART142);

	Rect DART151(68,133,51,82); //2
	dart15.push_back(DART151);
	Rect DART152(378,115,39,70);//miss
	dart15.push_back(DART152);
	Rect DART153(545,127,54,78);//1
	dart15.push_back(DART153);

	Rect DART9(93,225,99,118);
	dart9.push_back(DART9);
	//10 is a miss
	//13 is a miss
	//14 is a miss





	check_intersect(faces,dart14,frame);

}
