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
String cascade_name = "./dartcascade/cascade.xml";
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
	cout<< "total: " << total_detect<<"\n";
	cout<< "precision: " << double(precision)<<"\n";
	cout<< "recall: " << double(recall)<<"\n";
	cout<< "fmeasure: " << double(fmeasure)<<"\n";
}


void check_intersect(std::vector<Rect> rect_faces,std::vector<Rect> rect_real, Mat frame){
	drawBoxes(rect_real, frame, Scalar(200, 0, 255));
	int true_positive_matches =0 ;
	for( int i = 0; i < rect_real.size(); i++ ){
		for(int j=0; j<rect_faces.size();j++){
			if( bool intersects = (Intersection_over_Union(rect_real[i],rect_faces[j])> 0.5)){//figure out a good  value for the ratio
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
	std::vector<Rect> faces,dart0,dart1,dart2,dart3,dart4,dart5,dart6,dart7,dart8,dart9,dart10,dart11,dart12,dart13,dart14,dart15;
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

	// declare rectangle bounding boxes

	// Rect(x,y,width,height)

	Rect D0(445,15,150,175);
	dart0.push_back(D0);

	Rect D1(200,134,190,190);
	dart1.push_back(D1);

	Rect D2(70,70,145,145);
	dart2.push_back(D2);

	Rect D3(325,150,65,65);
	dart3.push_back(D3);

	Rect D4(185,95,165,205);
	dart4.push_back(D4);

	Rect D5(435,140,100,105);
	dart5.push_back(D5);

	Rect D6(195,95,95,100);
	dart6.push_back(D6);

	Rect D7(255,170,95,140);
	dart7.push_back(D7);

	Rect D8_1(845,215,110,110);
	dart8.push_back(D8_1);
	Rect D8_2(70,255,55,80);
	dart8.push_back(D8_2);

	Rect D9(200,45,235,235);
	dart9.push_back(D9);

	Rect D10_1(75,90,122,130);
	dart10.push_back(D10_1);
	Rect D10_2(585,130,55,80);
	dart10.push_back(D10_2);
	Rect D10_3(915,150,37,64);
	dart10.push_back(D10_3);

	Rect D11(175,105,55,50);
	dart11.push_back(D11);

	Rect D12(150,52,75,180);
	dart12.push_back(D12);

	Rect D13(280,120,120,130);
	dart13.push_back(D13);

	Rect D14_1(105,85,155,155);
	dart14.push_back(D14_1);
	Rect D14_2(970,80,155,155);
	dart14.push_back(D14_2);

	Rect D15(160,55,120,140);
	dart15.push_back(D15);

	check_intersect(faces,dart0,frame);
	///dart6,12,14 gets f1=0

}
