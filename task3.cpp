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

void combineDetections(vector<Rect> viola_dartboards, int a, int b, int r,Mat &frame, vector<Rect> &combined ) {
    //Point center(a,b);
    for (int detect=0; detect<viola_dartboards.size();detect++){

      if ((viola_dartboards[detect].x<a && a<(viola_dartboards[detect].x + viola_dartboards[detect].width) && viola_dartboards[detect].y<b && b<(viola_dartboards[detect].y + viola_dartboards[detect].height)) && ((viola_dartboards[detect].width < (2.25*r) && viola_dartboards[detect].width > (1.75*r)))) 
      //checks whether center is in rectangle and if radius is approximately half the rectangle
      {
      	Rect combined_rect(viola_dartboards[detect].x, viola_dartboards[detect].y, viola_dartboards[detect].width,viola_dartboards[detect].height);
      	combined.push_back(combined_rect); 
      //append the rectangle to the combined vector
      
      rectangle(frame, Point(viola_dartboards[detect].x, viola_dartboards[detect].y), Point(viola_dartboards[detect].x +
        viola_dartboards[detect].width, viola_dartboards[detect].y + viola_dartboards[detect].height), Scalar( 0, 255, 0 ), 2);
      //draw rectangle
      
      //erase the extra rectangles
      for(int i=0; i<combined.size()-1;i++){
      	for(int j=i+1;j<combined.size();j++){
      		if(combined[i]==combined[j]){
      			combined.erase(combined.begin()+j);
      		}
      	}
      }

      }
    }
  return;
}

void convertGrey(Mat src, Mat &src_gray) {
  // Prepare Image by turning it into Grayscale and normalising lighting
  cvtColor( src, src_gray, CV_BGR2GRAY );
}

void thresh(Mat input, Mat &output, uchar thresh) {
  // Threshold by looping through all pixels
  output.create(input.size(), input.type());

  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      uchar pixel = input.at<uchar>(i, j);
      if (pixel > thresh) output.at<uchar>(i, j) = 255;
      else output.at<uchar>(i, j) = 0;
    }
  }
}

void sobelEdges(Mat src_gray, Mat &thresh_mag) {
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  /// Generate gradientX and gradientY
  Mat gradientX, gradientY;
  Mat abs_gradientX, abs_gradientY;
  Mat magnitude;

  /// Gradient X
  Sobel( src_gray, gradientX, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( gradientX, abs_gradientX );

  /// Gradient Y
  Sobel( src_gray, gradientY, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( gradientY, abs_gradientY );

  /// Total Gradient
  addWeighted( abs_gradientX, 0.5, abs_gradientY, 0.5, 0, magnitude );

  // Create threshold from absolute image
  thresh(magnitude, thresh_mag, 130); 
}

// TODO
void houghCircles(Mat thresh_mag,vector<Rect> viola_dartboards, Mat &frame, vector<Rect> &combined) {
  int min_radius = 50;
  int max_radius = 120;

  int dim1 = thresh_mag.cols;
  int dim2 = thresh_mag.rows;
  int dim3 = max_radius - min_radius;

  int sizes[3] = {dim1,dim2,dim3};
  cv::Mat accumulator = cv::Mat(3, sizes, CV_32F, cv::Scalar(0));
  cv::Mat thresh_accumulator = cv::Mat(3, sizes, CV_32F, cv::Scalar(0));

 //VOTING
  int a;
  int b;

  for (int i = 0; i < dim2; i++) {
    for (int j = 0; j < dim1; j++) {
      if (thresh_mag.at<uchar>(i,j) == 255) {
        for (int r = 0; r < dim3; r++) {
          for (int theta = 0; theta <= 360; theta+=3) {
            a = j - ((r+min_radius) * cos(theta * M_PI / 180));
            b = i - ((r+min_radius) * sin(theta * M_PI / 180));
            if(((0 <= a) && (a < dim1)) && ((0 <= b) && (b < dim2))) {
              accumulator.at<float>(a,b,r) += 1;
            }
          }
        }
      }
    }
  }

  //find maximum votes
  int max = 0;
  int max_r=0;
  for (int i = 0; i< dim1; i++) {
        for (int j = 0; j < dim2; j++) {
          for (int k = 0; k < dim3; k++) {
            if (accumulator.at<float>(i,j,k) > max) {
              max = accumulator.at<float>(i,j,k);
            }
          }
        }
      }

      //printf("%d\n",max);
     // create thresh_accumulator and combine detections
  for(int a=0; a<dim1; a++){
    for(int b=0; b<dim2;b++){
      for(int r=0;r<dim3;r++){
        if(accumulator.at<float>(a,b,r)>=(max*0.48)){
          thresh_accumulator.at<float>(a,b,r) =accumulator.at<float>(a,b,r);
          combineDetections(viola_dartboards,a,b,(r+min_radius), frame, combined);
        }

      }
    }
  }






  //Creating the 2D space
  Mat hough2D;
  hough2D.create(dim2,dim1, thresh_mag.type()); 
  Mat hough2D_thresh;
  hough2D_thresh.create(dim2,dim1, thresh_mag.type());     // use this to create 2D space
  float rad_total;
  float rad_total_thresh;
  for (int a = 0; a < dim1; a++) {
    for (int b = 0; b < dim2; b++) {
      rad_total=0;
      rad_total_thresh=0;
      for (int r = 0; r < dim3; r++) {
        rad_total += accumulator.at<float>(a,b,r);
       	rad_total_thresh+=thresh_accumulator.at<float>(a,b,r);
      }
      hough2D.at<uchar>(b,a)=rad_total;
      hough2D_thresh.at<uchar>(b,a)=rad_total_thresh;
    }
  }
  //to display 2Dspace
  //imwrite("hough2D.jpg",hough2D);
  //imwrite("hough2D_thresh.jpg",hough2D_thresh);
}

void violaJonesDetector(Mat src, vector<Rect> &viola_dartboards) {
  // Perform Viola-Jones Object Detection
  cascade.detectMultiScale( src, viola_dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  // Draw box around faces found
  // for( int i = 0; i < viola_dartboards.size(); i++ ) {
  //   rectangle(out, Point(viola_dartboards[i].x, viola_dartboards[i].y), Point(viola_dartboards[i].x + viola_dartboards[i].width, viola_dartboards[i].y + viola_dartboards[i].height), Scalar( 0, 255, 0 ), 2);
  // }
  return;
}
void drawBoxes(std::vector<Rect> rects, Mat frame, Scalar rectColor) {
  for( int i = 0; i < rects.size(); i++ )
  {
    rectangle(frame, Point(rects[i].x, rects[i].y), Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height), rectColor, 2);
  }
}
double Intersection_over_Union(Rect rect1, Rect rect2) {
  int area_intersect = (rect1 & rect2).area();
  int area_union = rect1.area() + rect2.area() - area_intersect;
  return ((double)area_intersect / area_union);
}
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

int main( int argc, const char** argv )
{
   // Read Input Image
  Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  cv::Mat src_gray, thresh_mag;
  std::vector<Rect> viola_dartboards,combined,dart0,dart1,dart2,dart3,dart4,dart5,dart6,dart7,dart8,dart9,dart10,dart11,dart12,dart13,dart14,dart15;


  // Load the Strong Classifier in a structure called `Cascade'
  if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  convertGrey(frame, src_gray);

  sobelEdges(src_gray, thresh_mag);

  //imwrite("threshold.jpg", thresh_mag);
  
  violaJonesDetector(src_gray, viola_dartboards);
  houghCircles(thresh_mag,viola_dartboards,frame,combined);

  Rect D0(445,15,150,175);
  dart0.push_back(D0);

  Rect D1(200,134,190,190);
  dart1.push_back(D1);

  Rect D2(80,80,125,125);
  dart2.push_back(D2);

  Rect D3(325,150,65,65);
  dart3.push_back(D3);

  Rect D4(185,95,165,205);
  dart4.push_back(D4);

  Rect D5(435,140,100,105);
  dart5.push_back(D5);

  Rect D6(195,95,95,100);
  dart6.push_back(D6);

  Rect D7(255,170,150,140);
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

  //check_intersect(combined,dart2,frame);



  //combineDetections();

  // 4. Save Result Image
  imwrite( "detected.jpg", frame );

  return 0;
}
