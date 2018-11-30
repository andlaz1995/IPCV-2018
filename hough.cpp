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


void convertGrey(Mat src, Mat &src_gray) {
  // Prepare Image by turning it into Grayscale and normalising lighting
  cvtColor( src, src_gray, CV_BGR2GRAY );
  // equalizeHist(src_gray, src_gray);
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
  thresh(magnitude, thresh_mag, 35); //good value for img 1
}

// TODO
void houghCircles(Mat thresh_mag) {
  int min_radius = 10;
  int max_radius = 120;

  // allocate memory for accumulator array
  int dim1 = thresh_mag.rows;
  int dim2 = thresh_mag.cols;
  int dim3 = max_radius - min_radius;
  int *** accumulator = (int ***)malloc(dim1*sizeof(int**));
  for (int i = 0; i< dim1; i++) {
    accumulator[i] = (int **) malloc(dim2*sizeof(int *));
    for (int j = 0; j < dim2; j++) {
        accumulator[i][j] = (int *)malloc(dim3*sizeof(int));
    }
  }
  // initialise the 3d array with 0s
  for (int i = 0; i< dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      for (int k = 0; k < dim3; k++) {
        accumulator[i][j][k] = 0;
      }
    }
  }

  for (int i = 0; i < thresh_mag.rows; i++) {
    for (int j = 0; j < thresh_mag.cols; j++) {
      if (thresh_mag.at<uchar>(i, j) == 255) {
        for (int r = min_radius; r <= max_radius; r++) {
          double a = 0;
          double b = 0;
          for (int theta = 0; theta <= 360; theta++) {
            a = i - (r * cos(theta * M_PI / 180));
            b = j - (r * sin(theta * M_PI / 180));
            if(((0 <= a) && (a < dim1)) && ((0 <= b) && (b < dim2))) {
              accumulator[a, b, r] += 1;
            }
          }
        }
      }
    }
  }

  int total[dim1][dim2];
  for (int a = 0; a < thresh_mag.rows; a++) {
    for (int b = 0; b < thresh_mag.cols; b++) {
      for (int r = min_radius; r < max_radius; r++) {
        total[a][b] = total[a][b] + accumulator[a, b, r];
      }
    }
  }
  return;
}

void violaJonesDetector(Mat src, vector<Rect> viola_dartboards) {
  // Perform Viola-Jones Object Detection
  cascade.detectMultiScale( src, viola_dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  // Draw box around faces found
  // for( int i = 0; i < viola_dartboards.size(); i++ ) {
  //   rectangle(out, Point(viola_dartboards[i].x, viola_dartboards[i].y), Point(viola_dartboards[i].x + viola_dartboards[i].width, viola_dartboards[i].y + viola_dartboards[i].height), Scalar( 0, 255, 0 ), 2);
  // }
  return;
}

// TODO
void combineDetections() {
  return;
}

int main( int argc, const char** argv )
{
   // Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  cv::Mat src_gray, thresh_mag;
  std::vector<Rect> viola_dartboards;


  // Load the Strong Classifier in a structure called `Cascade'
  if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  convertGrey(frame, src_gray);

  sobelEdges(src_gray, thresh_mag);

  imshow("threshold", thresh_mag);
  waitKey(0);

  // houghCircles(magnitude);

	violaJonesDetector(src_gray, viola_dartboards);

  combineDetections();

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}


/*
void convolve (Mat input, Mat kernel, Mat &output) {
    output.create(input.size(), input.type());

    // pad input to prevent border effects
    double unscaledOutput[input.rows][input.cols];
    int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
    int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

    cv::Mat paddedInput;
    cv::copyMakeBorder(input, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, cv::BORDER_REPLICATE);

    // now we can do the convolution
    for ( int i = 0; i < input.rows; i++ ) {
        for( int j = 0; j < input.cols; j++ ) {
            double sum = 0.0;

            for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
                for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
                    // find the correct indices we are using
                    int imagex = i + m + kernelRadiusX;
                    int imagey = j + n + kernelRadiusY;
                    int kernelx = m + kernelRadiusX;
                    int kernely = n + kernelRadiusY;

                    // get the values from the padded image and the kernel
                    int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
                    double kernalval = kernel.at<double>( kernelx, kernely );

                    // do the multiplication
                    sum += imageval * kernalval;
                }
            }
            // set the output value as the sum of the convolution
            unscaledOutput[i][j] = sum;
        }
    }

    // scale the convolution by a ratio of max-min
    double min = 999999;
    double max = 0;
    for ( int i = 0; i < input.rows; i++ ) {
        for( int j = 0; j < input.cols; j++ ) {
            if(unscaledOutput[i][j] > max) {
                max = unscaledOutput[i][j];
            }
            if(unscaledOutput[i][j] < min) {
                min = unscaledOutput[i][j];
            }
        }
    }

    for ( int i = 0; i < input.rows; i++ ) {
        for( int j = 0; j < input.cols; j++ ) {
            output.at<uchar>(i, j) = (255)/(max-min)*(unscaledOutput[i][j] - max);
        }
    }
}


// TODO
void sobelEdges(Mat input, Mat &magnitude) {
    cv::Mat gradientX, gradientY;
    gradientX.create(input.size(), input.type());
    gradientY.create(input.size(), input.type());

    magnitude.create(input.size(), input.type());

    // values are flipped 180 because the convolve function is actually correlation??
    float xvalues[] = {-1,0,1,-2,0,2,-1,0,1};
    cv::Mat kernelX(3,3, CV_32F, xvalues);

    float yvalues[] = {1,-2,-1,0,0,0,1,2,1};
    cv::Mat kernelY(3,3, CV_32F, yvalues);

    // find dx and dy
    convolve(input, kernelX, gradientX);
    convolve(input, kernelY, gradientY);

    // imshow("gradx", gradientX);
    // imshow("grady", gradientY);

    // calculate the magnitude
    double unscaledMag[input.rows][input.cols];

    for (int i = 0; i < input.rows; i++) {
      for (int j = 0; j < input.cols; j++) {
        unscaledMag[i][j] = sqrt( (gradientX.at<uchar>(i, j) * gradientX.at<uchar>(i, j)) + (gradientY.at<uchar>(i, j) * gradientY.at<uchar>(i, j)) );
      }
    }

    // scale the magnitude
    double min = 999999;
    double max = 0;
    for ( int i = 0; i < input.rows; i++ ) {
        for( int j = 0; j < input.cols; j++ ) {
            if(unscaledMag[i][j] > max) {
                max = unscaledMag[i][j];
            }
            if(unscaledMag[i][j] < min) {
                min = unscaledMag[i][j];
            }
        }
    }

    for ( int i = 0; i < input.rows; i++ ) {
        for( int j = 0; j < input.cols; j++ ) {
            magnitude.at<uchar>(i, j) = (255)/(max-min)*(unscaledMag[i][j] - max);
        }
    }

    return;
}
*/
