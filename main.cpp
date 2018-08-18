#include "floydmarshall.h"
#include "isomap.h"
#include "math.h"
#include "pca.h"
#include <iostream>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

// functions for drawing
void Draw3DManifold(Mat& dataMatrix, char const* name, int nSamplesI, int nSamplesJ);
void Draw2DManifold(Mat& dataMatrix, char const* name, int nSamplesI, int nSamplesJ);
void start();

// functions for dimensionality reduction
// first parameter: features saved in rows
// second parameter: desired # of dimensions ( here: 2)
//Mat reducePCA(Mat& dataMatrix, unsigned int dim);
//Mat reduceIsomap(Mat& dataMatrix, unsigned int dim, int nSamplesI, int nSamplesJ);
// Mat reduceLLE(Mat &dataMatrix, unsigned int dim);

int main(int argc, char** argv)
{
    start();

    waitKey(0);
    return 0;
}

void start()
{
    // generate Data Matrix
    unsigned int nSamplesI = 10;
    unsigned int nSamplesJ = 10;
    Mat dataMatrix = Mat(nSamplesI * nSamplesJ, 3, CV_64F);
    // noise in the data
    double noiseScaling = 1000.0;

    for (int i = 0; i < nSamplesI; i++) {
        for (int j = 0; j < nSamplesJ; j++) {
            dataMatrix.at<double>(i * nSamplesJ + j, 0) = (i / (double)nSamplesI * 2.0 * 3.14 + 3.14)
                    * cos(i / (double)nSamplesI * 2.0 * 3.14)
                + (rand() % 100) / noiseScaling;
            dataMatrix.at<double>(i * nSamplesJ + j, 1) = (i / (double)nSamplesI * 2.0 * 3.14 + 3.14)
                    * sin(i / (double)nSamplesI * 2.0 * 3.14)
                + (rand() % 100) / noiseScaling;
            dataMatrix.at<double>(i * nSamplesJ + j, 2) = 10.0 * j / (double)nSamplesJ
                + (rand() % 100) / noiseScaling;
        }
    }

    // Draw 3D Manifold
    // Draw3DManifold(dataMatrix, "3D Points", nSamplesI, nSamplesJ);

    // PCA
    Mat temp;
    /*dataMatrix.copyTo(temp);
    Mat dataPCA = reducePCA(temp, 2);
    Draw2DManifold(dataPCA, "PCA", nSamplesI, nSamplesJ);*/

    // Isomap
    for (int i = 2; i < 3; i++) {
        dataMatrix.copyTo(temp);
        Mat dataIsomap = reduceIsomap(temp, i, nSamplesI, nSamplesJ);
        Draw2DManifold(dataIsomap, to_string(i).c_str(), nSamplesI, nSamplesJ);
    }
}

void Draw3DManifold(Mat& dataMatrix, char const* name, int nSamplesI, int nSamplesJ)
{
    Mat origImage = Mat(1000, 1000, CV_8UC3);
    origImage.setTo(0.0);
    for (int i = 0; i < nSamplesI; i++) {
        for (int j = 0; j < nSamplesJ; j++) {
            Point p1;
            p1.x = dataMatrix.at<double>(i * nSamplesJ + j, 0) * 50.0 + 500.0 - dataMatrix.at<double>(i * nSamplesJ + j, 2) * 10;
            p1.y = dataMatrix.at<double>(i * nSamplesJ + j, 1) * 50.0 + 500.0 - dataMatrix.at<double>(i * nSamplesJ + j, 2) * 10;
            circle(origImage, p1, 3, Scalar(255, 255, 255));

            Point p2;
            if (i < nSamplesI - 1) {
                p2.x = dataMatrix.at<double>((i + 1) * nSamplesJ + j, 0) * 50.0 + 500.0 - dataMatrix.at<double>((i + 1) * nSamplesJ + (j), 2) * 10;
                p2.y = dataMatrix.at<double>((i + 1) * nSamplesJ + j, 1) * 50.0 + 500.0 - dataMatrix.at<double>((i + 1) * nSamplesJ + (j), 2) * 10;

                line(origImage, p1, p2, Scalar(255, 255, 255), 1, 8);
            }
            if (j < nSamplesJ - 1) {
                p2.x = dataMatrix.at<double>((i)*nSamplesJ + j + 1, 0) * 50.0 + 500.0 - dataMatrix.at<double>((i)*nSamplesJ + (j + 1), 2) * 10;
                p2.y = dataMatrix.at<double>((i)*nSamplesJ + j + 1, 1) * 50.0 + 500.0 - dataMatrix.at<double>((i)*nSamplesJ + (j + 1), 2) * 10;

                line(origImage, p1, p2, Scalar(255, 255, 255), 1, 8);
            }
        }
    }

    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, origImage);
}

void Draw2DManifold(Mat& dataMatrix, char const* name, int nSamplesI, int nSamplesJ)
{
    Mat origImage = Mat(1000, 1000, CV_8UC3);
    origImage.setTo(0.0);
    for (int i = 0; i < nSamplesI; i++) {
        for (int j = 0; j < nSamplesJ; j++) {
            Point p1;
            p1.x = dataMatrix.at<double>(i * nSamplesJ + j, 0) * 1000.0 + 500.0;
            p1.y = dataMatrix.at<double>(i * nSamplesJ + j, 1) * 1000.0 + 500.0;
            // circle(origImage,p1,3,Scalar( 255, 255, 255 ));

            Point p2;
            if (i < nSamplesI - 1) {
                p2.x = dataMatrix.at<double>((i + 1) * nSamplesJ + j, 0) * 1000.0 + 500.0;
                p2.y = dataMatrix.at<double>((i + 1) * nSamplesJ + j, 1) * 1000.0 + 500.0;
                line(origImage, p1, p2, Scalar(255, 255, 255), 1, 8);
            }
            if (j < nSamplesJ - 1) {
                p2.x = dataMatrix.at<double>((i)*nSamplesJ + j + 1, 0) * 1000.0 + 500.0;
                p2.y = dataMatrix.at<double>((i)*nSamplesJ + j + 1, 1) * 1000.0 + 500.0;
                line(origImage, p1, p2, Scalar(255, 255, 255), 1, 8);
            }
        }
    }

    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, origImage);
    imwrite((String(name) + ".png").c_str(), origImage);
}
