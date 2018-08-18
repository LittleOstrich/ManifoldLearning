
#include <iostream>
#include <map>
#include <opencv/ml.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

// functions for drawing
void Draw3DManifold2(Mat &dataMatrix, char const *name, int nSamplesI,
                     int nSamplesJ);
void Draw2DManifold2(Mat &dataMatrix, char const *name, int nSamplesI,
                     int nSamplesJ);

// functions for dimensionality reduction
// first parameter: features saved in rows
// second parameter: desired # of dimensions ( here: 2)
Mat reducePCA2(Mat &dataMatrix, unsigned int dim);
Mat reduceIsomap2(Mat &dataMatrix, unsigned int dim);

void start2() {
  // generate Data Matrix
  unsigned int nSamplesI = 10;
  unsigned int nSamplesJ = 10;
  Mat dataMatrix = Mat(nSamplesI * nSamplesJ, 3, CV_64F);
  // noise in the data
  double noiseScaling = 1000.0;

  for (int i = 0; i < nSamplesI; i++) {
    for (int j = 0; j < nSamplesJ; j++) {
      dataMatrix.at<double>(i * nSamplesJ + j, 0) =
          (i / (double)nSamplesI * 2.0 * 3.14 + 3.14) *
              cos(i / (double)nSamplesI * 2.0 * 3.14) +
          (rand() % 100) / noiseScaling;
      dataMatrix.at<double>(i * nSamplesJ + j, 1) =
          (i / (double)nSamplesI * 2.0 * 3.14 + 3.14) *
              sin(i / (double)nSamplesI * 2.0 * 3.14) +
          (rand() % 100) / noiseScaling;
      dataMatrix.at<double>(i * nSamplesJ + j, 2) =
          10.0 * j / (double)nSamplesJ + (rand() % 100) / noiseScaling;
    }
  }

  //// Draw 3D Manifold
  // Draw3DManifold2(dataMatrix, "3D Points",nSamplesI,nSamplesJ);
  //// PCA
  // Mat dataPCA = reducePCA2(dataMatrix,2);
  // Draw2DManifold2(dataPCA,"PCA",nSamplesI, nSamplesJ);

  // Isomap
  Mat dataIsomap = reduceIsomap2(dataMatrix, 2);
  Draw2DManifold2(dataIsomap, "ISOMAP", nSamplesI, nSamplesJ);
}

Mat reducePCA2(Mat &dataMatrix, unsigned int dim) {
  Mat mean = Mat(1, 3, CV_64F);

  double meanz = 0;

  // calculate the mean of each cols
  for (int j = 0; j < dataMatrix.cols; j++) {
    for (int i = 0; i < dataMatrix.rows; i++) {
      meanz += dataMatrix.at<double>(i, j);
    }
    mean.at<double>(0, j) = meanz / dataMatrix.rows;
    meanz = 0;
  }

  // calculate the subtracted dataMatrix
  Mat subDataMatrix = dataMatrix.clone();
  for (int j = 0; j < dataMatrix.cols; j++) {
    for (int i = 0; i < dataMatrix.rows; i++) {
      subDataMatrix.at<double>(i, j) -= mean.at<double>(0, j);
    }
  }
  /*
          cout << "rows of mean: " <<
          cout << "mean:" << mean  << endl;

          cout << "\nsubDataMatrix: " << subDataMatrix << endl;
  */
  Mat TsubDataMatrix;
  transpose(subDataMatrix, TsubDataMatrix);

  Mat cov = (TsubDataMatrix * subDataMatrix) / (subDataMatrix.rows - 1);

  Mat eigenVecMat;
  Mat eigenVal;

  eigen(cov, eigenVal, eigenVecMat);
  /*
          cout << "\ncovariance Matrix" << cov << endl;
          cout << "\n eigenValue size: " << eigenVal.size() << endl;
          cout << "\neigenValue: " << eigenVal << endl;
          cout << "\neigenVecor: " << eigenVecMat << endl;

          cout << "\nsize of eigenVecMat: " << eigenVecMat.size() << endl;
  */

  Mat reduce = Mat(eigenVecMat.rows, dim, CV_64F);

  for (int i = 0; i < reduce.rows; i++) {
    for (int j = 0; j < reduce.cols; j++) {
      reduce.at<double>(i, j) = eigenVecMat.at<double>(j, i);
    }
  }

  /*
          cout << "\neigenVecMat: " << eigenVecMat << endl;
          cout << "\nreduce: " << reduce << endl;
          cout << "size of reduce: " << reduce.size() << endl;
          cout << "\nsize of dataMat: " << dataMatrix.size() << endl;
          cout << "\n dataMat: " << dataMatrix << endl;
  */

  /*
          Mat zwi = Mat(1, dim, CV_64F);
          for (int j = 0; j < reduce.cols; j++) {
                  zwi.at<double>(0, j) = 0;
          }

          for (int j = 0; j < reduce.cols; j++) {
                  for (int i = 0; i < reduce.rows; i++) {
                          zwi.at<double>(0,j) += pow(reduce.at<double>(i, j),
     2);
                  }
          }

          for (int j = 0; j < zwi.cols; j++) {
                  zwi.at<double>(0, j) = sqrt(zwi.at<double>(0, j));
          }
          cout << zwi << endl;
          for (int j = 0; j < reduce.cols; j++) {
                  for (int i = 0; i < reduce.rows; i++) {
                          reduce.at<double>(i, j) = reduce.at<double>(i, j) /
     zwi.at<double>(0, j);
                  }
          }
  */

  Mat end = dataMatrix * reduce / 35;
  return end;
}

void floyds(Mat dis) {
  int i, j, k;
  for (k = 0; k < dis.cols; k++) {
    for (i = 0; i < dis.cols; i++) {
      for (j = 0; j < dis.cols; j++) {
        if ((dis.at<double>(i, k) * dis.at<double>(k, i) != 0) && (i != j)) {
          if ((dis.at<double>(i, k) + dis.at<double>(k, j)) <
                  dis.at<double>(i, j) ||
              (dis.at<double>(i, j) == 0)) {
            dis.at<double>(i, j) = dis.at<double>(i, k) + dis.at<double>(k, j);
          }
        }
      }
    }
  }
}

Mat reduceIsomap2(Mat &dataMatrix, unsigned int dim) {

  int K = 10;
  Mat dis = Mat(dataMatrix.rows, dataMatrix.rows, CV_64F);
  // first get the whole distance Matrix of all the points using Euclideans
  // distance
  for (int i = 0; i < dataMatrix.rows; i++) {
    for (int a = 0; a < dataMatrix.rows; a++) {
      double first = dataMatrix.at<double>(i, 0) - dataMatrix.at<double>(a, 0);
      double second = dataMatrix.at<double>(i, 1) - dataMatrix.at<double>(a, 1);
      double third = dataMatrix.at<double>(i, 2) - dataMatrix.at<double>(a, 2);
      dis.at<double>(i, a) =
          sqrt(pow(first, 2) + pow(second, 2) + pow(third, 2));
    }
  }

  // corrected code:
  Mat zwi = dis.clone();
  for (int i = 0; i < zwi.rows; i++) {
    for (int a = 0; a < zwi.cols; a++) {
      for (int b = 0; b < zwi.cols; b++) {
        if (b + 1 >= zwi.cols)
          break;
        if (zwi.at<double>(a, b) > zwi.at<double>(a, b + 1)) {
          double temp = zwi.at<double>(a, b + 1);
          zwi.at<double>(a, b + 1) = zwi.at<double>(a, b);
          zwi.at<double>(a, b) = temp;
        }
      }
    }
  } // now zwi is a sorted Matrix with the smallest value on the left

  ////Mat zwi = dis.clone();
  ////for (int i = 0; i < zwi.rows; i++) {
  ////  for (int a = i; a < zwi.cols; a++) {
  ////    for (int b = 0; b < zwi.cols; b++) {
  ////      if (b + 1 >= zwi.cols) {
  ////        break;
  ////      }
  ////      if (zwi.at<double>(a, b) > zwi.at<double>(a, b + 1)) {
  ////        double temp = zwi.at<double>(a, b + 1);
  ////        zwi.at<double>(a, b + 1) = zwi.at<double>(a, b);
  ////        zwi.at<double>(a, b) = temp;
  ////      }
  ////    }
  ////  }
  ////} // now zwi is a sorted Matrix with the smallest value on the left
  ////for (int i = 0; i < zwi.cols; i++) {
  ////  cout << zwi.at<double>(0, i) << endl;
  ////}

  for (int i = 0; i < dis.rows; i++) {
    for (int j = 0; j < dis.cols; j++) {
      if (dis.at<double>(i, j) > zwi.at<double>(i, K)) {
        dis.at<double>(i, j) = 10000;
      }
      // cout<<"c=" << c << " ; (i,j)=" << "(" << i << ","  << j << ")" <<
      // endl;
    }
  }
  floyds(dis);

  Mat I = Mat(dis.rows, dis.rows, CV_64F);
  for (int i = 0; i < I.rows; i++) {
    for (int j = 0; j < I.cols; j++) {
      if (i != j) {
        I.at<double>(i, j) = 0;
      } else {
        I.at<double>(i, j) = 1;
      }
    }
  }

  Mat eins = Mat(dis.rows, 1, CV_64F);
  for (int i = 0; i < eins.rows; i++) {
    eins.at<double>(i, 0) = 1.0;
  }

  Mat einsT;
  transpose(eins, einsT);

  Mat C;

  C = I - eins * einsT / dis.rows;
  //	cout << C << endl;
  Mat B = -(C * dis.mul(dis) * C) / 2.0;

  Mat WX, LX, RX;
  SVD::compute(B, WX, LX, RX);

  Mat RXreduce = Mat(B.rows, dim, CV_64F);
  for (int i = 0; i < RXreduce.rows; i++) {
    for (int j = 0; j < RXreduce.cols; j++) {
      RXreduce.at<double>(i, j) = RX.at<double>(j, i);
    }
  }

  //	cout << RX << endl;
  //	cout << RX.size() << endl;

  Mat eigValB;
  Mat eigVecB;
  eigen(B, eigValB, eigVecB);

  Mat eigVecBT;
  transpose(eigVecB, eigVecBT);

  // schon transposed Matrix: 100 x 2
  Mat reduce = Mat(eigVecB.rows, dim, CV_64F);
  for (int i = 0; i < eigVecB.rows; i++) {
    for (int j = 0; j < dim; j++) {
      reduce.at<double>(i, j) = eigVecB.at<double>(j, i);
    }
  }

  /*
  Mat smallMatrix = Mat(dim, dim, CV_64F);
  for (int i = 0; i < dim; i++) {
          for (int j = 0; j < dim; j++) {
                  if (i != j) {
                          smallMatrix.at<double>(i, j) = 0;
                  }
                  else {
                          smallMatrix.at<double>(i, j) =
  sqrt(eigValB.at<double>(i, 0));
                  }
          }
  }

  */

  Mat bigMatrix2 = Mat(B.rows, B.cols, CV_64F);
  for (int i = 0; i < bigMatrix2.rows; i++) {
    for (int j = 0; j < bigMatrix2.cols; j++) {
      if (i != j) {
        bigMatrix2.at<double>(i, j) = 0;
      } else {

        bigMatrix2.at<double>(i, j) = eigValB.at<double>(i, 0);
      }
    }
  }

  Mat sqrtBigMatrix;
  sqrt(bigMatrix2, sqrtBigMatrix);

  Mat erg;
  erg = sqrtBigMatrix * eigVecBT / 10;
  Mat eVreduce = eigVecBT(cv::Rect(0, 0, dim, 100)).clone();
  return eVreduce;
}

void Draw3DManifold2(Mat &dataMatrix, char const *name, int nSamplesI,
                     int nSamplesJ) {
  Mat origImage = Mat(1000, 1000, CV_8UC3);
  origImage.setTo(0.0);
  for (int i = 0; i < nSamplesI; i++) {
    for (int j = 0; j < nSamplesJ; j++) {
      Point p1;
      p1.x = dataMatrix.at<double>(i * nSamplesJ + j, 0) * 50.0 + 500.0 -
             dataMatrix.at<double>(i * nSamplesJ + j, 2) * 10;
      p1.y = dataMatrix.at<double>(i * nSamplesJ + j, 1) * 50.0 + 500.0 -
             dataMatrix.at<double>(i * nSamplesJ + j, 2) * 10;
      circle(origImage, p1, 3, Scalar(255, 255, 255));

      Point p2;
      if (i < nSamplesI - 1) {
        p2.x = dataMatrix.at<double>((i + 1) * nSamplesJ + j, 0) * 50.0 +
               500.0 - dataMatrix.at<double>((i + 1) * nSamplesJ + (j), 2) * 10;
        p2.y = dataMatrix.at<double>((i + 1) * nSamplesJ + j, 1) * 50.0 +
               500.0 - dataMatrix.at<double>((i + 1) * nSamplesJ + (j), 2) * 10;

        line(origImage, p1, p2, Scalar(255, 255, 255), 1, 8);
      }
      if (j < nSamplesJ - 1) {
        p2.x = dataMatrix.at<double>((i)*nSamplesJ + j + 1, 0) * 50.0 + 500.0 -
               dataMatrix.at<double>((i)*nSamplesJ + (j + 1), 2) * 10;
        p2.y = dataMatrix.at<double>((i)*nSamplesJ + j + 1, 1) * 50.0 + 500.0 -
               dataMatrix.at<double>((i)*nSamplesJ + (j + 1), 2) * 10;

        line(origImage, p1, p2, Scalar(255, 255, 255), 1, 8);
      }
    }
  }

  namedWindow(name, WINDOW_AUTOSIZE);
  imshow(name, origImage);
}

void Draw2DManifold2(Mat &dataMatrix, char const *name, int nSamplesI,
                     int nSamplesJ) {
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
        p2.x =
            dataMatrix.at<double>((i + 1) * nSamplesJ + j, 0) * 1000.0 + 500.0;
        p2.y =
            dataMatrix.at<double>((i + 1) * nSamplesJ + j, 1) * 1000.0 + 500.0;
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
