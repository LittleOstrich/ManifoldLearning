#include "pca.h"

using namespace cv;
using namespace std;

Mat reducePCA(Mat& dataMatrix, unsigned int dim)
{
    std::vector<double> raw_summed_data;
    for (int i = 0; i < dim; i++) {
        raw_summed_data.push_back(0);
    }
    for (int i = 0; i < dataMatrix.rows; i++) {
        for (int k = 0; k < dim; k++) {
            raw_summed_data.at(k) += dataMatrix.at<double>(i, k) * dataMatrix.at<double>(i, k);
        }
    }

    for (int i = 0; i < dim; i++) {
        raw_summed_data.at(i) = sqrt(raw_summed_data.at(i));
    }

    for (int i = 0; i < dataMatrix.rows; i++) {
        for (int k = 0; k < dim; k++) {
            dataMatrix.at<double>(i, k) = dataMatrix.at<double>(i, k) / raw_summed_data.at(k);
        }
    }
    Mat m = Mat(dim, dim, CV_64F);
    Mat vec = Mat(dim, 1, CV_64F);
    for (int i = 0; i < dataMatrix.rows; i++) {
        Mat temp = Mat(dim, dim, CV_64F);
        for (int j = 0; j < dim; j++) {
            vec.at<double>(j, 0) = dataMatrix.at<double>(i, j);
        }
        mulTransposed(vec.t(), temp, dataMatrix.rows);
        m = m + temp;
    }

    Mat A, w, u, vt;
    SVD::compute(m, w, u, vt);

    Mat u2 = Mat::eye(dataMatrix.cols, dataMatrix.cols, CV_64F);
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            u2.at<double>(i, j) = u.at<double>(i, j);
        }
    }

    for (int i = 0; i < dataMatrix.rows; i++) {
        Mat temp = Mat(dataMatrix.cols, 1, CV_64F);
        for (int j = 0; j < dim; j++) {
            temp.at<double>(j, 0) = dataMatrix.at<double>(i, j);
        }
        temp = u2 * temp;
        for (int j = 0; j < dim; j++) {
            dataMatrix.at<double>(i, j) = temp.at<double>(j, 0);
        }
    }
    return dataMatrix;
}
