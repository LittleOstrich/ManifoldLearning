#include "isomap.h"

using namespace cv;
using namespace std;

Mat reduceIsomap(Mat& dataMatrix, unsigned int dim, int nSamplesI,
    int nSamplesJ)
{
    const int K = 10;
    Mat m = Mat(nSamplesI * nSamplesJ, nSamplesI * nSamplesJ, CV_64F);
    m = Scalar(100);
    vector<P> nodes;

    cout << "Formatting data" << endl;
    for (int i = 0; i < dataMatrix.rows; i++) {
        vector<double> features;
        for (int j = 0; j < 3; j++) {
            double fV = dataMatrix.at<double>(i, j);
            features.push_back(fV);
        }
        P p = P(features);
        p.index = i;
        nodes.push_back(p);
    }

    cout << "Data is formatted" << endl;

    // Finding k-nearest neigbours, setting up matrix
    for (int i = 0; i < nodes.size(); i++) {
        vector<P2> nb = findNeighbours(nodes, nodes.at(i), K, 3);
        for (int k = 0; k < K + 1; k++) {
            m.at<double>(i, nb.at(k).ind) = nb.at(k).dist;
        }
    }

    Mat m2 = floydWarshall(m);

    // squared distance matrix
    Mat S = Mat(m2.rows, m2.rows, CV_64F);
    for (int i = 0; i < m2.rows; i++) {
        for (int j = 0; j < m2.rows; j++) {
            S.at<double>(i, j) = m2.at<double>(i, j) * m2.at<double>(i, j);
        }
    }

    // centering matrix
    Mat H = Mat::eye(m2.rows, m2.rows, CV_64F);
    for (int i = 0; i < m2.rows; i++) {
        for (int j = 0; j < m2.rows; j++) {
            H.at<double>(i, j) = H.at<double>(i, j) - 1.0 / (nSamplesI * nSamplesJ);
        }
    }

    m2 = -0.5 * H * S * H;

    Mat eVal, V;
    eigen(m2, eVal, V);

    Mat vT;
    transpose(V, vT);

    Mat eVreduce = vT(cv::Rect(0, 0, dim, nSamplesI * nSamplesJ)).clone();
    return eVreduce;
}
