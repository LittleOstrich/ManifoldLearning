#pragma once

#include "utils.h"
#include <iostream>
#include <map>
#include <opencv/ml.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat reduceIsomap(cv::Mat& dataMatrix, unsigned int dim, int nSamplesI,
    int nSamplesJ);
