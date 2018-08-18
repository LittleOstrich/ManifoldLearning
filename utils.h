#pragma once
#include "floydmarshall.h"
#include "math.h"
#include "pca.h"
#include <iostream>
#include <opencv2\opencv.hpp>

struct P {
    std::vector<double> features;
    std::vector<P> neighbours;
    int index;
    P(const std::vector<double> features) { this->features = features; }
    P(int index, std::vector<P> neighbours)
    {
        this->index = index;
        this->neighbours = neighbours;
    }
};

struct P2 {
    int ind;
    double dist;
    P2(int ind, double dist)
    {
        this->ind = ind;
        this->dist = dist;
    }
};

std::vector<P2> findNeighbours(std::vector<P> nodes, P node, int K, int dim);
double computeDist(P n1, P n2, int dim);
