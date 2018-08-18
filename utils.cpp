#include "utils.h"

using namespace std;

bool sortP2(P2 p1, P2 p2)
{
    return p1.dist < p2.dist;
}

double computeDist(P n1, P n2, int dim)
{
    double d = 0;
    double t;
    for (int i = 0; i < dim; i++) {
        t = n1.features.at(i) - n2.features.at(i);
        d = d + t * t;
    }
    return sqrt(d);
}

vector<P2> findNeighbours(vector<P> nodes, P node, int K, int dim)
{
    vector<P2> temp;
    vector<P2> ret;
    for (int i = 0; i < nodes.size(); i++) {
        temp.push_back(P2(i, computeDist(nodes.at(i), node, dim)));
    }

    sort(
        temp.begin(), temp.end(), [](P2 p1, P2 p2) { return p1.dist < p2.dist; });
    for (int i = 0; i < K + 1; i++) {

        P2 el = temp.at(i);
        // cout << el.dist << " " << el.ind << endl;
        ret.push_back(el);
    }
    // cout << "-----" << endl;
    return ret;
}
