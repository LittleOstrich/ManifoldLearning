#include "floydmarshall.h"

using namespace cv;
using namespace std;
// Solves the all-pairs shortest path problem using Floyd Warshall algorithm
Mat floydWarshall(Mat m)
{
    cout << "floydWarshall started" << endl;

    /* dist[][] will be the output matrix that will finally have the shortest
  distances between every pair of vertices */
    int i, j, k, len;
    len = m.rows;
    Mat ret = Mat(len, len, CV_64F);

    /* Initialize the solution matrix same as input graph matrix. Or
  we can say the initial values of shortest distances are based
  on shortest paths considering no intermediate vertex. */
    for (i = 0; i < m.rows; i++)
        for (j = 0; j < m.cols; j++)
            ret.at<double>(i, j) = m.at<double>(i, j);

    /* Add all vertices one by one to the set of intermediate vertices.
  ---> Before start of a iteration, we have shortest distances between all
  pairs of vertices such that the shortest distances consider only the
  vertices in set {0, 1, 2, .. k-1} as intermediate vertices.
  ----> After the end of a iteration, vertex no. k is added to the set of
  intermediate vertices and the set becomes {0, 1, 2, .. k} */

    for (k = 0; k < len; k++) {
        // Pick all vertices as source one by one
        for (i = 0; i < len; i++) {
            // Pick all vertices as destination for the
            // above picked source
            for (j = 0; j < len; j++) {
                // If vertex k is on the shortest path from
                // i to j, then update the value of dist[i][j]
                if (ret.at<double>(i, k) + ret.at<double>(k, j) < ret.at<double>(i, j))
                    ret.at<double>(i, j) = ret.at<double>(i, k) + ret.at<double>(k, j);
            }
        }
    }

    cout << "floydWarshall finished" << endl;

    // Print the shortest distance matrix
    return ret;
}
