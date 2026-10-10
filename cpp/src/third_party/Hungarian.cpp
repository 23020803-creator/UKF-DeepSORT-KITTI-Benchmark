#include "third_party/Hungarian.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

namespace ukf_deepsort {

double HungarianAlgorithm::Solve(const std::vector<std::vector<double>>& costMatrix, std::vector<int>& assignment) {
    int nRows = static_cast<int>(costMatrix.size());
    if (nRows == 0) {
        assignment.clear();
        return 0.0;
    }
    int nCols = static_cast<int>(costMatrix[0].size());
    if (nCols == 0) {
        assignment.assign(nRows, -1);
        return 0.0;
    }

    std::vector<double> distMatrix(nRows * nCols);
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            distMatrix[i + nRows * j] = costMatrix[i][j];
        }
    }

    assignment.assign(nRows, -1);
    double cost = 0.0;

    assignmentoptimal(assignment.data(), &cost, distMatrix.data(), nRows, nCols);
    return cost;
}

void HungarianAlgorithm::assignmentoptimal(int* assignment, double* cost, double* distMatrixIn, int nOfRows, int nOfColumns) {
    double* distMatrix = new double[nOfRows * nOfColumns];
    bool* starMatrix = new bool[nOfRows * nOfColumns]();
    bool* primeMatrix = new bool[nOfRows * nOfColumns]();
    bool* newStarMatrix = new bool[nOfRows * nOfColumns]();
    bool* coveredColumns = new bool[nOfColumns]();
    bool* coveredRows = new bool[nOfRows]();

    for (int i = 0; i < nOfRows * nOfColumns; ++i) {
        distMatrix[i] = distMatrixIn[i];
    }

    int minDim = std::min(nOfRows, nOfColumns);

    // Bước 1: Trừ giá trị nhỏ nhất của mỗi hàng
    for (int row = 0; row < nOfRows; ++row) {
        double minVal = distMatrix[row];
        for (int col = 1; col < nOfColumns; ++col) {
            if (distMatrix[row + nOfRows * col] < minVal) {
                minVal = distMatrix[row + nOfRows * col];
            }
        }
        for (int col = 0; col < nOfColumns; ++col) {
            distMatrix[row + nOfRows * col] -= minVal;
        }
    }

    // Bước 2: Tìm các số 0 và đánh dấu sao (star)
    for (int row = 0; row < nOfRows; ++row) {
        for (int col = 0; col < nOfColumns; ++col) {
            if (std::abs(distMatrix[row + nOfRows * col]) < 1e-9) {
                if (!coveredColumns[col]) {
                    starMatrix[row + nOfRows * col] = true;
                    coveredColumns[col] = true;
                    break;
                }
            }
        }
    }

    for (int col = 0; col < nOfColumns; ++col) {
        coveredColumns[col] = false;
    }

    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

    computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);

    delete[] distMatrix;
    delete[] starMatrix;
    delete[] primeMatrix;
    delete[] newStarMatrix;
    delete[] coveredColumns;
    delete[] coveredRows;
}

void HungarianAlgorithm::buildassignmentvector(int* assignment, bool* starMatrix, int nOfRows, int nOfColumns) {
    for (int row = 0; row < nOfRows; ++row) {
        assignment[row] = -1;
        for (int col = 0; col < nOfColumns; ++col) {
            if (starMatrix[row + nOfRows * col]) {
                assignment[row] = col;
                break;
            }
        }
    }
}

void HungarianAlgorithm::computeassignmentcost(int* assignment, double* cost, double* distMatrixIn, int nOfRows) {
    *cost = 0.0;
    for (int row = 0; row < nOfRows; ++row) {
        int col = assignment[row];
        if (col >= 0) {
            *cost += distMatrixIn[row + nOfRows * col];
        }
    }
}

void HungarianAlgorithm::step2a(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim) {
    for (int col = 0; col < nOfColumns; ++col) {
        coveredColumns[col] = false;
        for (int row = 0; row < nOfRows; ++row) {
            if (starMatrix[row + nOfRows * col]) {
                coveredColumns[col] = true;
                break;
            }
        }
    }
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step2b(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim) {
    int count = 0;
    for (int col = 0; col < nOfColumns; ++col) {
        if (coveredColumns[col]) {
            count++;
        }
    }

    if (count >= minDim) {
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    } else {
        step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }
}

void HungarianAlgorithm::step3(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim) {
    bool zerosFound = true;
    while (zerosFound) {
        zerosFound = false;
        for (int col = 0; col < nOfColumns; ++col) {
            if (!coveredColumns[col]) {
                for (int row = 0; row < nOfRows; ++row) {
                    if ((!coveredRows[row]) && (std::abs(distMatrix[row + nOfRows * col]) < 1e-9)) {
                        primeMatrix[row + nOfRows * col] = true;
                        int starCol = -1;
                        for (int c = 0; c < nOfColumns; ++c) {
                            if (starMatrix[row + nOfRows * c]) {
                                starCol = c;
                                break;
                            }
                        }
                        if (starCol >= 0) {
                            coveredRows[row] = true;
                            coveredColumns[starCol] = false;
                            zerosFound = true;
                            break;
                        } else {
                            step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                            return;
                        }
                    }
                }
            }
        }
    }
    step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step4(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col) {
    for (int i = 0; i < nOfRows * nOfColumns; ++i) {
        newStarMatrix[i] = starMatrix[i];
    }
    newStarMatrix[row + nOfRows * col] = true;

    int starCol = col;
    int starRow = -1;
    for (int r = 0; r < nOfRows; ++r) {
        if (starMatrix[r + nOfRows * starCol]) {
            starRow = r;
            break;
        }
    }

    while (starRow >= 0) {
        newStarMatrix[starRow + nOfRows * starCol] = false;
        int primeRow = starRow;
        int primeCol = -1;
        for (int c = 0; c < nOfColumns; ++c) {
            if (primeMatrix[primeRow + nOfRows * c]) {
                primeCol = c;
                break;
            }
        }
        newStarMatrix[primeRow + nOfRows * primeCol] = true;
        starCol = primeCol;
        starRow = -1;
        for (int r = 0; r < nOfRows; ++r) {
            if (starMatrix[r + nOfRows * starCol]) {
                starRow = r;
                break;
            }
        }
    }

    for (int i = 0; i < nOfRows * nOfColumns; ++i) {
        primeMatrix[i] = false;
        starMatrix[i] = newStarMatrix[i];
    }
    for (int r = 0; r < nOfRows; ++r) {
        coveredRows[r] = false;
    }
    step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step5(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim) {
    double h = std::numeric_limits<double>::max();
    for (int row = 0; row < nOfRows; ++row) {
        if (!coveredRows[row]) {
            for (int col = 0; col < nOfColumns; ++col) {
                if (!coveredColumns[col]) {
                    if (distMatrix[row + nOfRows * col] < h) {
                        h = distMatrix[row + nOfRows * col];
                    }
                }
            }
        }
    }

    for (int row = 0; row < nOfRows; ++row) {
        if (coveredRows[row]) {
            for (int col = 0; col < nOfColumns; ++col) {
                distMatrix[row + nOfRows * col] += h;
            }
        }
    }
    for (int col = 0; col < nOfColumns; ++col) {
        if (!coveredColumns[col]) {
            for (int row = 0; row < nOfRows; ++row) {
                distMatrix[row + nOfRows * col] -= h;
            }
        }
    }
    step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

} // namespace ukf_deepsort
