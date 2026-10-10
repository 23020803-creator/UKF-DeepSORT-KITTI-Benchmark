#pragma once

#include <vector>

namespace ukf_deepsort {

/**
 * @brief Hiện thực hóa thuật toán Hungarian (Kuhn-Munkres) giải bài toán phân việc đồ thị hai phía tối ưu O(N^3).
 */
class HungarianAlgorithm {
public:
    static double Solve(const std::vector<std::vector<double>>& costMatrix, std::vector<int>& assignment);

private:
    static void assignmentoptimal(int* assignment, double* cost, double* distMatrixIn, int nOfRows, int nOfColumns);
    static void buildassignmentvector(int* assignment, bool* starMatrix, int nOfRows, int nOfColumns);
    static void computeassignmentcost(int* assignment, double* cost, double* distMatrixIn, int nOfRows);
    static void step2a(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
    static void step2b(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
    static void step3(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
    static void step4(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
    static void step5(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
};

} // namespace ukf_deepsort
