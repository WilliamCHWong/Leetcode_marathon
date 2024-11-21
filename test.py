from typing import List

def rotate(matrix: List[List[int]]) -> None:
    n = len(matrix)

    for i in range(n // 2):
        print("i: ", i)
        for j in range(i, n - 1 - i):
            print("j: ", j)
            temp = matrix[i][j]
            matrix[i][j] = matrix[n-1-j][i]
            matrix[n-1-j][i] = matrix[n-1-i][n-1-j]
            matrix[n-1-i][n-1-j] = matrix[j][n-1-i]
            matrix[j][n-1-i] = temp
    
matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
rotate(matrix)
print(matrix)