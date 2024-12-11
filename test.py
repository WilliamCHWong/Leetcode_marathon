from typing import List

board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
word = "SEE"
ptr = 0
m, n, w = len(board), len(board[0]), len(word)

def check(i: int, j: int, ptr: int):
    up = down = left = right = False

    if ptr >= w:
        return True
    
    # Up
    if i > 0 and board[i - 1][j] == word[ptr]:
        up = check(i - 1, j, ptr + 1)
    # Down
    if i < m - 1 and board[i + 1][j] == word[ptr]:
        down = check(i + 1, j, ptr + 1)
    # Right
    if j < n - 1 and board[i][j + 1] == word[ptr]:
        right = check(i, j + 1, ptr + 1)
    # Left
    if j > 0 and board[i][j - 1] == word[ptr]:
        left = check(i, j - 1, ptr + 1)
    
    return up or down or right or left

print(check(1, 3, 1))