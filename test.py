from typing import List

def combine(n: int, k: int) -> List[List[int]]:
    result = []

    def genPath(start: int, k: int, path: List[int]):
        if len(path) == k:
            result.append(path[:])
            return

        for i in range(start, n + 1):
            path.append(i)
            genPath(i + 1, k, path)
            path.pop()
    
    genPath(1, k, [])

    return result

print(combine(5, 3))