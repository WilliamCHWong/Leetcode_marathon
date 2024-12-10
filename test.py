from typing import List

def combine(n: int, k: int) -> List[List[int]]:
    result = []

    def genPath(start: int, k: int, path: List[int]):
        print("Loop: ", "start: " ,start, " k: ", k, "path: ", path)
        if len(path) == k:
            result.append(path)
            return

        for i in range(start + 1, n + 1):
            print("i: ", i)
            genPath(i, k, path.append(i))
    
    genPath(1, k, [])

    return result

print(combine(4, 2))