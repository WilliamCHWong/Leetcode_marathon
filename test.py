n = 3
results = [[0 for _ in range(n)] for _ in range(n)]
top, bottom, left, right = 0, n - 1, 0, n - 1
content = 1

for i in range (0, n):
    results[0][i] = content
    content += 1

print(results)