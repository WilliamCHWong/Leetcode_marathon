max_count = 0
for i in range(len(word2)):
    source, dest, count = 0, i, 0
    while source < len(word1) and dest < len(word2):
        if word1[source] == word2[dest]:
            dest += 1
            count += 1
        source += 1
    max_count = max(max_count, count)
