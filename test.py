keyboard = {
    2: ["a", "b", "c"],
    3: ["d", "e", "f"],
    4: ["g", "h", "i"],
    5: ["j", "k", "l"],
    6: ["m", "n", "o"],
    7: ["p", "q", "r", "s"],
    8: ["t", "u", "v"],
    9: ["w", "x", "y", "z"]
}

list1 = keyboard.get(2)
list2 = keyboard.get(3)

result = []
for letter1 in list1:
    for letter2 in list2:
        result.append(letter1+letter2)

print(result)