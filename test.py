def countAndSay(n: int) -> str:
    def convert(RLE: str) -> str:
        result = ""
        dict = {}
        
        # Iterate through RLE
        for i in range(len(RLE)):
            if RLE[i] in dict.keys():
                dict[RLE[i]] += 1
            else:
                dict[RLE[i]] = 1

        # Append the last group
        for key in dict:
            result = result + str(dict[key]) + str(key)

        return result

    # Base case
    if n == 1:
        return "1"
    
    # Recursive call
    return convert(countAndSay(n - 1))


for i in range(5):
    print(countAndSay(i))