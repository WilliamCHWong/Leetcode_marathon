def bitwisePlus (x: int, y: int)->int:
    while y != 0:
        carry = (x & y) << 1
        remain = x ^ y

        x = remain
        y = carry

    return x

print(bitwisePlus(3, 5))