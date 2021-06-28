def fact(n):
    global result
    if n > 1:
        result *= n
        fact(n - 1)
    else:
        print(result)


n = int(input())
result = 1
fact(n)
