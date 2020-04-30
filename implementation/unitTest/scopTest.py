def function1(a):
    b = a + 10
    c = (2, 4)

    def function2():
        for i in c:
            print(b+i)
    f = function2
    return f

a = function1(5)
a()
