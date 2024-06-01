class TestBase:
    age = 10


class Test1(TestBase):
    def __init__(self):
        super().__init__()


class Test2(TestBase):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    test1 = Test1()
    test2 = Test2()
    print(TestBase.age)
    TestBase.age = 20
    print(test1.age)
    test1.age = 30
    print(TestBase.age)
