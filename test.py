class test():
    def __init__(self,a,b,):
        print('Initialized')
        self.a = a
        self.b = b
    def __call__(self, *args, **kwargs):
        # for arg in args:
        #     print('Called', arg)
        # for key, value in kwargs.items():
        #     print("{} = {}".format(key, value))
        # print('Called Last')
        print(self.a*self.b)


x = test(10,20)
# __init__ Object is initialized

# x()
#__call__ Instance is called as function, prints Called Last


# Now them individually and see the difference
# x('1','2','3','4','5','6','7','8','9','0','-','=',kw = 'kw',arg = 'arg')