z = 1
def gen2(y):
    x = y+z
    def gen3():      
        return x**2
    return gen3,abs
        
      
print(gen2(1))

