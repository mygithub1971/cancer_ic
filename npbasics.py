import numpy as np

mylist = [1,2,3]

print(type(np.array(mylist)))

arr = np.array(mylist)

arr1 = np.arange(0,10) # start, end

z = np.zeros((3,2)) # size

o = np.ones((4,2)) # size

l = np.linspace(0,10,11) # start, end, total number of elements

#np.random.seed(0)

r = np.random.randint(0,10,(4,4)) # start, end, size

rn = np.random.normal(0,1,(3,2)) # loc, scale, size

rnr = rn.reshape(1,-1) # -1 for auto

print(rnr)
print(rnr.max())
print(rnr.argmax())

print("-"*30)
print(r)
print(r[1,1])
print(r[:,0])
print(r[1:3,3:4])
print(r>5)
print(r[r>5])
