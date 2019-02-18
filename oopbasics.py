#class Person(object):
#    def __init__(self, age, height): # constructor
#        self.age = age
#        self.height = height
#        self.name = ""
#    
#    def set_name(self, name):
#        self.name = name
#        
#class Student(Person):  
#    def __init__(self, age, height): # derived class constructor
#        self.score = None
#        Person.age = age
#        Person.height = height
#        Person.name = ""
#            
#    def set_score(self, score):
#        self.score = score
#        
#
#student = Student(16,160)
#print(student.age)
#print(student.name)
#student.set_name("Sarah")
#print(student.name)
#student.set_score(96)
#print(student.score)


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Circle(object):
    def __init__(self, color, radius, center):
        self.color = color
        self.radius = radius
        self.center = center
        
    def isinside(self, point): # return True if the point is inside the circle
        dist_sq = (point.x - self.center.x)**2 + (point.y - self.center.y)**2
        if dist_sq < self.radius*self.radius:
            return True
        return False
    
    
    
class Square(object):
    def __init__(self, llc, side_length):
        self.llc = llc
        self.side_length = side_length
        self.luc = Point(0.0,0.0)
        #self.ruc = None
        #self.rlc = None
        
    def unpack_points(self):
        self.luc.x = self.llc.x
        self.luc.y = self.llc.y + self.side_length
        
    
def is_square_in_circle(square, circle):
    square.unpack_points()
    if circle.isinside(square.llc) and circle.isinside(square.luc):
        return True
    return False
    
    
# Write a class of a Square on a 2D plane (give it some attributes to define the size and position)
# Write a method def isincircle(self, circle<--object): return True if the square is completely inside the circle
# make an instance of the Square class and test your method
        
center = Point(0.0, 0.0)
circle = Circle("red", 1.0, center)
square = Square(Point(0.0, 0.0), 2.0)
#print(circle.isinside(circle.center))
#print(circle.isinside(Point(1.0,2.0)))
print(is_square_in_circle(square, circle))
        