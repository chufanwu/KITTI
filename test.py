# test.py
def returnNone():
    return None 

try:
    a, b = returnNone()
except:
    print "None!"