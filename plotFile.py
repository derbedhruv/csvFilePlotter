# this is for reading in csv 

# first, the import statemtnts
import numpy
import matplotlib.pyplot as plt

# taken from the old thing, just a definition of the file and headerlines
fileName = "I:\NewFile0.csv"		# enter the file name, bitch
headLines = 3		

# we first acquire the data. In this case the extra 'z' is given because there was a 2 channel oscilloscope, one channel of which was OFF.
t, chanA, chanB = numpy.genfromtxt(fileName, skip_header=headLines, unpack=True, delimiter=',')

# now we plot the files
plt.plot(t, chanA, 'k')
plt.axis([-6,6,0,1])
plt.show()		# don't forget this!
