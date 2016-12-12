import matplotlib.pyplot as plt
import numpy as np

plt.plot([0.5,2],[1.5,1],'r+',markersize=15.0,markeredgewidth=2.0)
plt.plot([1.5,1.5],[0.5,2],'bo',markersize=3.0,markeredgewidth=5.0)
plt.plot([0.75 for i in range(4)],color='g')
plt.plot([1.75 for i in range(4)],color='g')
plt.axvline(x=1.25,color='g')

# Fill between
x = np.arange(0,3.5,0.1)
y1 = 0.75
y2 = 1.75

plt.fill_between(x,y1,y2,color='grey',alpha='0.5')
plt.fill_between(np.arange(0,1.25,0.001),y2,2.5,color='grey',alpha='0.5')
plt.fill_between(np.arange(0,1.25,0.001),0.0,y1,color='grey',alpha='0.5')

plt.show()