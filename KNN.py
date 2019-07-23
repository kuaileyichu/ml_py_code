
import numpy as np
X_raw = [[13.23,  5.64],
       [13.2 ,  4.38],
       [13.16,  4.68],
       [13.37,  4.8 ],
       [13.24,  4.32],
       [12.07,  2.76],
       [12.43,  3.94],
       [11.79,  3.  ],
       [12.37,  2.12],
       [12.04,  2.6 ]]
y_raw = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

X = np.array(X_raw)
y = np.array(y_raw)
x_test= np.array([12.08,  3.3 ])

#KNN
import math
distances = [math.sqrt(np.sum(np.square(x_test-p))) for p in X]
print(distances)

sort = np.argsort(distances)
# sort = np.array(distances).argsort()
print(sort)

from collections import Counter
K = 3
topK = [y[i] for i in sort[:K]]
print(topK)
votes = Counter(topK)
cls = votes.most_common()[0][0]
'''List the n most common elements and their counts from the most
        common to the least.  If n is None, then list all element counts.

        >>> Counter('abcdeabcdabcaba').most_common(3)
        [('a', 5), ('b', 4), ('c', 3)]
        '''
print(cls)


