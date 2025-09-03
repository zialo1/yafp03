python script to produce diagrams for a report.
input are measurements of temperatures.

statistical descriptions of a set of measurements is stored in class sdatasets.
the mean, std, stderr is in the subclasses bb/owall/iwalls/iwall_left/iwall_right
in the code the instance of the class is called stdata.
each stdata dataset has +/-/** operations. always resulting in np.ndarray results.
data types: the data is stored in lists of np.float64 types.

usage:
temp4_diff = stdata.bb.avg ** 4 - stdata.iwalls.avg ** 4 # ** gives np.array

