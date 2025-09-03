# purpose

Python script to produce diagrams for a report.
input are measurements of temperatures.

Statistical descriptions of a set of measurements is stored in class sdatasets.
the mean, std, stderr is in the subclasses bb/owall/iwalls/iwall_left/iwall_right.
For the programmer in this code the instance of the class is called stdata.
Each stdata dataset has +/-/** operations. always resulting in np.ndarray results.

Data types: the data is stored in lists of np.float64 types.

## usage of sdataset class:

stdata.append(bb_temp_K[start:end],
                  tleft_in_K[start:end], tright_in_K[start:end], tleft_out_K[start:end])
                  
temp4_diff = stdata.bb.avg ** 4 - stdata.iwalls.avg ** 4 # ** gives np.array

## other
### hint:

show_plot varianle. 
the figures are saved but the display can be suppressed by entering 0 in the string
