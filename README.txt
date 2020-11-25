Input: a video recorded while car driving
Output: The allowed speed limit

The process is divided into 4 steps:
1. extraction of the street sign (01.12.2020)
2. edit the extracted sign (resize, kill background...) (15.12.2020)
3. classification of the extracted sign (maybe with comparing pixel values or with a neural network) (06.01.2020)
4. - the program should work in real time (maybe 1 image per second - should be enough)
   - also it should work at night