# FTSwishPlus
FTSwish with mean shifting added to increase performance.  Original FTSwish is from this paper:
https://arxiv.org/abs/1812.06247



# FTSwishPlus = This is FTSwish, but with a mean shift added to drive the mean to zero on a base kaiming init'ed tensor.  

This is to help ensure a more stable starting setup and the concept developed by FastAI/Jeremy Howard.
By adding mean shift of - .1, performance improved so that it was able to win an initial comparison of Relu, General Relu, LiSHT for image classification.

<img src='images/final-results2-imagenette-res18.jpg' width=80% height=80% />

More detailed article with testing results:  https://medium.com/@lessw/comparison-of-activation-functions-for-deep-learning-initial-winner-ftswish-f13e2621847
