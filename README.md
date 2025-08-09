This repository contains my attempts to work through various adverserial learning algorithms on the MobileNetV2 image classification model.

Credit to Google for their Fast Gradient Sign Method tutorial https://www.tensorflow.org/tutorials/generative/adversarial_fgsm

Credit to Jan Krepl for his tutorial on Projected Gradient Descent tutorial https://www.youtube.com/watch?v=5lFiZTSsp40&ab_channel=mildlyoverfitted

Credit to Yap Jit Feng for his presentation on DeepFool https://www.youtube.com/watch?v=9D0A_HBFF24&ab_channel=JitYap

In each method attempted to write or rewrite using functionality using pytorch. I explored the effects of various hyper-parameters and analyze their effects on the confidence of the labeled class, the confidence of the most confident class and the confidence of the 5th most confident class since top 5 accuracy is often a tracked metric of image classification models.
