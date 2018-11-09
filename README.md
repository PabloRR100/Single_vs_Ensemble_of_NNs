# Single_vs_Ensemble_of_NNs

The aim of this project is to understand whether it could be a better approach to go for ensemble learning instead of going deeper in Deep Learning.

There are some dashboards with the results comparing different ResNet architectures on CIFAR10 with an ensemble of smaller ResNets that have the same (or less) paramaters in total.
This way, we can say that for a fixed network capacity, splitting the parameters into an ensemble of models is more efficient than gathering them into a single deeper structure.

In this [separated repository][1] can be found Dashboards to explore the results obtained. 
Some of them already deployed (requires username and password): 

- [ResNet56 vs 3 ResNet20][2]
- [ResNet110 vs 6 ResNet20][3]

<embed src="https://resnet110.herokuapp.com/">



[1]: https://github.com/PabloRR100/Single_vs_Ensemble_Dashboards
[2]: https://resnet56.herokuapp.com/
[3]: https://resnet110.herokuapp.com/
