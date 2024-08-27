## NPTVGC - Nonparametric Time-Varying Granger Causality

This framework can be used to test for Granger causality nonparametrically in time-varying conditions. 
We use exponentially weighted moving average density estimators. The test we use is the Diks Panchenko test [[1]](#1).


## Basic example
```
using NPTVGC, Random

# Simulate data
N = 1000
y = randn(N)
x = randn(N)

# Define test
test_struct = NPTVGC.NPTVGC_test(y, x)
test_struct.γ = 0.996
test_struct.ϵ = 1.0

# Calculate time-varying test statistics and p-values   
NPTVGC.estimate_tv_tstats(test_struct, 1)
p_vals = test_struct.pvals                 
```


## References
<a id="1">[1]</a> 
Diks, C. and Panchenko, V. (2006). A new statistic and practical guidelines for nonparametric
granger causality testing. Journal of Economic Dynamics and Control, 30(9-10):1647–1669.
