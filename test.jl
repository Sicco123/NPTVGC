using NPTVGC
x = randn(20)
y = randn(20)
x = NPTVGC.normalise(x)
y = NPTVGC.normalise(y)
obj = NPTVGC.NPTVGC_test(x, y)  # Replace with the actual type of your object
obj.filter = "smoothing"
obj.max_iter = 100
obj.max_iter_outer = 100
# Set other properties of obj as needed

# Test that estimate_LDE does not throw an error
NPTVGC.estimate_LDE(obj)
    