using NPTVGC
using Test


@testset "NPTVGC.jl" begin
    # Write your tests here.

    @test NPTVGC.weights!((1, 1), 1.0, "e", "smoothing") == [1]

    # create struct object
    test_obj = NPTVGC.NPTVGC_test([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    @test  test_obj.x == [1.0, 2.0, 3.0]

    @test NPTVGC.estimate_tv_tstats(test_obj, 1) == nothing
    
    x = [3.6133192031340955e-001,
    3.6133192031340955e-001,
   -2.5330724283309947e-001,
   -2.0395372587599631e-001,
   -1.7767130682598453e-001,
    1.0788311445951138e+000,
   -3.1048184957868985e-001,
    2.1841971113808270e-001,
    4.4729835245848670e-002,
   -2.6207015832699887e-001,
    5.4919370794996225e-001,
   -1.3516527058006282e+000,
   -3.2769293772270347e-001,
   -9.0317632922186908e-001,
   -4.3916046154673127e-001,
    1.1626042528418563e+000,
    7.0185164227227681e-001,
   -2.5924692582754272e-001,
    5.5929764407028681e-002,
   -5.8009665826208479e-001]

   y = [ -8.1498728210450277e-001,
   1.2127666343457797e+000,
   5.9542188227248116e-001,
   2.9923270146589437e-001,
  -9.0494552633926806e-001,
   5.0050272294337361e-001,
  -6.1577287341115616e-001,
  -2.6912777714557751e-001,
  -3.1162575707270468e-001,
  -7.0020040533146211e-001,
   1.2884322382325897e-001,
  -9.4056164797012509e-002,
  -5.8671052787535771e-001,
   1.1471499767265225e-001,
  -6.7813704635008143e-001,
  -7.3584799621415620e-003,
   5.3950752591383233e-001,
  -1.9078234180270853e-001,
   5.3751732617422654e-001,
   6.6844098034554866e-001]

   x = NPTVGC.normalise(x)
   y = NPTVGC.normalise(y)

   test_obj = NPTVGC.NPTVGC_test(y, x)
   NPTVGC.estimate_tv_tstats(test_obj, 1)
   # test stat equal to 0.13 with 0.01 uncertainty
   @test test_obj.Tstats[1] ≈ 0.13 atol=0.01

   @testset "total_likelihoods! tests" begin
    # Define your test inputs here
    x = randn(100)
    y = randn(100)
    x = NPTVGC.normalise(x)
    y = NPTVGC.normalise(y)
    N = 100
    m = 1
    mmax = 1
    ϵ = 0.5
    weights = rand(100)

    # Test that total_likelihoods! does not throw an error
    @test try
        liks = NPTVGC.total_likelihoods!(x, y, N, m, mmax, ϵ, weights)
        println(liks)
        true
    catch e
        println("Error: ", e)
        false
    end
    end

@testset "lik_cv tests" begin
    # Define your test inputs here
    x = randn(100)
    y = randn(100)
    x = NPTVGC.normalise(x)
    y = NPTVGC.normalise(y)
    obj = NPTVGC.NPTVGC_test(x,y)
    # Test that lik_cv does not throw an error
    @test try
        lik = NPTVGC.lik_cv(obj,  (1.0, 0.5))
        println(lik)
        true
    catch e
        println("Error: ", e)
        false
    end
end

@testset "estimate_LDE tests" begin
    # Define your test inputs here
    x = randn(100)
    y = randn(100)
    x = NPTVGC.normalise(x)
    y = NPTVGC.normalise(y)
    obj = NPTVGC.NPTVGC_test(x, y)  # Replace with the actual type of your object
    obj.filter = "smoothing"
    obj.max_iter = 2
    obj.max_iter_outer = 2
    # Set other properties of obj as needed

    # Test that estimate_LDE does not throw an error
    @test try
        NPTVGC.estimate_LDE(obj)
        true
    catch e
        println("Error: ", e)
        false
    end
end
end

