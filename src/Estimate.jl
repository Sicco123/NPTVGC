function estimate_LDE(obj, fix_γ = NaN, multiple_bandwidth = false)
   
    if obj.filter == "filtering" && isnan(fix_γ) && !multiple_bandwidth 
        if obj.method == "sequential"
            p0 = [0.95, 0.55 * std(obj.x[1:obj.offset2-1])]
            p0[1] = inverse_sigmoid_map(p0[1], obj.a_γ, obj.b_γ)
            p0[2] = inverse_sigmoid_map(p0[2], obj.a_ϵ, obj.b_ϵ)

            p = fill(NaN, (2, obj.ssize))

            for t in obj.offset2:obj.ssize

                seq_obj_function(params) = begin
                    γ_transformed, ϵ_transformed,  = params
                    ϵ = sigmoid_map(ϵ_transformed, obj.a_ϵ, obj.b_ϵ)
                    γ = sigmoid_map(γ_transformed, obj.a_γ, obj.b_γ)
                    
                    neg_likelihood = loglik(obj, t, [γ, ϵ])
                    neg_likelihood
                end
            
                res = optimize(seq_obj_function, p0, BFGS(), Optim.Options(g_tol=obj.g_tol, iterations=obj.max_iter, outer_iterations=obj.max_iter_outer, show_trace=obj.show_trace, show_warnings=obj.show_warnings))
           
                
                p0 = Optim.minimizer(res)


                γ_res = sigmoid_map(Optim.minimizer(res)[1], obj.a_γ, obj.b_γ)
                ϵ_res = sigmoid_map(Optim.minimizer(res)[2], obj.a_ϵ, obj.b_ϵ)
                
                p[:, t] = [γ_res, ϵ_res]
            end
            obj.γ = p[1, :]
            obj.ϵ = p[2, :]


        elseif obj.method == "fullsample" 
            p0 = [0.95, 0.55 * std(obj.x[1:obj.offset2-1])]
            p0[1] = inverse_sigmoid_map(p0[1], obj.a_γ, obj.b_γ)
            p0[2] = inverse_sigmoid_map(p0[2], obj.a_ϵ, obj.b_ϵ)

            fs_obj_function(params) = begin
                γ_transformed, ϵ_transformed,  = params
                ϵ = sigmoid_map(ϵ_transformed, obj.a_ϵ, obj.b_ϵ)
                γ = sigmoid_map(γ_transformed, obj.a_γ, obj.b_γ)
                
                neg_likelihood = loglik(obj, obj.ssize, [γ, ϵ])
                neg_likelihood
            end

            res = optimize(fs_obj_function, p0, BFGS(), Optim.Options(g_tol=obj.g_tol, iterations=obj.max_iter, outer_iterations=obj.max_iter_outer, show_trace=obj.show_trace, show_warnings=obj.show_warnings))

            γ_res = sigmoid_map(Optim.minimizer(res)[1], obj.a_γ, obj.b_γ)
            ϵ_res = sigmoid_map(Optim.minimizer(res)[2], obj.a_ϵ, obj.b_ϵ)

            obj.γ = γ_res
            obj.ϵ = ϵ_res

        else
            error("Method \"$(obj.method)\" not allowed.")
        end
        # fix_γ = NaN and  fix_γ = NaN
    elseif obj.filter == "smoothing" && isnan(fix_γ) && !multiple_bandwidth
        p0 = [0.95, 0.55 * std(obj.x[1:obj.offset2-1])]
    
        p0[1] = inverse_sigmoid_map(p0[1], obj.a_γ, obj.b_γ)
        p0[2] = inverse_sigmoid_map(p0[2], obj.a_ϵ, obj.b_ϵ)
        
        sm_obj_function(params) = begin
            γ_transformed, ϵ_transformed  = params
            ϵ = sigmoid_map(ϵ_transformed, obj.a_ϵ, obj.b_ϵ)
            γ = sigmoid_map(γ_transformed, obj.a_γ, obj.b_γ)

            neg_likelihood = lik_cv(obj, [γ, ϵ])
            neg_likelihood
        end
        
        
        res = optimize(
            sm_obj_function,
            p0,
            BFGS(),
            Optim.Options(
                g_tol=obj.g_tol,
                iterations=obj.max_iter,
                outer_iterations=obj.max_iter_outer,
                show_trace=obj.show_trace,
                show_warnings=obj.show_warnings,
            )
        )


        # res = optimize(sm_obj_function, p0, BFGS(), Optim.Options(g_tol=obj.g_tol, iterations=obj.max_iter, outer_iterations=obj.max_iter_outer, show_trace=obj.show_trace, show_warnings=obj.show_warnings))

        γ_res = sigmoid_map(Optim.minimizer(res)[1], obj.a_γ, obj.b_γ)
        ϵ_res = sigmoid_map(Optim.minimizer(res)[2], obj.a_ϵ, obj.b_ϵ)

        obj.γ = γ_res
        obj.ϵ = ϵ_res

    elseif obj.filter == "smoothing" && !isnan(fix_γ) && !multiple_bandwidth
        p0 = [0.55 * std(obj.x[1:obj.offset2-1])]
    
        p0[1] = inverse_sigmoid_map(p0[1], obj.a_ϵ, obj.b_ϵ)
        
        sm_obj_function_fixed(params) = begin
            ϵ_transformed  = params[1]
            ϵ = sigmoid_map(ϵ_transformed, obj.a_ϵ, obj.b_ϵ)
            
            neg_likelihood = lik_cv(obj, [fix_γ, ϵ])
            neg_likelihood
        end
        
        
        res = optimize(
            sm_obj_function_fixed,
            p0,
            BFGS(),
            Optim.Options(
                g_tol=obj.g_tol,
                iterations=obj.max_iter,
                outer_iterations=obj.max_iter_outer,
                show_trace=obj.show_trace,
                show_warnings=obj.show_warnings,
            )
        )


        # res = optimize(sm_obj_function, p0, BFGS(), Optim.Options(g_tol=obj.g_tol, iterations=obj.max_iter, outer_iterations=obj.max_iter_outer, show_trace=obj.show_trace, show_warnings=obj.show_warnings))

        ϵ_res = sigmoid_map(Optim.minimizer(res)[1], obj.a_ϵ, obj.b_ϵ)

        obj.γ = fix_γ
        obj.ϵ = ϵ_res
    
    elseif obj.filter == "smoothing" && !isnan(fix_γ) && multiple_bandwidth
        p0 = [0.55 * std(obj.x[1:obj.offset2-1]), 0.55 * std(obj.x[1:obj.offset2-1]), 0.55 * std(obj.x[1:obj.offset2-1])]
    
        p0[1] = inverse_sigmoid_map(p0[1], obj.a_ϵ, obj.b_ϵ)
        p0[2] = inverse_sigmoid_map(p0[2], obj.a_ϵ, obj.b_ϵ)
        p0[3] = inverse_sigmoid_map(p0[3], obj.a_ϵ, obj.b_ϵ)
        
        sm_obj_function_fixed_multiple_ϵ(params) = begin
            ϵ1_transformed, ϵ2_transformed, ϵ3_transformed = params[1], params[2], params[3] 
            ϵ1 = sigmoid_map(ϵ1_transformed, obj.a_ϵ, obj.b_ϵ)
            ϵ2 = sigmoid_map(ϵ2_transformed, obj.a_ϵ, obj.b_ϵ)
            ϵ3 = sigmoid_map(ϵ3_transformed, obj.a_ϵ, obj.b_ϵ)

            neg_likelihood = lik_cv(obj, [fix_γ, ϵ1, ϵ2, ϵ3])
            neg_likelihood
        end
        
        
        res = optimize(
            sm_obj_function_fixed_multiple_ϵ,
            p0,
            BFGS(),
            Optim.Options(
                g_tol=obj.g_tol,
                iterations=obj.max_iter,
                outer_iterations=obj.max_iter_outer,
                show_trace=obj.show_trace,
                show_warnings=obj.show_warnings,
            )
        )


        # res = optimize(sm_obj_function, p0, BFGS(), Optim.Options(g_tol=obj.g_tol, iterations=obj.max_iter, outer_iterations=obj.max_iter_outer, show_trace=obj.show_trace, show_warnings=obj.show_warnings))

        ϵ1_res = sigmoid_map(Optim.minimizer(res)[1], obj.a_ϵ, obj.b_ϵ)
        ϵ2_res = sigmoid_map(Optim.minimizer(res)[2], obj.a_ϵ, obj.b_ϵ)
        ϵ3_res = sigmoid_map(Optim.minimizer(res)[3], obj.a_ϵ, obj.b_ϵ)

        obj.γ = fix_γ
        obj.ϵ = [ϵ1_res, ϵ2_res, ϵ3_res]
       
    else
        error("Filtering approach \"$(obj.filter)\" is not allowed.")
    end

end

function estimate_LDE_grid(obj, grid_size_γ = 3, grid_size_ϵ = 3)
    
    γ_options = range(obj.a_γ, obj.b_γ, length = grid_size_γ)
    ϵ_options = range(obj.a_ϵ, obj.b_ϵ, length = grid_size_ϵ)

    # all combinations 
    params = Iterators.product(γ_options, ϵ_options) |> collect

    # Map the likelihood function over all parameter combinations in parallel
    results = pmap(params) do param
        γ, ϵ = param
        neg_likelihood = lik_cv(obj, [γ, ϵ])
        (γ, ϵ, neg_likelihood)
    end
    results = collect(results)

    # Find the parameter combination with the minimum likelihood
    min_index = argmin([x[3] for x in results])
    γ_res, ϵ_res, min_likelihood = results[min_index]
    # Update the object with the best parameters found
    obj.γ = γ_res
    obj.ϵ = ϵ_res

    return γ_res, ϵ_res, min_likelihood, results
end
   
