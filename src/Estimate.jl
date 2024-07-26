function estimate_LDE(obj)

    if obj.filter == "filtering"
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
    elseif obj.filter == "smoothing"
        p0 = [0.95, 0.55 * std(obj.x[1:obj.offset2-1])]
        
        p0[1] = inverse_sigmoid_map(p0[1], obj.a_γ, obj.b_γ)
        p0[2] = inverse_sigmoid_map(p0[2], obj.a_ϵ, obj.b_ϵ)

        sm_obj_function(params) = begin
            γ_transformed, ϵ_transformed,  = params
            ϵ = sigmoid_map(ϵ_transformed, obj.a_ϵ, obj.b_ϵ)
            γ = sigmoid_map(γ_transformed, obj.a_γ, obj.b_γ)
            
            neg_likelihood = lik_cv(obj, [γ, ϵ])
            neg_likelihood
        end

        res = optimize(sm_obj_function, p0, BFGS(), Optim.Options(g_tol=obj.g_tol, iterations=obj.max_iter, outer_iterations=obj.max_iter_outer, show_trace=obj.show_trace, show_warnings=obj.show_warnings))

        γ_res = sigmoid_map(Optim.minimizer(res)[1], obj.a_γ, obj.b_γ)
        ϵ_res = sigmoid_map(Optim.minimizer(res)[2], obj.a_ϵ, obj.b_ϵ)

        obj.γ = γ_res
        obj.ϵ = ϵ_res

       
    else
        error("Filtering approach \"$(obj.filter)\" is not allowed.")
    end

end
