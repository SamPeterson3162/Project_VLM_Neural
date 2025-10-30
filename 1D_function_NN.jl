# Load necessary Packages
using Lux
using Plots
using Optimisers
using Zygote
using Random

#= We will make a 1D function, create training data with noise, use it to train our model, and then plot the date with
the actual function side by side to show the loss. We will also crate a line for input to apply the trained model=#

function train(true_func)
    return true_func
end

function main()
    true_func(x) = x^2 + 2*x - 4
    model = train(true_func)
    while true
        println("What x-value would you like me to compute the y-value for?")
        x_val = parse(Float32,readline())
        y_val = model(x_val)
        true_y = true_func(x_val)
        println("X-Value:           $x_val")
        println("Predicted Y-Value: $y_val")
        println("True Y-Value:      $true_y")
        println("Would you like me to compute another value? (Y/N)")
        ans = readline()
        if (ans == "N"|| ans == "n")
            break
        end
    end
end
