#= The purpose of this file will be to create a neural network that will be trained to find the 
coefficient of lift for any single lifting surface, being trained with values derived from using
the Vortex Lattice Method. It should be able to predict the CL for different geometries of lifting
surfaces as well. =#
#= Strategy: First, create a working neural network that can accurately predict the Coefficient of
lift within a percent using the angle of attack as the only changing variable. Next, add variables
which can accurately describe other parameters whcih would affect our VLM analysis, especially geometry.
Train the neural network to accurately predict the CL based on both the geometry and freestream. =#

# Load Necessary Packages
using VortexLattice
using Lux
using Optimisers
using Zygote
using Random

function alpha_analysis(is_training, alpha_val)
    #Set up the geometric properties of half the wing
    xle = [0.0, 0.4] # leading edge x-position
    yle = [0.0, 7.5] # leading edge y-position
    zle = [0.0, 0.0] # leading edge z-position
    chord = [2.2, 1.8] # chord length
    theta = [2.0*pi/180, 2.0*pi/180] # twist (in radians)
    phi = [0.0, 0.0] # section rotation about the x-axis
    fc = fill((xc) -> 0, 2) # camberline function for each section (y/c = f(x/c))
    # define the number of panels in the spanswise and chordwise directions
    ns = 12 # number of spanwise panels
    nc = 6  # number of chordwise panels
    spacing_s = Sine() # spanwise discretization scheme
    spacing_c = Uniform() # chordwise discretization scheme
    # generate a lifting surface for the defined geometry
    grid, ratio = wing_to_grid(xle, yle, zle, chord, theta, phi, ns, nc;
    fc = fc, spacing_s=spacing_s, spacing_c=spacing_c, mirror=true)
    # combine all grids to one vector as well as ratios into one vector
    grids = [grid]
    ratios = [ratio]
    system = System(grids; ratios)
    # define freestream and reference for the model
    Sref = 30.0 # reference area
    cref = 2.0  # reference chord
    bref = 15.0 # reference span
    rref = [0.50, 0.0, 0.0] # reference location for rotations/moments (typically the c.g.)
    Vinf = 1.0 # reference velocity (magnitude)
    ref = Reference(Sref, cref, bref, rref, Vinf)
    # freestream definition
    # make lists for the different coefficients we are going to be plotting
    if is_training == true
        a0 = -20
        af = 60
        CL_lst = zeros((af-a0 + 1))
        AA_lst = zeros((af-a0 + 1))
        # Insert range for angle in degrees below
        # Iterate through the different angles of attack, one degree at a time
        for i in range(a0, af)
            alpha = i*pi/180 # angle of attack, where i is degrees but the program uses radians.
            beta = 0.0 # sideslip angle
            Omega = [0.0, 0.0, 0.0] # rotational velocity around the reference location
            fs = Freestream(Vinf, alpha, beta, Omega)
            # we already mirrored, so we do not need a symmetric calculation
            symmetric = false

            steady_analysis!(system, ref, fs; symmetric)
            # Extract all body forces
            CF, CM = body_forces(system; frame=Wind())
            # extract aerodynamic forces
            CD, CY, CL = CF
            Cl, Cm, Cn = CM
            CDiff = far_field_drag(system)
            # add all of the coefficients for this specific angle of attack to their lists, including the angle of attack list
            CL_lst[i-a0+1] = CL
            AA_lst[i-a0+1] = i
        end
        return CL_lst, AA_lst
    else
        alpha = alpha_val*pi/180 # angle of attack, where i is degrees but the program uses radians.
        beta = 0.0 # sideslip angle
        Omega = [0.0, 0.0, 0.0] # rotational velocity around the reference location
        fs = Freestream(Vinf, alpha, beta, Omega)
        # we already mirrored, so we do not need a symmetric calculation
        symmetric = false

        steady_analysis!(system, ref, fs; symmetric)
        # Extract all body forces
        CF, CM = body_forces(system; frame=Wind())
        # extract aerodynamic forces
        CD, CY, CL = CF
        Cl, Cm, Cn = CM
        CDiff = far_field_drag(system)
        return CL
    end
end

function train_model()
    y_train_vec, x_train_vec = alpha_analysis(true, 0)
    # --- 1. Define Constants and Generate Training Data ---
    println("Generating training data...")
    num_samples = length(y_train_vec)

    # Data must be shaped as (features x samples) for Lux: 1 row x 200 columns
    x_train = reshape(x_train_vec, 1, num_samples)
    y_train = reshape(y_train_vec, 1, num_samples)

    # --- 2. Define the Neural Network Model (The Hypothesis) ---
    println("Defining the model...")

    # In Lux, models are defined using the sequential layer constructor
    nodes = 21
    model = Lux.Chain(
        #Lux.Dense(1 => 5, relu),
        #Lux.Dense(2 => nodes, relu),
        #Lux.Dense(nodes => nodes, relu),
        #Lux.Dense(nodes => nodes, relu),
        Lux.Dense(1 => 15, relu),
        Lux.Dense(15 => 1) # Making an inbetweeen layer
    )

    # Initialize the model state and parameters
    rng = Random.default_rng()
    params, state = Lux.setup(rng, model)

    println("Model defined.")
    while true
        # --- 3. Define Loss Function and Optimizer ---
        # Loss function: Mean Squared Error (MSE)
        function loss_function(model, ps, st, x, y)
            y_pred, st_new = model(x, ps, st) # Get prediction and updated state
            # Calculate MSE loss
            l = sum((y_pred .- y).^2)
            return l, st_new # Return loss and the *new* state
        end

        # Optimizer: ADAM
        optimizer = Optimisers.Adam(0.01f0) # Learning rate
        opt_state = Optimisers.setup(optimizer, params)

        println("Loss and optimizer defined.")

        # --- 4. Training Loop ---
        println("\nStarting training...")
        epochs = 3000
        # Directly update params, state, and opt_state defined outside the loop
        l_val = 1
        for epoch in 1:epochs
            # Calculate gradients using Zygote.withgradient
            # Pass the current params and state from the outer scope
            (loss_val, updated_state), grads = Zygote.withgradient(
                # The function to differentiate still takes parameters `ps` as its input
                ps -> loss_function(model, ps, state, x_train, y_train),
                params # Take gradient with respect to the *current* outer `params`
            )

            # Update optimizer state and parameters using the gradients
            # This directly modifies the outer `opt_state` and `params` variables
            opt_state, params = Optimisers.update(opt_state, params, grads[1])

            # Update the state for the next iteration using the value returned by loss_function
            state = updated_state

            # Print loss every 100 epochs
            if epoch % 100 == 0
                println("Epoch: $epoch, Loss: $loss_val")
            end
            if loss_val < 0.01
                l_val = loss_val
                break
            end
        end
        if l_val < 0.01
            break
        end
    end

    println("Training finished.")

    # --- 5. Evaluate and Compare Parameters ---
    # Extract learned parameters from the 'params' NamedTuple
    learned_weight = params.layer_1.weight[1, 1]
    learned_bias = params.layer_1.bias[1]

    println("\n--- Final Learned Parameters ---")
    println("Target Slope (Weight): 5.0")
    println("Learned Slope (Weight): $(learned_weight)")
    println("Target Intercept (Bias): 3.0")
    println("Learned Intercept (Bias): $(learned_bias)")

    # --- 6. Plotting the Results ---
    # Predictions in Lux require passing the final params and state
    x_plot = reshape(collect(range(-20.0f0, 60.0f0, length=100)), 1, 100)
    # Use Lux.apply for prediction after training
    y_pred, _ = Lux.apply(model, x_plot, params, state)

    # Create the plot
    plot(x_train_vec, y_train_vec, seriestype=:scatter, label="Noisy Training Data",
        title="NN Learning Linear Function")
    plot!(vec(x_plot), vec(y_pred), label="NN Prediction (Learned)", linewidth=3, linestyle=:dash, color=:blue)
    xlabel!("x")
    ylabel!("y")
    savefig("nn_linear_regression_plot.png")
    println("Plot saved as nn_linear_regression_plot.png")
end

function main()
    train_model()
end



