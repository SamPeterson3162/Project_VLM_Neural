using VortexLattice # load package we will be using
using DelimitedFiles

# This file writes our vlm_data file that will be used to train the neural network


function analyze_system(x1, x2, x3, x4)
    #Set up the geometric properties of half the wing
    xle = [0.0, 2-x1/2, 2-x2/2, 2-x3/2, 2-x4/2] # leading edge x-position
    yle = [0.0, 2.5, 5, 7.5, 10] # leading edge y-position
    zle = [0.0, 0.0, 0.0, 0.0, 0.0] # leading edge z-position
    chord = [4, x1, x2, x3, x4] # chord length
    theta = [0.0*pi/180, 0.0*pi/180, 0.0*pi/180, 0.0*pi/180, 0.0*pi/180] # twist (in radians)
    phi = [0.0, 0.0, 0.0, 0.0, 0.0] # section rotation about the x-axis
    fc = fill((xc) -> 0, 5) # camberline function for each section (y/c = f(x/c))
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
    # Insert range for angle in degrees below
    # Iterate through the different angles of attack, one degree at a time
    alpha = 5*pi/180 # angle of attack, where i is degrees but the program uses radians.
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
    return CL, CDiff
end

function main()
    n = 4
    # Define function to get chords, s is starting or larger chord value, e is smallest chord value or ending value, and y gives us a different chord for each value
    f(y,s,e) = s-.95*e*(y-1)/(n-1)
    input_lst = zeros(5,n^4)
    c = 1
    for i in 1:n
        for j in 1:n
            for k in 1:n
                for l in 1:n
                    x1 = f(i,4,2)
                    x2 = f(j,x1, x1/3)
                    x3 = f(k,x2, x2/3)
                    x4 = f(l,x3, x3/3)
                    input_lst[2,c] = x1
                    input_lst[3,c] = x2
                    input_lst[4,c] = x3
                    input_lst[5,c] = x4
                    c+=1
                end
            end
        end
    end
    for i in 1:n^4
        x1, x2, x3, x4 = input_lst[2:5,i]
        CL, CDiff = analyze_system(x1, x2, x3, x4)
        input_lst[1,i] = CL/CDiff
    end
    # Write to file
    output_file = "vlm_neural_net/vlm_data_file.data"
    delimiter = ' ' 
    writedlm(output_file, transpose(input_lst), delimiter)

    println("File '$output_file' written successfully.")
end