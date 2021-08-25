using Oceananigans
using Statistics
using Random
using UnicodePlots

function two_dimensional_turbulence(ν=1e-4; Nx=128, architecture=CPU(), advection=WENO5())
    Random.seed!(123)
    Lx = 2π

    model = NonhydrostaticModel(timestepper = :RungeKutta3,
                                architecture = architecture,
                                grid = RegularRectilinearGrid(size=(Nx, Nx), halo=(3, 3), extent=(Lx, Lx), topology=(Periodic, Periodic, Flat)),
                                advection = advection,
                                buoyancy = nothing,
                                tracers = nothing,
                                closure = IsotropicDiffusivity(ν=ν))

    u, v, w = model.velocities
    uᵢ = rand(size(u)...)
    vᵢ = rand(size(v)...)

    uᵢ .-= mean(uᵢ)
    vᵢ .-= mean(vᵢ)

    set!(model, u=uᵢ, v=vᵢ)

    progress(sim) = @info "(ν = $ν) iteration: $(sim.model.clock.iteration), time: $(round(Int, sim.model.clock.time))"

    simulation = Simulation(model, Δt=0.1, stop_time=30, iteration_interval=50, progress=progress)

    run!(simulation)

    return Array(interior(model.velocities.u))[:, :, 1]
end

u_reference = two_dimensional_turbulence()

fig = heatmap(u_reference)

display(fig)

viscosities = [1e-5, 2e-5, 5e-5, 8e-5, 9e-5, 1.1e-5, 1.2e-4, 2e-4, 5e-4, 1e-3]
errors = zeros(0)

for ν in viscosities
    u = two_dimensional_turbulence(ν)
    
    error = norm(u - u_reference) / norm(u_reference)
    
    push!(errors, error) 
end

fig = scatterplot(viscosities, errors)

display(fig)