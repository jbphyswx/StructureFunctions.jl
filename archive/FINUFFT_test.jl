# I wanna test if I can get FINUFFT to take a 2D NUFFT
# 2D NUFFT type 1


include("/home/jbenjami/Research_Schneider/CliMA/StructureFunctions.jl/src/FINUFFT.jl")


create_svector(x) = SVector{length(x)}(x)
np = 1251
a = create_svector( rand(Float32,  Int(np)));
b = create_svector( rand(Float32,  Int(np)));
xv = Array(a);
yv = Array(b);
# UnicodePlots.scatterplot(xv, yv)
uv = sin.(xv*10) 
vv = sin.(yv*20) 
# UnicodePlots.surfaceplot(xv, yv, uv.^2 + vv.^2 ,colormap=:jet, height=50, width=100) #xlabel="k_x", ylabel="k_y", title="E(k_x, k_y)", color=:blues, transpose=true)

x = Vector(create_svector.(zip(xv,yv)))
u = Vector(create_svector.(zip(uv,vv)))

# k from the mode range, which is integers lying in [-m/2, (m-1)/2]
ms::Int = 128
mt::Int = 128
k_x = range(-ms/2, stop=(ms-1)/2, length=ms) # k from the mode range, which is integers lying in [-m/2, (m-1)/2]
k_y = range(-mt/2, stop=(mt-1)/2, length=mt) # k from the mode range, which is integers lying in [-m/2, (m-1)/2]

k = sqrt.(k_x.^2 .+ (k_y.^2)') # k from the mode range, which is integers lying in [-m/2, (m-1)/2] # some of these are repeated, we have to do binning?

c_ku, c_kv = test_2d_nufft(x,u; ms=ms, mt=mt)

E_u = abs2.(c_ku) # f_ku .* conj(f_ku)
E_v = abs2.(c_kv) # f_kv .* conj(f_kv)

E = E_u[:,:,1] .+ E_v[:,:,1] # remove extraneous third dim


# group energy by wavenumber and take mean
E_vec = E[:]
k_vec = k[:]

# group E_vec by k_vec and take mean for each unique k_vec
unique_k_vec = unique(k_vec)
mean(x) = sum(x) / length(x)
E_mean = [mean(E_vec[k_vec .== k]) for k in unique_k_vec]



using UnicodePlots
UnicodePlots.heatmap(E ) #xlabel="k_x", ylabel="k_y", title="E(k_x, k_y)", color=:blues, transpose=true)
UnicodePlots.scatterplot(k[:], E[:], xscale=:log10, yscale=:log10)
UnicodePlots.scatterplot(unique_k_vec, E_mean, xscale=:log10, yscale=:log10)

UnicodePlots.scatterplot(unique_k_vec, E_mean, xscale=:identity, yscale=:identity)



## Can we calculate a NUFFT at every timestep? :o


# p1 = UnicodePlots.scatterplot(truth'[1,:], prediction[1,:],);
# p2 = UnicodePlots.scatterplot(truth'[2,:], prediction[2,:],);
# UnicodePlots.lineplot!(p1, 1:n, 1:n),
# UnicodePlots.lineplot!(p2, 1:n,  1:n) # if you use unicode plots, you can vizualize the correlation between the truth and the prediction



## ---------------------------------------------------------------- ##
a  = create_svector( rand(Float32,  Int(np)));
b = Vector(a);
nt::Int = 300
aaa  = repeat(a, 1, nt ,2);

# f = Tuple(create_svector.(zip(a,a)));
f = Vector(create_svector.(zip(a,a))); # slow?
fff = (x->collect(zip(x...))).( collect(zip(eachrow.(eachslice(aaa, dims=3))...)) );


# --------------------------------------------------------------- #
u_test = rand(Float32, np, nt, 2);
include("/home/jbenjami/Research_Schneider/CliMA/StructureFunctions.jl/src/FINUFFT_3D.jl")
test_2d_nufft_3D(x, u_test; ms=ms, mt=mt)