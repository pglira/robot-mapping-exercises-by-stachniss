using Plots
using LinearAlgebra

function PDF_moments_parametrization(μ, Σ, x)
    p = LinearAlgebra.det(2*π*Σ)^(-1/2)*exp(-1/2*transpose(x-μ)*inv(Σ)*(x-μ))
end

function PDF_canonical_parametrization(ξ, Ω, x)
    μ = inv(Ω)*ξ
    p = (exp(-1/2*transpose(μ)*ξ)/LinearAlgebra.det(2*π*inv(Ω))^(1/2))*
        exp(-1/2*transpose(x)*Ω*x+transpose(x)*ξ)
end

function main()

    # Given: mean and covariance matrix of a twodimensional stochastic variable
    μ = [1
         2]
    Σ = [0.3 0.1
         0.1 0.5]

    # First, let's compute the PDF at the query point x
    x = [1.2
         2.4]

    # PDF at query point in moments parametrization, i.e. covariance form
    p = PDF_moments_parametrization(μ, Σ, x)
    println("p = $p (moments parametrization, i.e. covariance form)")

    # PDF at query point in canonical parametrization, i.e. information form
    Ω = inv(Σ)
    ξ = inv(Σ)*μ
    p = PDF_canonical_parametrization(ξ, Ω, x)
    println("p = $p (canonical parametrization, i.e. information form)")

    # Now let's plot the 3-dimensional PDF
    pyplot()
    dx = dy = 2
    # f(x,y) = PDF_moments_parametrization(μ, Σ, [x;y])
    f(x,y) = PDF_canonical_parametrization(ξ, Ω, [x;y])
    x = range(μ[1]-dx, stop=μ[1]+dx, length=100)
    y = range(μ[2]-dy, stop=μ[2]+dy, length=100)
    plot(x, y, f, st=:surface, cmap="coolwarm", camera=(-30,30))
    png("PDF.png")

end

main()
