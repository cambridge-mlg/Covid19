using Test, Covid19

@testset "NegativeBinomial2" begin
    μ = 1.
    ϕ = 2.
    dist = NegativeBinomial2(μ, ϕ)

    @test mean(dist) ≈ μ
    @test var(dist) ≈ μ + μ^2 / ϕ
end

@testset "GammaMeanCv" begin
    μ = 1.
    cv = √2
    dist = GammaMeanCv(μ, cv)

    @test mean(dist) ≈ μ
    @test var(dist) ≈ cv^2
end
