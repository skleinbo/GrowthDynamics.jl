using Documenter, GrowthDynamics

makedocs(sitename="GrowthDynamics.jl",
    modules=[GrowthDynamics],
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Overview" => "index.md",
        "Dynamics" => "dynamics.md",
        "Observables" => "observables.md"
    ],
    clean=true)

deploydocs(
    repo = "github.com/skleinbo/GrowthDynamics.jl.git",
)