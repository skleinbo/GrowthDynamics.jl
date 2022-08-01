using Documenter, GrowthDynamics

makedocs(sitename="GrowthDynamics.jl",
    modules=[GrowthDynamics],
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Overview" => "index.md",
        "States" => "states.md",
        "Meta Data" => "metadata.md",
        "Phylogenies" => "phylogenies.md",
        "Dynamics" => "dynamics.md",
        "Observables" => "observables.md"
    ],
    clean=true)

deploydocs(
    repo = "github.com/skleinbo/GrowthDynamics.jl.git",
)
