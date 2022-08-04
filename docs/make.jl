using Documenter, GrowthDynamics

makedocs(sitename="GrowthDynamics.jl",
    modules=[GrowthDynamics.Lattices,
             GrowthDynamics.TumorConfigurations,
             GrowthDynamics.LatticeTumorDynamics,
             GrowthDynamics.Phylogenies],
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Overview" => "index.md",
        "States" => ["states.md",
            "Lattices" => "lattices.md",
            "Meta Data" => "metadata.md",
            "Phylogenies" => "phylogenies.md"],
        "Dynamics" => "dynamics.md",
        "Observables" => "observables.md"
    ],
    clean=true)

deploydocs(
    repo = "github.com/skleinbo/GrowthDynamics.jl.git",
)
