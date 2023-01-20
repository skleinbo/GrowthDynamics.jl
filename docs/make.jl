using Documenter, GrowthDynamics

module_load_expr = quote
    using GrowthDynamics
end
DocMeta.setdocmeta!(GrowthDynamics, :DocTestSetup, module_load_expr; recursive=true)

makedocs(sitename="GrowthDynamics.jl",
    modules=[GrowthDynamics.Lattices,
             GrowthDynamics.Populations,
             GrowthDynamics.LatticeDynamics,
             GrowthDynamics.Phylogenies,
             GrowthDynamics.Observables],
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Overview" => "index.md",
        "States" => ["states.md",
            "Lattices" => "lattices.md",
            "Metadata" => "metadata.md",
            "Phylogenies" => "phylogenies.md"],
        "Dynamics" => "dynamics.md",
        "Observables" => "observables.md"
    ],
    clean = (@isdefined CLEAN) && !CLEAN ? false : true,
    draft = (@isdefined DRAFT) && DRAFT ? true : false)

deploydocs(
    repo = "github.com/skleinbo/GrowthDynamics.jl.git",
    push_preview = true
)
