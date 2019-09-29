using Documenter, GrowthDynamics

run(`attrib -R build /D`)
makedocs(sitename="GrowthDynamics.jl",
    modules=[GrowthDynamics],
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Overview" => "index.md",
        "Dynamics" => "dynamics.md",
        "Observables" => "observables.md"
    ],
    clean=true)
