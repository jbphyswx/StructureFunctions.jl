module ExampleResources

"""Validate clima's scheduler allocation before parallel or GPU example work."""
function require_allocation(; gpu=false, cpus=1)
    node = strip(read(`hostname -s`, String))
    node == "clima" || return nothing
    job = get(ENV, "SLURM_JOB_ID", "")
    step = get(ENV, "SLURM_STEP_ID", "")
    occursin(r"^\d+$", job) && !isempty(step) ||
        error("On clima, run this example inside a Slurm job step; see examples/README.md")
    uid = strip(read(`id -u`, String))
    record = read(`scontrol show job -o $job`, String)
    occursin("JobState=RUNNING", record) && occursin(Regex("UserId=[^ ]*\\(" * uid * "\\)"), record) ||
        error("Slurm allocation is not running or belongs to another user")
    nodes = match(r"(?:^| )NodeList=([^ ]+)", record)
    nodes !== nothing || error("Slurm allocation has no node list")
    node in split(read(`scontrol show hostnames $(nodes[1])`, String)) ||
        error("Current node is outside the allocation")
    parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "0")) >= cpus || error("Insufficient allocated CPUs")
    step_record = read(`scontrol show step $job.$step`, String)
    occursin("State=RUNNING", step_record) && occursin(Regex("UserId=" * uid * "(?: |\n)"), step_record) ||
        error("Slurm step is not running or belongs to another user")
    if gpu
        detail = read(`scontrol show job -dd $job`, String)
        assignment = findfirst(line -> occursin("Nodes=" * node * " ", line) && occursin("GRES=", line), split(detail, '\n'))
        assignment === nothing && error("No scheduler GPU assignment on this node")
        indices = match(r"GRES=gpu:[^ ]*\(IDX:([0-9,\-]+)\)", split(detail, '\n')[assignment])
        indices === nothing && error("No scheduler-confirmed GPU indices")
        expand(spec) = reduce(union, (begin
            bounds = parse.(Int, split(part, '-'))
            Set(first(bounds):last(bounds))
        end for part in split(spec, ',')); init=Set{Int}())
        assigned = get(ENV, "SLURM_STEP_GPUS", "")
        !isempty(assigned) && issubset(expand(assigned), expand(indices[1])) || error("GPU is outside the allocated step")
        get(ENV, "CUDA_VISIBLE_DEVICES", "") in ("", "-1", "NoDevFiles") && error("No GPU exposed by Slurm")
    end
    return nothing
end

function points()
    n = parse(Int, get(ENV, "SF_EXAMPLE_POINTS", "32"))
    n >= 2 || throw(ArgumentError("SF_EXAMPLE_POINTS must be at least two"))
    n > 64 && require_allocation()
    return n
end
end
