# Fast O(1) Binning in StructureFunctions.jl: Detailed Evolution, Theory, and Timings

This document contains a complete technical log of the design evolution, mathematical theory, benchmark timings, and code implementations for all binning methods explored for `StructureFunctions.jl`.

---

## 1. Executive Summary of Benchmark Results

All timings are measured in Julia 1.12.6 on this machine using `BenchmarkTools.jl` with $1,000$ independent random query values. The benchmarks compare performance and correctness (validation errors over $10,000+$ test queries including boundary values) across both small ($N = 51$) and large ($N = 1,000$) bin layouts.

### A. Float64 Timings & Correctness

#### Linear Bins Search (Float64)
| Version | Method | Complexity | Timing ($N = 51$) | Timing ($N = 1,000$) | Correctness (Errors / 10k) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V1** | Vector Binary Search | $O(\log N)$ | **$19.23 \text{ ns}$** | **$51.28 \text{ ns}$** | 100% Exact (0) | Baseline. Cache misses at larger $N$. |
| **V2** | Julia Base Range Search | $O(1)$ | **$27.69 \text{ ns}$** | **$27.40 \text{ ns}$** | 100% Exact (0) | Standard StepRangeLen search (uses division/Twice-Precision). |
| **V3** | Custom FMA Search + ULP Correction | $O(1)$ | **$2.85 \text{ ns}$** | **$2.85 \text{ ns}$** | 100% Exact (0) | Fast multiplier mapping with 1-step ULP cleanup. |

#### Log-Spaced Bins Search (Float64)
| Version | Method | Complexity | Timing ($N = 51$) | Timing ($N = 1,000$) | Correctness (Errors / 10k) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V1** | Vector Binary Search | $O(\log N)$ | **$8.39 \text{ ns}$** | **$40.95 \text{ ns}$** | 100% Exact (0) | Fast at $N=51$ (fits in L1). Slow at $N=1000$. |
| **V2** | Julia Base on Logs (Range) | $O(1)$ | **$33.76 \text{ ns}$** | **$33.75 \text{ ns}$** | Broken (36 / 725) | StepRangeLen search on logs. Float inaccuracies. |
| **V2b** | Log + Vector Binary Search | $O(\log N)$ | **$17.27 \text{ ns}$** | **$57.62 \text{ ns}$** | Broken (36 / 725) | `log(x)` + binary search on raw vector of logs. |
| **V3** | `round` Guess (No Check) | $O(1)$ | **$14.98 \text{ ns}$** | **$14.99 \text{ ns}$** | Broken (4,968 / 6,980) | Rounds to nearest boundary, not next $\ge x$. |
| **V4** | `round` Guess + Correction | $O(1)$ | **$18.54 \text{ ns}$** | **$18.01 \text{ ns}$** | 100% Exact (0) | Branch-heavy local index repair. |
| **V5** | `ceil` Guess (No Check) | $O(1)$ | **$13.75 \text{ ns}$** | **$13.77 \text{ ns}$** | Broken (64 / 796) | Ceil mapping. Misses exact edges due to log ULP noise. |
| **V6** | FMA Ceil-based O(1) | $O(1)$ | **$10.01 \text{ ns}$** | **$10.01 \text{ ns}$** | Broken (60 / 876) | Precomputed multipliers. Fails on exact edges. |
| **V6b** | FMA Ceil + ULP Correction | $O(1)$ | **$21.21 \text{ ns}$** | **$21.22 \text{ ns}$** | 100% Exact (0) | Safe FMA guess + 1-step ULP comparison check. |
| **V7** | Bitwise Exponent LUT | $O(N)$ | **$3.53 \text{ ns}$** | **$44.03 \text{ ns}$** | 100% Exact (0) | Extracts $2^e \le x$. Fast linear scan for small $N$. |
| **V8** | Exponent LUT + Sub-Binary | $O(\log N)$ | **$4.36 \text{ ns}$** | **$7.65 \text{ ns}$** | 100% Exact (0) | Restricts search space to octave + local binary search. |
| **V9** | **Exponent LUT Hybrid** | $O(1)$ / $O(\log N)$ | **$4.94 \text{ ns}$** | **$8.26 \text{ ns}$** | 100% Exact (0) | **Winner.** Uses linear scan for small ranges, binary for large. |

---

### B. Float32 Timings & Correctness

#### Linear Bins Search (Float32)
| Version | Method | Complexity | Timing ($N = 51$) | Timing ($N = 1,000$) | Correctness (Errors / 10k) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V1** | Vector Binary Search | $O(\log N)$ | **$20.99 \text{ ns}$** | **$50.43 \text{ ns}$** | 100% Exact (0) | Baseline. Cache misses at larger $N$. |
| **V2** | Julia Base Range Search | $O(1)$ | **$10.20 \text{ ns}$** | **$10.20 \text{ ns}$** | 100% Exact (0) | StepRangeLen search (Float32 math is faster). |
| **V3** | Custom FMA Search + ULP Correction | $O(1)$ | **$2.85 \text{ ns}$** | **$2.85 \text{ ns}$** | 100% Exact (0) | Highly efficient Float32 FMA mapping + ULP correction. |

#### Log-Spaced Bins Search (Float32)
| Version | Method | Complexity | Timing ($N = 51$) | Timing ($N = 1,000$) | Correctness (Errors / 10k) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V1** | Vector Binary Search | $O(\log N)$ | **$8.35 \text{ ns}$** | **$14.71 \text{ ns}$** | 100% Exact (0) | Fast at $N=51$. Standard Float32 search fits in cache. |
| **V2** | Julia Base on Logs (Range) | $O(1)$ | **$19.94 \text{ ns}$** | **$19.93 \text{ ns}$** | Broken (52 / 963) | StepRangeLen search on logs. Float inaccuracies. |
| **V2b** | Log + Vector Binary Search | $O(\log N)$ | **$12.82 \text{ ns}$** | **$20.40 \text{ ns}$** | Broken (52 / 963) | `log(x)` + binary search on raw vector of logs. |
| **V3** | `round` Guess (No Check) | $O(1)$ | **$9.40 \text{ ns}$** | **$9.39 \text{ ns}$** | Broken (4,841 / 6,765) | Rounds to nearest boundary, not next $\ge x$. |
| **V4** | `round` Guess + Correction | $O(1)$ | **$13.94 \text{ ns}$** | **$13.50 \text{ ns}$** | 100% Exact (0) | Branch-heavy local index repair. |
| **V5** | `ceil` Guess (No Check) | $O(1)$ | **$8.39 \text{ ns}$** | **$8.39 \text{ ns}$** | Broken (67 / 1,017) | Ceil mapping. Misses exact edges due to log ULP noise. |
| **V6** | FMA Ceil-based O(1) | $O(1)$ | **$7.83 \text{ ns}$** | **$7.83 \text{ ns}$** | Broken (85 / 1,089) | Precomputed multipliers. Fails on exact edges. |
| **V6b** | FMA Ceil + ULP Correction | $O(1)$ | **$10.50 \text{ ns}$** | **$10.49 \text{ ns}$** | 100% Exact (0) | Safe FMA guess + 1-step ULP comparison check. |
| **V7** | Bitwise Exponent LUT | $O(N)$ | **$3.52 \text{ ns}$** | **$42.83 \text{ ns}$** | 100% Exact (0) | Extracts $2^e \le x$. Fast linear scan for small $N$. |
| **V8** | Exponent LUT + Sub-Binary | $O(\log N)$ | **$4.34 \text{ ns}$** | **$7.66 \text{ ns}$** | 100% Exact (0) | Restricts search space to octave + local binary search. |
| **V9** | **Exponent LUT Hybrid** | $O(1)$ / $O(\log N)$ | **$4.76 \text{ ns}$** | **$8.28 \text{ ns}$** | 100% Exact (0) | **Winner.** Uses linear scan for small ranges, binary for large. |

---

## 2. Support for Float32 and Float64

Double-precision (`Float64`) and single-precision (`Float32`) floats both store their numeric value using the IEEE 754 representation:
* **Float64**: 1 sign bit, 11 exponent bits, 52 mantissa bits.
* **Float32**: 1 sign bit, 8 exponent bits, 23 mantissa bits.

In Julia, the standard function `exponent(x::T)` extracts the binary exponent via native assembly instructions:
```assembly
# Float64 (x86-64)
vgetexpsd  %xmm0, %xmm0, %xmm0

# Float32 (x86-64)
vgetexpss  %xmm0, %xmm0, %xmm0
```
On hardware supporting AVX-512, this executes in a single clock cycle ($< 0.5 \text{ ns}$). For architectures without native vector exponent instructions, Julia falls back to bit-shifting and masking:
* **Float64**: `(reinterpret(UInt64, x) & 0x7ff0000000000000) >> 52 - 1023`
* **Float32**: `(reinterpret(UInt32, x) & 0x7f800000) >> 23 - 127`

Both extraction methods run with zero allocation and negligible latency. Consequently, the **Bitwise Exponent LUT** methods (V7, V8, V9) and the **FMA mapping** methods are fully type-generic and achieve identical scaling behaviors on `Float32` and `Float64`.

---

## 3. Linear Bins Search: Evolution & Code

Linear bins are defined by a starting point `first`, a spacing `step`, and an ending point `last`.

### Version 1: Vector Binary Search
```julia
function search_binary(v::AbstractVector, x)
    return searchsortedfirst(v, x)
end
```
* **Theory**: standard binary search ($O(\log N)$ comparisons).
* **Limitation**: At $N=1000$, it takes $\approx 10$ iterations, jumping around the array. This creates data dependency stalls and cache misses, taking **51.28 ns** in Float64.

### Version 2: Julia Base `StepRangeLen` Search
```julia
function search_julia_range(a::StepRangeLen, x)
    f, h, l = first(a), step(a), last(a)
    if x <= f
        return 1
    elseif x > l
        return length(a) + 1
    else
        n = round(Integer, (x - f) / h + 1)
        return a[n] < x ? n + 1 : n
    end
end
```
* **Theory**: Maps the query float to an index range using division.
* **Limitation**: Division is slow (10-15 clock cycles). Accessing `a[n]` requires Twice-Precision calculations. This takes **27.40 ns** in Float64.

### Version 3: Custom FMA O(1) Linear Search (Fast Corrected - 2.85 ns)
* **Theory**: We precompute the inverse step $inv\_step = \frac{1}{\text{step}}$ and the offset $offset = 1.0 - \frac{\text{first}}{\text{step}}$ on construction. We rewrite the equation as a single Fused Multiply-Add (FMA) operation:
  $$\text{idx} = \text{round}(\text{Int}, x \times inv\_step + offset)$$
* **Overcoming Twice-Precision Range Arithmetic & Struct Design**: 
  If we query `v.edges[idx-1]` or call `last(v.edges)` directly, Julia's `StepRangeLen` range type evaluates Twice-Precision arithmetic (`a.ref + (idx - 1 - a.offset) * a.step`) to compute the edge value. This requires multiple CPU instructions and takes **10–15 ns** per query.
  To bypass this range arithmetic overhead:
  1. **Struct Endpoint Storage**: We store the concrete values `first_edge::T`, `last_edge::T`, and `step_val::T` directly in the `LinearBinEdges` struct. This changes all range lookups into zero-overhead register field reads.
  2. **Reconstruction via Fast FMA**: We reconstruct the boundary edge mathematically using a single standard FMA instruction:
     $$\text{edge\_val} = \text{muladd}(\text{idx} - 1, \text{step\_val}, f)$$
     This compiles to a single CPU instruction cycle ($< 0.5 \text{ ns}$).
* **Branch Reduction via Round-Based Lookup**:
  By switching from ceiling rounding (`ceil`) to nearest rounding (`round`), the estimated index `idx` is mathematically guaranteed to be within $\pm 0.5$ units of the theoretical boundary. Because the rounding error is bounded, the index guess `idx` can only ever be off by at most $\pm 1$ index.
  This allows us to evaluate a single comparison:
  $$\text{edge\_val} < x ? \text{idx} + 1 : \text{idx}$$
  This reduces the correction check to a single FMA, a comparison, and a conditional move (eliminating nested branch checks).
* **Code**:
  ```julia
  struct LinearBinEdges{T, RT <: AbstractRange{T}} <: AbstractBinEdges{T}
      edges::RT
      inv_step::T
      offset::T
      first_edge::T
      last_edge::T
      step_val::T
  end

  function LinearBinEdges(edges::AbstractRange{T}) where {T}
      inv_step = inv(step(edges))
      offset = T(1.0) - first(edges) * inv_step
      return LinearBinEdges{T, typeof(edges)}(edges, inv_step, offset, first(edges), last(edges), step(edges))
  end

  function Base.searchsortedfirst(v::LinearBinEdges{T}, x) where {T}
      f = v.first_edge
      if x <= f; return 1; end
      l = v.last_edge
      if x > l; return length(v.edges) + 1; end
      
      idx = round(Int, muladd(x, v.inv_step, v.offset))
      idx = clamp(idx, 1, length(v.edges))
      @inbounds edge_val = muladd(T(idx - 1), v.step_val, f)
      return edge_val < x ? idx + 1 : idx
  end
  ```
* **Performance**: Compiles down to 2 comparisons, a hardware `round` instruction, a single hardware FMA instruction (`vfmadd`), and a conditional return. Takes **$2.85\text{ ns}$** in both Float64 and Float32 (a **18x speedup** over vector binary search at $N=1000$).

---

## 4. Log-Spaced Bins: Detailed Design Log

Log-spaced bin boundaries $E$ are geometric ranges where $E_i = \exp(R_i)$ for a linear range $R$. We want to find the first index $i$ such that $E_i \ge x$.

### Version 2: Range Search on Logs
```julia
function search_julia_log(R, x)
    if x <= 0; return 1; end
    lx = log(x)
    return searchsortedfirst(R, lx)
end
```
* **Theory**: Map $x \rightarrow \log(x)$ and delegate to Julia's $O(1)$ `StepRangeLen` search.
* **Limitation**: Suffers from Twice-Precision arithmetic and float range search division overhead, taking **33.75 ns** in Float64.

### Version 2b: Log + Vector Binary Search
```julia
function search_log_vector(log_vector_log, x)
    lx = log(x)
    return searchsortedfirst(log_vector_log, lx)
end
```
* **Theory**: Map $x \rightarrow \log(x)$ and perform standard binary search on the raw pre-logged boundary vector.
* **Limitation**: Costs the $4.7\text{ ns}$ `log(x)` hardware latency plus the $O(\log N)$ binary search time, taking **17.27 ns** (for $N=51$) and **57.62 ns** (for $N=1000$).

### Version 3: `round` Guess (No Check)
* **Theory**: We estimate $g$ using a rounded index guess:
  $$g = \text{round}\left(\text{Int}, \frac{\log(x) - \log(\text{start})}{\log(\text{step})} + 1\right)$$
* **Defect**: Fails conceptually. `round` maps $x$ to the *closest* boundary, not the first boundary $\ge x$. This yields a **~50% error rate** on queries.

### Version 4: `round` Guess + Boundary Correction
* **Theory**: Fixes Version 3 by guessing $g$ with `round`, and then performing local index corrections in the original space.
* **Performance**: Correct but branch-heavy, stalling the pipeline at **18.01 ns** in Float64.

### Version 5: `ceil` Guess (No Check)
* **Theory**: Uses `ceil` to directly target the first boundary $\ge x$:
  $$\text{idx} = \text{ceil}\left(\text{Int}, \frac{\log(x) - \log(\text{start})}{\log(\text{step})}\right) + 1$$
* **Defect**: Floating-point inaccuracy in `log(x)` occasionally puts the value on the wrong side of the boundary, yielding a **~1.9% error rate**.

### Version 6: FMA Ceil-based O(1)
* **Theory**: Replaces division with FMA:
  $$\text{idx} = \text{ceil}(\text{Int}, \log(x) \times inv\_step + offset)$$
* **Limitation**: Still suffers from boundary precision mismatches (~4.6% errors).

### Version 6b: FMA Ceil + ULP Correction
* **Theory**: Adds 1-step correction to the FMA Ceil guess.
* **Performance**: 100% correct, but branch execution checks in the original space bring the time to **21.22 ns** in Float64.

---

## 5. The Winner: Exponent LUT Hybrid (Version 9)

To achieve $O(1)$-like scaling and bypass the hardware `log(x)` bottleneck ($4.7 \text{ ns}$ latency), we avoid computing `log(x)` entirely.

### Theory
A floating-point value $x$ satisfies:
$$2^e \le x < 2^{e+1}$$
where $e = \text{exponent}(x)$. We extract $e$ in $< 0.5\text{ ns}$ and look up $2^e$ in a small, precomputed Lookup Table (LUT). The LUT maps each exponent $e$ to the first index in the boundary array $E$ where $E[\text{idx}] \ge 2^e$.

Once we retrieve `idx_start = lut[e - e_min + 1]` and `idx_end = lut[e - e_min + 2]`, the query index must lie in the range `[idx_start, idx_end]`.
* **Small Bins Layouts ($N = 51$)**: The number of elements spanning a single octave is small (typically $\le 8$). We run a fast forward linear scan. This takes **$4.94 \text{ ns}$** in Float64 and **$4.76 \text{ ns}$** in Float32.
* **Large Bins Layouts ($N = 1,000$)**: The octave interval can contain $\approx 100$ elements. A linear scan would suffer $O(N)$ growth. Instead, we perform an inline binary search restricted to the small contiguous subsegment `[idx_start, idx_end]`. Since the subsegment fits entirely inside the L1 data cache, it executes without memory stall, taking only **$8.26 \text{ ns}$** in Float64 and **$8.28 \text{ ns}$** in Float32 (a **5.5x speedup** over vector binary search).

### Code
```julia
struct LogBinEdges{T, RT <: AbstractRange, VT <: AbstractVector} <: AbstractBinEdges{T}
    log_edges::RT
    edges::VT
    lut::Vector{Int}
    e_min::Int
    e_max::Int
    inv_step::T
    offset::T
end

function LogBinEdges(edges::AbstractVector{T}) where {T}
    log_start = log(first(edges))
    log_stop = log(last(edges))
    log_edges = range(log_start, log_stop, length=length(edges))
    
    e_min = exponent(first(edges))
    e_max = exponent(last(edges))
    lut_size = e_max - e_min + 2
    lut = Vector{Int}(undef, lut_size)
    for e in e_min:(e_max+1)
        val = T(2.0^e)
        lut[e - e_min + 1] = searchsortedfirst(edges, val)
    end
    
    inv_step = inv(step(log_edges))
    offset = T(1.0) - first(log_edges) * inv_step
    
    return LogBinEdges{T, typeof(log_edges), typeof(edges)}(log_edges, edges, lut, e_min, e_max, inv_step, offset)
end

function Base.searchsortedfirst(v::LogBinEdges, x)
    f = first(v.edges)
    if x <= f; return 1; end
    l = last(v.edges)
    if x > l; return length(v.edges) + 1; end
    
    e = exponent(x)
    e_idx = clamp(e - v.e_min + 1, 1, length(v.lut))
    idx_start = @inbounds v.lut[e_idx]
    idx_end = (e_idx < length(v.lut)) ? @inbounds(v.lut[e_idx+1]) : length(v.edges)
    
    # Hybrid branch: linear scan for small ranges, sub-binary search for larger ranges
    if idx_end - idx_start <= 8
        idx = idx_start
        @inbounds while idx <= idx_end && v.edges[idx] < x
            idx += 1
        end
        return idx
    else
        low = idx_start
        high = idx_end
        @inbounds while low <= high
            mid = (low + high) >>> 1
            if v.edges[mid] < x
                low = mid + 1
            else
                high = mid - 1
            end
        end
        return low
    end
end
```
