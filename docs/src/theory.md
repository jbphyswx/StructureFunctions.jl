# Theory: Structure Functions in Turbulence

This document explains the mathematical foundation of structure functions and how they're used to characterize turbulent flows.

## Table of Contents
- [Definition](#definition)
- [Longitudinal vs Transverse](#longitudinal-vs-transverse)
- [Order](#order)
- [Dimensional Variants](#dimensional-variants)
- [Curved Geometry: Structure Functions on a Sphere](#curved-geometry-structure-functions-on-a-sphere)
- [Kolmogorov Theory](#kolmogorov-theory)
- [References](#references)

## Definition

A **structure function** measures the statistical properties of velocity differences in a turbulent flow. Formally, the nth-order structure function in 1D is:

$$S_n(\Delta x) = \left\langle |\Delta u(\Delta x)|^n \right\rangle$$

where:
- $\Delta u(\Delta x) = u(x + \Delta x) - u(x)$ is the velocity increment over separation $\Delta x$
- The angle brackets $\langle \cdot \rangle$ denote ensemble/time averaging
- $n$ is the order (typically 2, 3, or higher)

### Physical Interpretation

Structure functions quantify how velocity changes as you move through a turbulent medium:

- **$S_2$ (Second-order)**: Measures the variance of velocity increments; related to the energy cascade
- **$S_3$ (Third-order)**: Directly related to the energy dissipation rate; satisfies the famous "-4/5 law" in Kolmogorov-scale turbulence
- **Higher orders**: Capture tail statistics and intermittency; used to detect coherent structures

## Longitudinal vs Transverse

For a point pair separated by vector $\vec{r}$, there are two natural projections of velocity differences:

### Longitudinal Structure Function

The velocity difference **along** the separation vector:

$$S^L_n(r) = \left\langle ([\Delta \vec{u}(\vec{r})] \cdot \hat{r})^n \right\rangle$$

where $\hat{r} = \vec{r} / |\vec{r}|$ is the unit vector in the direction of separation.

**Intuition**: Measures how velocity changes *in the direction you're moving*.

### Transverse Structure Function

The velocity difference **perpendicular** to the separation vector:

$$S^T_n(r) = \left\langle ([\Delta \vec{u}(\vec{r})] \times \hat{r})^2 \right\rangle^{n/2}$$

**Intuition**: Measures how velocity changes *perpendicular to your direction of motion*.

### Anisotropy

In most real turbulent flows (e.g., stratified atmosphere, rotating fluids), longitudinal and transverse structure functions differ. The ratio $S^L_n / S^T_n$ quantifies flow **anisotropy** — deviations from isotropy.

## Order

### Second-Order: Energy Spectrum

The second-order structure function is most commonly used:

$$S_2(\Delta x) = \langle (\Delta u)^2 \rangle$$

In isotropic turbulence, this relates to the energy spectrum $E(k)$ via:

$$S_2(r) = \int_0^{k_r} E(k) \sin(kr) / (kr) dk$$

### Third-Order: Energy Dissipation

The third-order longitudinal structure function has special significance:

$$S_3^L(r) = -\frac{4}{5} \varepsilon r$$

This **"-4/5 law"** (Kolmogorov, 1941; Landau & Lifshitz, 1987) directly relates the third-order SF to the energy dissipation rate $\varepsilon$, provided you're in the *inertial range* of turbulence.

### Higher Orders: Intermittency

Higher-order SFs are sensitive to **intermittency** — the non-Gaussian nature of velocity increments in real turbulence. The scaling exponents $\zeta_n$ defined by:

$$S_n(r) \sim r^{\zeta_n}$$

deviate from linear scaling ($\zeta_n \neq n/3$) in intermittent flows. These anomalous exponents are studied in **multifractal analysis** of turbulence.

## Dimensional Variants

StructureFunctions.jl supports calculations in multiple dimensions:

### 2D Flows

For 2D velocity fields $(u_x, u_y)$ in horizontal planes:
- Compute longitudinal/transverse SFs in the $(x, y)$ plane
- Study anisotropy between meridional and zonal components
- Common in meteorology (weather systems, jet streams)

### 3D Flows

For full 3D velocity fields $(u_x, u_y, u_z)$:
- Compute SFs along any separation vector
- Study isotropy/anisotropy in all directions
- Standard in direct numerical simulations (DNS)

### Projected / Vector Variants

- **Full vector SF**: Uses complete velocity difference vector
- **Projected SF**: Projects onto specific directions (e.g., vertical, meridional)
- Flexibility for analyzing anisotropic flows

## Curved Geometry: Structure Functions on a Sphere

Everything above assumes flat space: $\hat{r} = \vec{r}/|\vec{r}|$ is a single vector, and $\Delta\vec{u}$
is a plain difference of two velocity vectors. On a sphere neither holds. The two points have
*different* tangent planes, so "the separation direction" is two different vectors, and subtracting
velocities that live in different tangent spaces is not defined.

### The transported frame

The separation is the great-circle arc. With $\hat{p}, \hat{q}$ the unit position vectors,
$c = \hat{p}\cdot\hat{q}$, $\vec{w} = \hat{p}\times\hat{q}$, $s = |\vec{w}| = \sin\sigma$:

$$\sigma = 2\,\mathrm{atan2}\!\left(|\hat{q}-\hat{p}|,\ |\hat{q}+\hat{p}|\right), \qquad r = R\sigma$$

$$\hat{t}_A = \frac{\hat{q} - c\,\hat{p}}{s}, \qquad
  \hat{t}_B = \frac{c\,\hat{q} - \hat{p}}{s}, \qquad
  \hat{m}   = \frac{\vec{w}}{s}$$

$\hat{t}_A$ and $\hat{t}_B$ are the geodesic tangents at each endpoint, and $\hat{m}$ is the
great-circle normal. Two facts make this cheap: $|\hat{q}-c\hat{p}|^2 = |c\hat{q}-\hat{p}|^2 = s^2$,
so one reciprocal square root normalises both; and $\hat{m}$ is **the same vector at both endpoints**,
so the transverse direction needs no transport at all.

The increments are then

$$\delta u_L = \vec{u}_B\cdot\hat{t}_B - \vec{u}_A\cdot\hat{t}_A, \qquad
  \delta u_T = (\vec{u}_B - \vec{u}_A)\cdot\hat{m}$$

Projecting each velocity onto its *own* geodesic frame before differencing **is** parallel transport,
not an approximation: a geodesic parallel-transports its own tangent, and transport on $S^2$ is an
orientation-preserving isometry, so the frame carried from $A$ arrives at $B$ rotated by exactly the
difference of forward azimuths.

### Why a flat frame is wrong, and by how much

Using one flat direction for both endpoints ignores the **meridian convergence** — the angle $\psi$
between the two local frames:

$$\psi \approx \frac{r}{R}\tan\varphi$$

| separation | latitude | $\psi$ |
|---|---|---|
| 100 km | 45° | 0.9° |
| 1000 km | 45° | 9° |
| 1000 km | 70° | 25° |

$D_{LT}$ vanishes identically under reflection symmetry, so this leaks $O(\psi)\,(D_{TT}-D_{LL})/2$
into a quantity whose true value is zero, and the third-order cascade diagnostics $D_{LLL}$, $D_{LTT}$
inherit $O(\psi)$. It is the same pathology as spurious B-modes from flat-sky projection in weak
lensing. A direct measurement of the effect: for a solid-body rotation — which has *no* strain, so
$D_{LL}$ must vanish at every separation — the transported frame gives $\Sigma D_{LL}/\Sigma S_2
\approx 5\times10^{-32}$, while a flat lon/lat frame puts **36% of the total signal** into $D_{LL}$.

**Guidance:** below ~10 km a flat tangent plane is fine; beyond ~100 km, or anywhere poleward of
about 60°, use a spherical metric.

### Conventions and limits

- **Radial component.** For a thin shell ($D = 3$, $u = (\text{east},\text{north},\text{up})$) the
  radial component is differenced as a scalar and never transported: the geodesic frame is tangent to
  the shell, hence orthogonal to $\hat{p}$ at both endpoints, so radial motion cannot leak into
  $\delta u_L$ or $\delta u_T$. Ambient-transporting the full 3-vector would leak radial into
  tangential at $O(r/R)$ — a *first*-order error.
- **Coordinates.** A point on a shell is located by two numbers. `x` is `(2, N)` lon/lat while `u` may
  be `(2, N)` or `(3, N)`; the shell radius belongs to the metric, not to each point.
- **Angle units follow the metric.** `Distances.Haversine` is degrees, `Distances.SphericalAngle` is
  radians. Mixing them rescales every separation by ~57.
- **Handedness.** $\hat{n} = \hat{z}\times\hat{r}$, so $(\hat{r},\hat{n},\hat{z})$ is right-handed.
- **Degenerate pairs are excluded.** Coincident points have no direction, and antipodal points are
  joined by infinitely many great circles, so parallel transport is genuinely non-unique. Both are
  masked out rather than producing a NaN.
- **Precision.** Sub-metre separations on Earth need `Float64` input: at 45°, one `Float32` ulp of
  latitude is already ~4 m on the ground.
- **Isotropy relations are planar.** `KHM.jl`'s `transverse_incompressibility_residual`
  ($D_{TT} = \mathrm{d}(rD_{LL})/\mathrm{d}r$, Lindborg 1999 eq. 53) is a *planar* isotropic relation
  and holds on the sphere only to $O((r/R)^2)$.

### What the literature does

Published velocity-SF work is almost entirely regional ($\lesssim 500$ km), where $\psi$ is
negligible, and works in a local tangent plane: Balwada, LaCasce & Speer (2016) define
$\vec{r} = \Delta x\,\hat{i} + \Delta y\,\hat{j}$; `FluidSF` uses a great-circle distance for the
*scalar separation only* and then rotates with a flat grid-space angle. The transported-frame
convention adopted here is the one established in CMB polarization and weak-lensing shear
$\xi_\pm$, which face the identical problem.

## Kolmogorov Theory

### The Inertial Range

Kolmogorov's 1941 theory predicts that at scales $\eta \ll r \ll L$ (the *inertial range*):

$$S_2(r) \sim \varepsilon^{2/3} r^{2/3}$$

where:
- $\eta$ is the Kolmogorov scale (smallest scales, dominated by viscosity)
- $L$ is the integral length scale (largest scales, energy injection)
- $\varepsilon$ is the energy dissipation rate

### K41 Predictions

Kolmogorov predicted universal scaling exponents in isotropic turbulence:
- $\zeta_n = n/3$ (all orders scale identically)
- $S_2 \sim r^{2/3}$, $S_3 \sim r^{1}$, $S_4 \sim r^{4/3}$, etc.

### Deviations: Intermittency Correction

Real turbulence exhibits **anomalous exponents**:
$$\zeta_n = \frac{n}{3} + \delta_n$$

where $\delta_n > 0$ represents intermittency. These corrections are significant for $n > 3$.

## References

### Foundational

1. **Kolmogorov, A. N. (1941).** "The Local Structure of Turbulence in Incompressible Viscous Fluid for Very Large Reynolds Numbers." *Dokl. Akad. Nauk SSSR*, 30, 301–305.
   - Classic paper establishing the inertial-range energy cascade and K41 theory.

2. **Landau, L. D., & Lifshitz, E. M. (1987).** *Fluid Mechanics* (2nd ed.). Pergamon Press.
   - Chapter on turbulence; comprehensive treatment of SF theory and K41 predictions.

### Modern Reviews

3. **Frisch, U. (1995).** *Turbulence: The Legacy of A. N. Kolmogorov.* Cambridge University Press.
   - Modern perspective on Kolmogorov theory and intermittency corrections.

4. **Bos, W. J. T., & Rubinstein, R. (2014).** "On the energy spectrum of isotropic turbulence." *Physics of Fluids*, 26, 055107.
   - Recent review of SF scaling and spectrum relationships.

### Applications

5. **Balwada, D., Smith, K. S., & Flierl, G. (2016).** "Layer-specific parameterization of shortwave penetration radiation in the ocean." *Journal of Advances in Modeling Earth Systems*, 8, 1545–1567.
   - Application of structure functions to ocean turbulence.

6. **Skamarock, W. C., et al. (2019).** "A Description of the Advanced Research WRF Version 4." NCAR Technical Note.
   - Structure functions used in validation of weather model turbulence.

### Curved Geometry

8. **Kamionkowski, M., Kosowsky, A., & Stebbins, A. (1997).** "Statistics of cosmic microwave background polarization." *Physical Review D*, 55, 7368.
   - Establishes the transported-frame convention for two-point vector statistics on a sphere; flat-sky projection is documented to produce spurious B-modes, the same failure mode as spurious $D_{LT}$.

9. **Absil, P.-A., Mahony, R., & Sepulchre, R. (2008).** *Optimization Algorithms on Matrix Manifolds.* Princeton University Press, Ex. 8.1.1.
   - Closed-form parallel transport on $S^2$.

10. **Aluie, H. (2019).** "Convolutions on the sphere: commutation with differential operators." *GEM — International Journal on Geomathematics*, 10, 9. [doi:10.1007/s13137-019-0123-9](https://doi.org/10.1007/s13137-019-0123-9)
    - Prop. 2: normal/tangent character is preserved, which is why the radial component is differenced rather than transported.

11. **Lindborg, E. (1999).** "Can the atmospheric kinetic energy spectrum be explained by two-dimensional turbulence?" *Journal of Fluid Mechanics*, 388, 259–288. [doi:10.1017/S0022112099004851](https://doi.org/10.1017/S0022112099004851)
    - §5 longitudinal–transverse–vertical convention; eq. 53 the planar isotropy relation.

12. **Balwada, D., LaCasce, J. H., & Speer, K. G. (2016).** "Scale-dependent distribution of kinetic energy from surface drifters in the Gulf of Mexico." *Geophysical Research Letters*, 43. [doi:10.1002/2016GL069405](https://doi.org/10.1002/2016GL069405)
    - Representative of the local-tangent-plane convention used in regional studies.

### Multifractal & Intermittency

7. **She, Z. S., & Leveque, E. (1994).** "Universal dimensionality of intermittent structures in fully developed turbulence." *Physical Review Letters*, 72, 336.
   - Anomalous exponents and multifractal model.

---

## Related Topics in StructureFunctions.jl

- [Backends](backends.md): How to select the right computational backend for your data size
- [Architecture](architecture.md): How the library implements calculations
- [API Reference](../README.md#api-reference): Full function documentation
- [Examples](../examples/README.md): Worked examples
