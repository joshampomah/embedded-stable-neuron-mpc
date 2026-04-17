# Architecture

This is a code-level companion to the project report
([`doc/ampomah_4yp_closed_loop_dbs.pdf`](doc/ampomah_4yp_closed_loop_dbs.pdf)).
The report derives the controllers, the stable-neuron elimination, and the
float32 Mehrotra+PMM solver; this document walks the source that implements
them, from the motivation down to the individual file-and-line where each
design decision lives.

It is intentionally longer than the README. Section 6 on the QP solver
(PIQP → custom IPM) is built up from first principles so the later mapping
makes sense without prior IPM background.

- [1. Overview](#1-overview)
- [2. Problem & motivation](#2-problem--motivation)
- [3. Two controllers at a glance](#3-two-controllers-at-a-glance)
- [4. Predictor pipeline](#4-predictor-pipeline)
- [5. Stable-neuron elimination](#5-stable-neuron-elimination)
- [6. QP solver — from PIQP to the custom IPM](#6-qp-solver--from-piqp-to-the-custom-ipm)
- [7. Memory architecture](#7-memory-architecture)
- [8. Firmware & hardware-in-the-loop](#8-firmware--hardware-in-the-loop)
- [9. Design rationale](#9-design-rationale)
- [10. Further reading](#10-further-reading)

---

## 1. Overview

Closed-loop deep brain stimulation (DBS) modulates neural activity in
response to measured symptoms instead of running at a fixed amplitude.
The only adaptive DBS device currently on the market (Medtronic Percept
PC) reacts with simple threshold switching. Predictive model-predictive
control (MPC) could do better — if it fits the power and compute envelope
of an implantable pulse generator.

This repo demonstrates that it does, on an **STM32L476RG** (Cortex-M4F,
80 MHz, 128 KB SRAM, float32 FPU), and ships two controllers:

| Controller     | Prediction model                           | QP per step                  | Median step time |
|----------------|--------------------------------------------|------------------------------|------------------|
| **DCNN-MPC**   | DC decomposition of two ICNNs per horizon  | Stability-reduced + SCP loop | ~41 ms           |
| **Koopman-MPC**| Lifted-linear with 46 Lasso features       | Single dense QP              | ~9.5 ms          |

Both share the same embedded interior-point solver. They differ in model
capacity, compute cost, and the assumptions they make about how control
enters the dynamics.

Full derivations, patient-level results, cross-patient validation, and
on-target power measurements are in
[`doc/ampomah_4yp_closed_loop_dbs.pdf`](doc/ampomah_4yp_closed_loop_dbs.pdf).

## 2. Problem & motivation

DBS pulse generators are implanted in the chest and powered by a
primary-cell battery with a ~4-year lifetime. Any closed-loop controller
has to land inside that envelope. The Percept PC demonstrates that
reactive hardware is deployable; nothing on the market yet runs a
predictive controller.

The open question the project addresses is **feasibility**: can a model
that is expressive enough to anticipate dynamics (rather than react to
them) fit on the class of hardware already used for DBS? Luo et al.
(2020) showed Cortex-M4-class chips handle real-time neural control; the
contribution here is to show that predictive MPC specifically — which
normally means a QP with hundreds of variables — can be reduced to a
form that meets the same budget.

The core idea is simple: most of the complexity of the ICNN-based MPC QP
is dead weight because the rate-constrained input cannot actually exercise
most of the ReLU neurons. Eliminating those neurons shrinks the problem
roughly 20× and makes a bespoke float32 interior-point solver tractable
on the target.

## 3. Two controllers at a glance

### DCNN-MPC (ICNN + SCP)

**Prediction** is a DC decomposition:

```math
\hat{y}_{k+i} = f_{i,1}(z_k,\, u_{k:k+i-1}) - f_{i,2}(z_k,\, u_{k:k+i-1})
```

where each $`f_{i,k}`$ is an Input-Convex Neural Network (ICNN) with 32
hidden units. Non-negative internal weights + ReLU guarantee convexity,
and the DC form can express any smooth map (see report §modeling:dcnn).

**Control** is Successive Convex Programming (SCP): linearise $`f_{i,2}`$
at the current nominal $u$, solve a convex QP, iterate until convergence
(typically 1–2 iterations).

**Pipeline files:**

- `src/scp_controller.cpp` — outer SCP loop, warm-start, convergence checks
- `src/icnn_forward.cpp`, `src/jacobian.cpp` — forward pass and
  analytical Jacobian w.r.t. the control sequence
- `src/neuron_classifier.cpp` — interval-arithmetic classification of
  each neuron as stably-active, stably-inactive, or unstable
- `src/qp_builder.cpp` — reduced-variable QP assembly from
  classification output

### Koopman-MPC (lifted-linear + single QP)

**Prediction** is an affine map over a lifted state:

```math
\hat{y}_{k+i} = e_i + F_{i,:}\, u, \qquad e_i = C A_i\, \psi(z_k)
```

with $`\psi(z) = [z,\, \phi(z)]`$ and $\phi$ learned as a linear
projection of 46 Lasso-selected features on the 30-D raw state (see
report §modeling:koopman).

No SCP — one QP per timestep, well-conditioned because both primal blocks
of $P$ carry meaningful cost.

**Pipeline files:**

- `src/koopman_encoder.cpp` — the 46-feature lift
- `src/koopman_controller.cpp` — QP build and solve

### Shared backend

- `src/custom_ipm.cpp`, `include/stable_neuron_solver/custom_ipm.hpp` —
  the float32 Mehrotra + PMM interior-point solver both controllers use.
- `src/solver.cpp` — thin wrapper; selects PIQP (desktop debug) or the
  custom IPM (embedded) at compile time via `USE_CUSTOM_IPM`.

### When to pick which

| If you care about…              | Pick        |
|--------------------------------|-------------|
| Maximum modelling capacity      | DCNN-MPC    |
| Meeting a 20 ms step budget     | Koopman-MPC |
| Clean robustness story          | Koopman-MPC |
| Fitting nonlinear input-state coupling | DCNN-MPC |

The Koopman form is linear-in-control by construction, so it assumes the
stimulation enters the dynamics linearly. For this simulated patient
model, where DBS enters the $\log\beta$ dynamics linearly, that holds;
for in-vivo deployment where it does not, DCNN-MPC is the safer choice.

## 4. Predictor pipeline

The two controllers share nothing upstream of the QP; this section walks
both predictors.

### ICNN forward + Jacobian (DCNN path)

Each ICNN is a 3-layer network with free first/skip weights and
non-negative internal weights. Convexity comes from the combination of
non-negative weights and the ReLU activation. Inputs split into a state
part $z$ and a control part $u$; the Jacobian needed for SCP is
$\partial f / \partial u$, not $\partial f / \partial z$.

Two optimisations matter on the target:

- **Cached state-matvec.** The state-dependent contribution to each
  pre-activation ($c_0 = W_{0,z}\, z + b_0$ and the skip-const per layer)
  changes only once per MPC timestep. It is computed once in
  `NeuronClassifier::classify()` and reused by every forward and Jacobian
  call that timestep. See `src/icnn_forward.cpp:icnn_forward_cached` and
  `src/jacobian.cpp:icnn_forward_and_jacobian_u_cached`.
- **Sparse active-set GEMM.** The Jacobian chain rule only propagates
  through active ReLU rows. The cached-Jacobian variant maintains an
  `active_idx[]` list per layer and skips inactive rows in the $W_x B$
  update, saving ~50 % of FMAs and flash reads on a typical step
  (`src/jacobian.cpp:117-137` and `:169-190`).

### Koopman encoder

`src/koopman_encoder.cpp` computes a 46-element handcrafted feature
vector from $z = [y_{0:14},\, u_{0:14}]$:

- 14 signed first differences $y_{i+1} - y_i$
- $y_{\text{mean}}$, $u_{\text{sum}}$
- 7 absolute first differences at specific indices
- 9 absolute second differences at specific indices
- 12 quadratic $y$ products
- 1 cross-term $y_{12} \cdot u_{14}$
- $y_{\text{range}} = \max(y) - \min(y)$

The specific index sets were Lasso-selected on Ramsey-RESET residuals of
the linear model — see report §modeling:koopman for the derivation. The
comment at the top of `compute_features` points there.

The encoded state is then

```math
\psi(z) = [z,\; W_{\text{proj}}\, \text{features}(z) + b_{\text{proj}}]
```

which feeds directly into the Koopman QP build.

### Where Eigen still appears in the predictor

`src/qp_builder.cpp` uses Eigen for the per-layer symbolic epigraph
construction (e.g. line 534: `Matrix<Scalar, N_HIDDEN, Eigen::Dynamic> pre_coeffs = Wx * pvv`).
These allocations are small (`~N_HIDDEN × n_vars` floats), non-hot-path
(once per SCP iteration, not per IPM iteration), and the template surface
is narrow. The full PIQP → bespoke rewrite was driven by hot-path
constraints the QP builder doesn't hit; see §9.

## 5. Stable-neuron elimination

The central observation: with a tight rate constraint
$|\Delta u_k| \leq 0.0024$ at 50 Hz, interval-arithmetic bounds on each
pre-activation are narrow, and for most of the 640 ReLU neurons those
bounds do not straddle zero. A neuron whose pre-activation has the same
sign everywhere in the feasible input set is **stable**: its ReLU output
is either the pre-activation (always active) or zero (always inactive),
and never needs an epigraph variable in the QP.

**The full-size QP has two distinct numerical pathologies:**

1. $P$ is heavily rank-deficient — 640 of 670 variables are zero-cost
   epigraph terms, so $P$ has hundreds of zero eigenvalues. This is
   fixed by PMM's primal regulariser $\rho I$ (see §6.4) but still adds
   cost.
2. The reduced KKT matrix is genuinely ill-conditioned
   ($\kappa \sim 10^{11}$). Each ReLU constraint row is a composition of
   weight matrices through the network, and the singular-value spread of
   those rows compounds with depth.

Stable-neuron elimination does **not** improve $\kappa$ markedly — the
surviving unstable neurons inherit the layer composition through their
substituted constraint coefficients, so the per-row spread is largely
preserved. What it wins is **dimension**: a 30-variable KKT system makes
PIQP's proximal regularisation cheap enough to meet the real-time budget.

### Classification — `src/neuron_classifier.cpp`

Each neuron's pre-activation splits into a state-dependent constant and
a control-dependent affine term. Splitting each weight matrix into
positive and negative parts and propagating the rate-constrained interval
$[u_{\text{lo}},\, u_{\text{hi}}]$ through the layer gives tight
$[a_j^{\min},\, a_j^{\max}]$ in a single pass. The ICNN non-negativity
constraint $W_x \geq 0$ prevents interval blow-up through deeper layers.

A neuron is classified as:

- `STABLE_ACTIVE` if $a_j^{\min} > \varepsilon$ (ReLU is the identity everywhere)
- `STABLE_INACTIVE` if $a_j^{\max} < -\varepsilon$ (ReLU outputs zero everywhere)
- `UNSTABLE` otherwise (needs an epigraph variable)

where $\varepsilon = 10^{-4}$ is a numerical safety margin. The fused
`fused_wx_bounds` pass at `src/neuron_classifier.cpp:47-93` halves flash
reads by computing $a_j^{\min}$ and $a_j^{\max}$ in the same loop over
$W_u$ columns.

At 50 Hz typically >98 % of the 640 neurons are stable (median 7
unstable, max ~17).

### Symbolic expression propagation — `src/qp_builder.cpp`

Once classification is done, each layer's output can be written as an
affine map of the reduced variable vector $`x \in \mathbb{R}^{n_{\text{vars}}}`$:

```math
h_j = h_j^{\text{const}} + (h_j^{\text{coeff}})^\top x
```

Stably-active neurons contribute their pre-activation. Stably-inactive
neurons contribute zero. Unstable neurons get an epigraph variable
(written into $x$). Propagating layer by layer gives the final network
output as an affine map too.

The implementation uses **ping-pong buffers** (`src/qp_builder.cpp:128+`):
two layer-sized coefficient buffers alternate as "current" and "previous"
across layers, saving ~115 KB SRAM versus caching all layer coefficients
at once.

### Variable layout — `src/qp_builder.cpp:68-122`

The QP variable order is:

```
[u(N), s_max(N), s_min(N), t(N), y_f?(≤N), y_f_2?(≤N), h_amb_1, h_amb_2, ...]
```

Where:

- `u(N)` — control sequence (N=5 DCNN, N=7 Koopman)
- `s_max(N), s_min(N)` — tube slack bounds
- `t(N)` — tracking cost epigraph
- `y_f?` — one variable per (step, network) **only if** that network's
  last hidden layer contains at least one unstable neuron. Otherwise the
  output is affine in `u` and gets substituted directly into the tube
  cost, saving a variable.
- `h_amb_*` — one per unstable neuron, up to `MAX_UNSTABLE_VARS`

Beyond-cap unstable neurons are conservatively treated as
`STABLE_ACTIVE` (`unstable_var_idx_[...] = -1`), sacrificing some
tightness to guarantee bounded problem size. In practice, the cap is
large enough that this fallback rarely triggers.

Net effect: the 670-variable full QP reduces to ~30 variables at 50 Hz
(median 27-29, max ~40).

## 6. QP solver — from PIQP to the custom IPM

This section builds up interior-point methods from scratch, describes
the algorithm PIQP implements, explains why PIQP cannot ship on the
target directly, and maps each PIQP feature to the embedded reimplementation.

### 6.1 What PIQP solves

PIQP (Schwan et al. 2023) is a dense proximal-regularised interior-point
solver for the standard convex QP:

```math
\begin{aligned}
\min_{x} \quad & \tfrac{1}{2} x^\top P x + q^\top x \\
\text{s.t.} \quad & A x = b \\
& G x \leq h
\end{aligned}
```

with $P \succeq 0$. It is written in C++ using Eigen, supports both dense
and sparse backends, and is fast for small-to-medium problems.

### 6.2 Interior-point methods from scratch

**Inequality-constrained QP and KKT.** Introduce slacks $s \geq 0$ with
$G x + s = h$, dual multipliers $y$ for the equalities and $z \geq 0$
for the inequalities. The KKT conditions are:

```math
\begin{aligned}
P x + q + A^\top y + G^\top z &= 0 && \text{(stationarity)} \\
A x - b &= 0 && \text{(primal eq)} \\
G x + s - h &= 0 && \text{(primal ineq)} \\
S Z e &= 0 && \text{(complementarity)} \\
s,\, z &\geq 0
\end{aligned}
```

where $S = \operatorname{diag}(s)$, $Z = \operatorname{diag}(z)$, and
$e$ is the all-ones vector.

**Log-barrier and central path.** Replace the $s \geq 0$ constraint with
a log-barrier $-\mu \sum_i \log s_i$ added to the objective. For each
$\mu > 0$, stationarity is unchanged and complementarity relaxes to
$S Z e = \mu e$. The locus of solutions parameterised by $\mu$ is the
**central path**; as $\mu \to 0$, it converges to the true KKT solution.

**Perturbed Newton step.** At the current iterate, linearise the
perturbed KKT residual and solve for the update
$(\Delta x,\, \Delta s,\, \Delta y,\, \Delta z)$. For a QP this gives
the block system:

```math
\begin{bmatrix}
P + \rho I & 0 & A^\top & G^\top \\
0 & Z & 0 & S \\
A & 0 & 0 & 0 \\
G & I & 0 & 0
\end{bmatrix}
\begin{bmatrix} \Delta x \\ \Delta s \\ \Delta y \\ \Delta z \end{bmatrix}
= -
\begin{bmatrix} r_d \\ r_{sz} \\ r_p \\ r_{\text{ineq}} \end{bmatrix}
```

The $\rho I$ block is the PMM primal regulariser (covered in §6.4); set
$\rho = 0$ for the textbook version.

The centrality residual is $r_{sz} = S Z e - \sigma \mu e$ where
$\sigma \in [0, 1]$ is the centring parameter — $\sigma$ close to 0
takes an aggressive step toward the current guess of the KKT solution,
$\sigma$ close to 1 re-centres on the central path.

**Schur complement to $n \times n$.** The block system is too big to
factor directly. Eliminate $\Delta s$ from the fourth block row
($\Delta s = -r_{\text{ineq}} - G \Delta x$) and $\Delta z$ from the
second ($\Delta z = S^{-1}(r_{sz} - Z \Delta s)$) to get a reduced
system in $\Delta x$ alone:

```math
(P + \rho I + G^\top \Sigma G + \delta^{-1} A^\top A)\, \Delta x = r,
\qquad \Sigma = \operatorname{diag}(z_i / s_i)
```

For the stable-neuron-reduced problem $n \approx 30$, so this is a
dense $30 \times 30$ LDL$^\top$ factorisation — trivially cheap.
$\Delta s$ and $\Delta z$ are recovered by back-substitution afterwards.

**Fraction-to-boundary step.** After solving for
$(\Delta x,\, \Delta s,\, \Delta z)$, the update is
$(x, s, z) \leftarrow (x, s, z) + \alpha \cdot (\Delta x, \Delta s, \Delta z)$
with $\alpha \in (0, 1]$ chosen so that $s + \alpha \Delta s \geq 0$ and
$z + \alpha \Delta z \geq 0$ strictly. A common choice is

```math
\alpha = \tau \cdot \max\{\alpha' \in (0, 1] : s + \alpha' \Delta s \geq 0,\; z + \alpha' \Delta z \geq 0\}
```

with $\tau = 0.99$.

**Progress.** The duality gap $\mu = s^\top z / m$ (where $m$ is the
number of inequalities) decreases roughly geometrically; the solver
stops when primal, dual, and centrality residuals all fall below a
tolerance.

### 6.3 Mehrotra predictor-corrector

The textbook IPM has to pick $\sigma$ before each step without knowing
how far the affine direction will progress. Mehrotra's observation
(1992) is that two solves per factorisation gives a much better
$\sigma$ — and since the factorisation is the expensive part, the
second solve is nearly free.

**Predictor.** Solve the system with $\sigma = 0$ (purely affine direction):

```math
K\, \Delta x^{\text{aff}} = r^{\text{aff}} \quad \longrightarrow \quad (\Delta s^{\text{aff}},\, \Delta z^{\text{aff}}) \; \text{by back-sub}
```

Compute the predicted step sizes $\alpha_s^{\text{aff}},\, \alpha_z^{\text{aff}}$
from the fraction-to-boundary rule, and the predicted gap

```math
\mu_{\text{aff}} = \frac{(s + \alpha_s^{\text{aff}} \Delta s^{\text{aff}})^\top (z + \alpha_z^{\text{aff}} \Delta z^{\text{aff}})}{m}.
```

**Heuristic.** If the affine step made good progress
($\mu_{\text{aff}} \ll \mu$), trust it: take $\sigma$ small. If not,
re-centre: take $\sigma$ near 1. Mehrotra's cubic rule:

```math
\sigma = \left(\frac{\mu_{\text{aff}}}{\mu}\right)^3
```

**Corrector.** The same $K$ is used with a new RHS that folds in both
the centring target $\sigma \mu$ and a second-order correction
$-\Delta s^{\text{aff}} \circ \Delta z^{\text{aff}}$ that cancels the
$S Z e$ curvature the affine step couldn't see:

```math
r^s_{\text{corr}} = -S Z e - \Delta s^{\text{aff}} \circ \Delta z^{\text{aff}} + \sigma \mu e
```

Because $K$ is unchanged, the same LDL$^\top$ factor handles both
solves. Net: two solves per factor $\approx$ halves the iteration count
compared with a single-direction method.

### 6.4 PMM regularisation

Without regularisation, the textbook IPM has two failure modes on this
problem:

**Singular $P$.** The DCNN QP has 640 zero-cost epigraph variables; $P$
has hundreds of zero eigenvalues, and without help
$K = P + G^\top \Sigma G$ is rank-deficient when few constraints are
active. PMM (Pougkakiotis & Gondzio 2021) adds a primal regulariser
$\rho I$ that keeps $K$ positive definite. The bias vanishes as
$\rho \to 0$, so tightness is recoverable.

**Barrier-weight blowup.** As a constraint activates ($s_i \to 0$), the
standard barrier weight $w_i = z_i / s_i$ diverges, blowing the
condition number of $G^\top \Sigma G$. PMM replaces it with

```math
w_i = \frac{z_i}{s_i + \delta\, z_i}, \qquad w_i \leq \frac{1}{\delta}.
```

Two benefits: $w_i$ is bounded, and the regularisation is dual to the
primal $\rho I$ in the PMM framework. As $\delta \to 0$, the bounded
weight recovers the unbounded $z_i / s_i$, so at convergence the
original KKT conditions are satisfied.

**How regularisation is scheduled here.** $\delta$ decays with the
complementarity gap each iteration ($\delta \mathrel{{*}{=}} (1 - \mu_{\text{rate}})$ at
`src/custom_ipm.cpp:968-983`), so the bounded-weight perturbation
vanishes at convergence (`src/custom_ipm.cpp:964-979`). $\rho$ is held
fixed at `rho_init` (1e-4 on the target, 1e-8 on desktop) rather than
decayed: it is already small relative to the problem scale, and driving
it toward zero would risk LDL$^\top$ pivots falling below the float32
clamp (see §6.6). The residual $\rho I$ bias is below the IPM's
stopping tolerance and does not affect the converged solution.

### 6.5 Why PIQP cannot ship on Cortex-M4F

PIQP is excellent for what it is — a double-precision, Eigen-based,
heap-allocating solver. Three dependencies prevent direct use on the
STM32L476RG, and none are isolated patches:

1. **Dynamic memory.** Eigen uses heap allocation (`malloc`). The
   firmware is bare-metal with no heap; it has a bump allocator in 32 KB
   of SRAM2 (`firmware/board.cpp`). `Eigen::MatrixX`-sized allocations
   expect `new`/`malloc`, which doesn't exist.
2. **Double precision.** PIQP assumes `double`. The Cortex-M4F FPU is
   float32-only — every `double` operation falls back to software
   emulation at roughly 10× the cost. Retyping is not one line, because
   of float32-specific numerical pathologies (see §6.6).
3. **Code size.** Eigen's template instantiation alone blows the flash
   budget of the target (~512 KB). Even minimal use drags in large
   amounts of generated code.

These are deep structural dependencies, not isolated calls that can be
patched — hence the full ground-up rewrite in `src/custom_ipm.cpp`.

### 6.6 Mapping PIQP → `CustomIPM`

The custom IPM implements exactly the PIQP algorithm — same Mehrotra
predictor-corrector, same PMM bounded-weight regularisation — but on
plain C-style loops over static arrays with float32 arithmetic. The
table below maps each PIQP feature to its location in the embedded code.

| Feature                                    | File                                                  | Lines        | Note |
|--------------------------------------------|-------------------------------------------------------|--------------|------|
| Problem form + algorithm overview          | `include/stable_neuron_solver/custom_ipm.hpp`         | 1-10         | Header comment spells out the exact standard form. |
| Row-major → col-major, symmetrise $P$, `P_REG` | `src/custom_ipm.cpp:store_problem`                | 111-166      | Also pairs equalities into inequalities (see §9). |
| Ruiz equilibration                         | `src/custom_ipm.cpp:ruiz_equilibrate`                 | 168-253      | On by default for Koopman (`ruiz_iter = 10`); off for DCNN. |
| $G$ sparsity analysis (CSC + CSR)          | `src/custom_ipm.cpp:analyze_sparsity`                 | 255-304      | CSC for $G^\top z$ / $Gx$; CSR for the $G^\top \Sigma G$ rank-1 update. |
| Mehrotra warm-start $x_0 = -\operatorname{diag}(P)^{-1} q$ | `src/custom_ipm.cpp:initialize`        | 327-339      | Standard IPM cold-start; $s$, $z$ shifted positive. |
| Residuals + duality gap $\mu$              | `src/custom_ipm.cpp:compute_residuals`                | 381-440      | Fused $G^\top z$ + $Gx$ pass using CSC. |
| KKT build $K = P + \rho I + G^\top \Sigma G$ | `src/custom_ipm.cpp:build_kkt`                      | 442-515      | Row-wise rank-1 via CSR; accumulation order preserves float32 precision. |
| Bounded-$w$ clamp $w_i \geq 10^{-8}$       | `src/custom_ipm.cpp`                                  | 462-463      | Float32 fix — prevents barrier-weight overflow collapsing the LDL$^\top$ diagonal. |
| LDL$^\top$ factor + pivot clamp $D_j \geq 10^{-6}$ | `src/custom_ipm.cpp:ldlt_factor`              | 517-554      | Caches $D^{-1}$ and $D L$ per column; cuts `VDIV` out of the inner loop (clamp at line 524). |
| LDL$^\top$ solve                           | `src/custom_ipm.cpp:ldlt_solve`                       | 556-583      | Forward + diagonal + backward; register accumulator in backward pass. |
| Iterative refinement                       | `src/custom_ipm.cpp:ldlt_solve_refined`               | 622-644      | One correction solve via `kkt_matvec`; disabled on embedded (speed). |
| Predictor-step KKT RHS                     | `src/custom_ipm.cpp`                                  | 759-810      | Consistent with bounded-$w$: $\text{tmp} = w (r_{\text{ineq}} - s)$. |
| Mehrotra $\sigma = (\mu_{\text{aff}} / \mu)^3$ | `src/custom_ipm.cpp`                              | 849-851      | Cubic centring rule. |
| Corrector with factor reuse                | `src/custom_ipm.cpp`                                  | 853-903      | "$K$ is the same, so we REUSE the LDL$^\top$ factorization" — comment at 857. |
| Hybrid dual recovery                       | `src/custom_ipm.cpp`                                  | 929-935      | $s_i \geq z_i$ → standard form (÷ $s_i$); $z_i > s_i$ → bounded form (÷ $z_i$). |
| Fraction-to-boundary step size             | `src/custom_ipm.cpp:step_size`                        | 646-661      | NaN-safe. |
| $\delta$ scheduling, $\rho$ fixed          | `src/custom_ipm.cpp`                                  | 964-979      | PIQP rule for $\delta$; $\rho$ held at `rho_init` (see §9). |
| `-fno-fma` on LDL$^\top$ path only         | `CMakeLists.txt`                                      | 138-160      | `custom_ipm.cpp` deliberately excluded from the FMA re-enable list. |

Two of these rows are genuine algorithmic additions beyond textbook PIQP
— forced by float32:

- **Barrier-weight clamp ($w_i \geq 10^{-8}$).** Even with the bounded
  weight $w_i = z_i / (s_i + \delta z_i)$, slacks near the float32 floor
  create weights large enough to collapse the LDL$^\top$ diagonal
  $D_j$. Clamping $w_i$ from below and $D_j$ from below
  ($D_j \geq 10^{-6}$) keeps the factorisation stable.
- **Hybrid dual recovery.** PIQP's standard recovery
  $\Delta z_i = (r^s_i - z_i \Delta s_i) / s_i$ divides by a small number
  as a constraint activates. The analytically equivalent form
  $\Delta z_i = w_i (-\Delta s_i + r^s_i / z_i)$ divides by $z_i$
  instead, which is safe in the opposite regime. The code picks the
  branch per constraint based on whichever of $s_i$, $z_i$ is larger.

Together these raise SCP convergence from ~60 % in the naïve float32
port to 99.97 % on-target (see report table §ipm:float32).

## 7. Memory architecture

From report §memory, realised as follows:

| Region   | Size      | Contents |
|----------|-----------|----------|
| Flash    | ~130 KB   | ICNN/Koopman weights — structures in `include/stable_neuron_solver/weights.hpp`, bound at link time from `generated/network_weights.hpp` and `generated/koopman_weights.hpp` |
| SRAM1    | ~67 KB    | Static arrays (all `MAX_VARS`/`MAX_INEQ`-sized, fixed at compile time — see `include/stable_neuron_solver/config.hpp`) |
| SRAM2    | ~32 KB    | IPM workspace via bump allocator (`firmware/board.cpp`) |

Peak scratch use is cut from a naïve 128 KB to 13 KB (12.5×) by the
**ping-pong buffer** scheme in `src/qp_builder.cpp` (declared at
`qp_builder.hpp:123-126`): two `N_HIDDEN × MAX_VARS` buffers alternate as
"current" and "previous" layer during symbolic propagation, so only two
layers' worth of coefficients are live at once instead of all three.

Everything is statically sized at compile time (`MAX_VARS = 34`,
`MAX_AMB = 10`, `MAX_EQ`, `MAX_INEQ`), enforced by the build budget and
by the IPM's reliance on fixed arrays with no dynamic allocation.

## 8. Firmware & hardware-in-the-loop

- `firmware/board.cpp` — STM32L4 init: HSI+PLL to 80 MHz, USART2 for
  host I/O, DWT cycle counter for microsecond timing, LED, bump
  allocator (`heap_reset_to_baseline` for per-solve scratch reset).
- `firmware/startup.cpp` — ARM Cortex-M4 startup vectors and reset
  handler.
- `firmware/main.cpp` / `firmware/main_koopman.cpp` — HIL test harnesses.
  Each runs 100 solves against a reference test case
  (`firmware/reference_test_case.hpp` / `firmware/koopman_reference_test_case.hpp`
  — synthetic, see `DISCLAIMER.md`), reports per-phase timing over UART,
  and confirms pass/fail against expected controls.
- `test/hil_main.cpp` / `test/hil_koopman.cpp` — host-side HIL binaries.
  Read state frames from stdin, return optimal controls on stdout.
  Used by the Python tooling in `python/embedded_mpc/` for closed-loop
  simulation.

## 9. Design rationale

A handful of choices in the code aren't obvious from reading the source
alone. Each is non-standard enough to surprise a reviewer, so the
reasoning is set out here.

### Eigen stayed in `qp_builder.cpp` but was cut from the IPM

The three constraints that ruled out Eigen for the IPM apply less forcefully
to the QP builder:

- Its dynamic allocations are small ($\sim N_{\text{hidden}} \times n_{\text{vars}}$
  floats, a few KB at most) and can be serviced by the bump allocator.
- It is not in the per-iteration hot loop — QP build runs once per SCP
  iteration, not per IPM iteration.
- Its template surface is narrow (`Map`, one `GEMM`), so the flash
  footprint is bounded.

Full rewrite of `src/qp_builder.cpp` would be considerable work for a
proportionally small win. (`src/qp_builder.cpp:493-534`.)

### `-fno-fma` only on `custom_ipm.cpp`

The Cortex-M4F `VFMA.F32` fused multiply-add is both faster (1 cycle vs 2)
and more accurate (single rounding) — it's the right choice almost
everywhere. The LDL$^\top$ diagonal update

```math
D_j = K_{jj} - \sum_{k < j} L_{jk}^2\, D_k
```

is the one place where different rounding of the multiply-then-add
matters in float32: the updated diagonal diverges by ULPs that compound
through later columns. Compiling `custom_ipm.cpp` with default
fp-contract (no FMA) and leaving everything else at `-ffp-contract=fast`
recovers the speed where it is safe and the stability where it matters.
(`CMakeLists.txt:147-156`.)

### Equalities paired to inequalities

PIQP's PMM handles equalities via $\delta^{-1} A^\top A$ in the KKT
matrix. When $\delta$ is small this block is very large — and because
it is a rank-$p$ contribution (with $p$ the number of equalities)
repeated every iteration, it keeps the KKT ill-conditioned even when
the problem isn't. The workaround: convert $A x = b$ to
$A x \leq b \land -A x \leq -b$ before handing the problem to the IPM.
The solver then only sees inequalities, and $\delta^{-1} A^\top A$
disappears. (`src/custom_ipm.cpp:119-166`; the equality branches
elsewhere are dead code at runtime — noted at `:427`.)

### CSC + CSR for $G$, not one or the other

- `compute_residuals` needs $G x$ and $G^\top z$ — both column-natural,
  so CSC.
- `build_kkt` does $G^\top \Sigma G$ as a row-wise rank-1 sum
  $\sum_i w_i\, g_i g_i^\top$ — each iteration needs all columns of one
  $G$ row, so CSR.

Storing both is cheap (`MAX_G_NNZ = 1024` ints × 2 arrays for row/col
indices) and avoids awkward indexing inside the two hottest loops.
(`src/custom_ipm.cpp:259-308`.)

### $\delta$ decays but $\rho$ is held fixed

See §6.4. At convergence the bounded-weight perturbation must vanish,
so $\delta$ has to decay. The primal regulariser $\rho$ is already
small relative to the problem scale (`rho_init = 1e-4` on target), and
further reduction would risk LDL$^\top$ pivots dropping below the
float32 clamp $D_j \geq 10^{-6}$, collapsing the factorisation. The
residual bias from a fixed $\rho$ is below the IPM's stopping
tolerance. (`src/custom_ipm.cpp:964-979`.)

### Always cold-start the IPM; warm-start primal $x$ only

Fully warm-starting the IPM (reusing $s$, $z$ across MPC timesteps)
accumulated float32 errors that eventually crashed the solver. The
current scheme re-runs `initialize()` each solve and uses the previous
$x$ only as a starting point; slacks $s = h - G x$ (shifted positive)
and duals $z = \mathbf{1}$ are rebuilt from the new problem each time.
This is essentially free compared to the factor+solve cost and
eliminates an entire class of numerical-drift failures.
(`src/solver.cpp:135-178`, `src/custom_ipm.cpp:warm_start_x`.)

### CDC25 reference

`include/stable_neuron_solver/scp_controller.hpp:92` — "Implements
Algorithm 1 from CDC25" is a pointer to the associated paper submission,
not the 4YP report. If you want the derivation, that paper is the
primary reference for the SCP algorithm itself; the 4YP report expands
on the float32/embedded adaptation.

## 10. Further reading

- **Full derivations, patient data protocol, cross-patient validation,
  on-target power measurements:** `doc/ampomah_4yp_closed_loop_dbs.pdf`.
- **PIQP — the reference solver:** Schwan, Jiang, Kuhn, Jones (2023).
  *PIQP: A Proximal Interior-Point Quadratic Programming Solver.* IEEE
  CDC 2023.
- **Mehrotra predictor-corrector:** Mehrotra (1992). *On the
  Implementation of a Primal-Dual Interior Point Method.* SIAM J. Optim.
- **PMM framework:** Pougkakiotis & Gondzio (2021). *An Interior Point-Proximal
  Method of Multipliers for Convex Quadratic Programming.*
  Computational Optimization and Applications.
- **ICNN model class:** Amos, Xu, Kolter (2017). *Input Convex Neural
  Networks.* ICML.
- **Koopman MPC lifting:** Korda & Mezić (2018). *Linear predictors for
  nonlinear dynamical systems: Koopman operator meets model predictive
  control.* Automatica.

All are cited from the 4YP report.
