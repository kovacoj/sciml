# Harmonic Oscillator Weak Formulation

## Problem Statement

We consider the one-dimensional quantum harmonic oscillator:
$$ \left(-\frac{1}{2}\frac{d^2}{dx^2}+\frac{1}{2}x^2\right)\psi_n=E_n\psi_n $$

## Weighted Sobolev Space

The correct energy space for the whole-line problem ($\Omega=\mathbb{R}$) is the weighted Sobolev space:
$$ V := \{u \in H^1(\mathbb{R}) : xu \in L^2(\mathbb{R})\} = H^1(\mathbb{R}) \cap L^2(\mathbb{R}; x^2 dx) $$

This is necessary because $x^2$ is unbounded on $\mathbb{R}$, and $H^1(\mathbb{R})$ alone is insufficient. For example, $u(x) = \frac{1}{\sqrt{1+x^2}} \in H^1(\mathbb{R})$ but $\int_{\mathbb{R}} x^2 u(x)^2 dx = \infty$.

**Norm:** $|u|_V^2 = |u|_{L^2}^2 + |u'|_{L^2}^2 + |xu|_{L^2}^2$

Note: The $L^2$ term is controlled by the other two via the harmonic-oscillator inequality:
$$ |u|_{L^2}^2 \leq |u'|_{L^2}^2 + |xu|_{L^2}^2 $$

Thus an equivalent norm is: $|u|_a^2 = |u'|_{L^2}^2 + |xu|_{L^2}^2$

## Weak Formulation

**Find** $u \in V$ such that:
$$ a(u,v) = \ell(v) \quad \forall v \in V $$

where:
- $a(u,v) = \int_{\mathbb{R}} u'v' \, dx + \int_{\mathbb{R}} x^2 uv \, dx$
- $\ell(v) = 2\langle f, v \rangle_{V^*, V}$ (with $f \in V^*$ or $f \in L^2(\mathbb{R})$)

## Lax-Milgram Verification

### 1. $V$ is a Hilbert Space
- **Completeness:** If $(u_n)$ is Cauchy in $V$, then $u_n \to u$ in $H^1(\mathbb{R})$ and $xu_n \to g$ in $L^2(\mathbb{R})$. Local convergence implies $g = xu$, so $xu \in L^2(\mathbb{R})$ and $u \in V$.

### 2. Continuity of $a$
$$ |a(u,v)| \leq |u'|_{L^2}|v'|_{L^2} + |xu|_{L^2}|xv|_{L^2} \leq |u|_V |v|_V $$

### 3. Coercivity of $a$
$$ a(u,u) = |u'|_{L^2}^2 + |xu|_{L^2}^2 \geq \frac{1}{2}|u|_V^2 $$

Coercivity constant: $\alpha = \frac{1}{2}$

### 4. Continuity of $\ell$
$$ |\ell(v)| = 2|\langle f, v \rangle| \leq 2|f|_{V^*}|v|_V $$

## Uniqueness Proof

If $u_1, u_2 \in V$ are solutions, let $w = u_1 - u_2$:
$$ \int_{\mathbb{R}} |w'|^2 dx + \int_{\mathbb{R}} x^2|w|^2 dx = 0 $$

Both integrals are nonnegative, so $w' = 0$ a.e. and $xw = 0$ a.e. Thus $w$ is constant and $w = 0$ a.e. (since only the zero constant is in $L^2(\mathbb{R})$).

## Boundary Conditions and Integration by Parts

- Start with $v \in C_c^\infty(\mathbb{R})$ to avoid boundary terms
- Extend to all $v \in V$ by density and continuity
- Do not claim pointwise vanishing at infinity for arbitrary $H^1$ functions

**Important:** While $u(x) \to 0$ as $x \to \pm\infty$ for $u \in H^1(\mathbb{R})$, this does NOT imply $u'(x) \to 0$. An $L^2$ function can have narrow spikes preventing pointwise convergence.

## Bounded vs Unbounded Domain

| Domain | Energy Space |
|--------|--------------|
| $\Omega = (-L, L)$ | $V = H^1(\Omega)$ or $H_0^1(\Omega)$ |
| $\Omega = \mathbb{R}$ | $V = \{u \in H^1(\mathbb{R}) : xu \in L^2(\mathbb{R})\}$ |

On bounded domains, $x^2$ is bounded, so $H^1$ suffices. On $\mathbb{R}$, the weighted space is essential.
