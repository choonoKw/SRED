# SRED
For a fixed receive filter, we can reformulate the optimization problem with respect to $\mathbf{s}$ as follows:

```math
\max_{\mathbf{s}\in{\mathcal{S}^{N_tN}}}\ \min_{\forall m}\ \text{SINR}_m(\mathbf{s},\mathbf{w}_{m,\text{opt}})=\frac{\mathbf{s}^H\mathbf{H}_m(\mathbf{w}_{m,\text{opt}})\mathbf{s}}{\mathbf{s}^H\mathbf{G}_m(\mathbf{w}_{m,\text{opt}})\mathbf{s}}
```


As we solve the problem above with fixed $`\mathbf{w}_m\ \forall m`$, we simplify $`\mathbf{G}_m=\mathbf{G}_m(\mathbf{w}_m)`$ and $`\mathbf{H}_m=\mathbf{H}_m(\mathbf{w}_m)`$ for brevity from this section.
To enforce a waveform to satisfy the constant modulus constraint (CMC), we aim to optimize the phase variable $`\boldsymbol{\phi}`$ such that $`\boldsymbol{\phi}=[\phi_1,...,\phi_{N_tN}]^T`$ and $`e^{j\phi_i}=\sqrt{N_tN}s_i`$ for $`i=1,...,N_tN`$. Therefore, any change in $`\boldsymbol{\phi}`$ still guarantees the CMC. Additionally, we reformulate the problem above as following with a lemma:

### Lemma
The local minima of the following problem such that $\mathbf{s}^H\mathbf{G}_m\mathbf{s}\neq 0$ and $\mathbf{s}^H\mathbf{H}_m\mathbf{s}\neq 0$ are equivalent to the ones of the problem above.
```math
    \min_{\mathbf{s}\in{\mathcal{S}^{N_tN}}}\ \max_{\forall m}\ \frac{1}{\text{SINR}_m(\mathbf{s},\cdot)}=\frac{\mathbf{s}^H\mathbf{G}_m\mathbf{s}}{\mathbf{s}^H\mathbf{H}_m\mathbf{s}}.
```

However, minimax problems, especially in nonconvex setting, is hard to solve directly. To address this minimax problem, we define a new cost function $f(\boldsymbol{\phi})$ as the *sum-of-reciprocals* of SINRs, thereby, reformulate the minimax problem into the following minimization problem:

```math
\begin{split}
    \min_\boldsymbol{\phi} f(\boldsymbol{\phi})=\sum_{m=1}^Mf_m(\boldsymbol{\phi})
    &=\sum_{m=1}^M\frac{1}{	\text{SINR}_m\left(\mathbf{s}(\boldsymbol{\phi}),\cdot\right)}\\
    &=\sum_{m=1}^M\frac{\mathbf{s}^H(\boldsymbol{\phi})\mathbf{G}_m\mathbf{s}(\boldsymbol{\phi})}
    {\mathbf{s}^H(\boldsymbol{\phi})\mathbf{H}_m\mathbf{s}(\boldsymbol{\phi})}.
\end{split}

```
where $`f_m(\boldsymbol{\phi})`$ is the reciprocal of $`\text{SINR}_m(\mathbf{s}(\boldsymbol{\phi}),\cdot)`$. The choice of this new cost function is motivated from the efficiency of utilizing its exact descent (negative gradient) to maximize the worst-case SINR, which is described in the following section.

## Exact-Gradient of the Sum-of-Reciprocals

The motivations of formulating new problem in the lemma is 1) minimizing $`f(\boldsymbol{\phi})`$ can lead maximization of overall SINRs, and 2) the gradient of the cost function can guide enhancing the worst case of SINR among $K$ targets analytically. Expressing $`\mathbf{s}=\mathbf{s}(\boldsymbol{\phi})`$ for brevity, the gradient of the cost function becomes 
```math
\begin{split}
    &\nabla_\boldsymbol{\phi} f(\boldsymbol{\phi})=\sum_{m=1}^M \nabla_\boldsymbol{\phi} f_m(\boldsymbol{\phi})\\
    &=\sum_{m=1}^M\frac{-1}{\{	\text{SINR}_m(\mathbf{s},\mathbf{w}_m)\}^2}
    \nabla_\boldsymbol{\phi} 	\text{SINR}_m(\mathbf{s},\mathbf{w}_m),
\end{split}
```
which is weighted-some of $`\nabla_\boldsymbol{\phi}\text{SINR}_m(\cdot)`$ for $`m=1,...,M`$. Notably, the net descent direction is dominated by $`\nabla_\boldsymbol{\phi}\text{SINR}_{m'}(\cdot)`$ such that $`\text{SINR}_{m'}(\cdot) \geq \text{SINR}_m(\cdot)`$ for $`m'\neq m\ \forall m`$. 

Following the chain rule of the partial derivatives, we can deduce the following.
```math
    \nabla_\boldsymbol{\phi} f_m(\cdot) = 
    \frac{\mathbf{s}^H\mathbf{H}_m\mathbf{s}\left\{\nabla_\boldsymbol{\phi} \mathbf{s}^H\mathbf{G}_m\mathbf{s}\right\}
    - \mathbf{s}^H\mathbf{G}_m\mathbf{s}\left\{\nabla_\boldsymbol{\phi} \mathbf{s}^H\mathbf{H}_m\mathbf{s}\right\}}
    {\left\{\mathbf{s}^H\mathbf{H}_m\mathbf{s}\right\}^2},
```
and we show the following lemma about the first exact gradient of the SINR with respect to phase-code to the best of our knowledge.

### Lemma
The exact gradient of $f_m(\boldsymbol{\phi})$ with respect to the phase-code vector $\boldsymbol{\phi}$ is given by:
```math
\begin{split}
    &\nabla_\boldsymbol{\phi} f_m(\boldsymbol{\phi})\\
    &=\beta_m\text{Im}\left[\left\{
    (\mathbf{s}^H \mathbf{H}_m \mathbf{s})\mathbf{G}_m\mathbf{s}-(\mathbf{s}^H \mathbf{G}_m \mathbf{s})\mathbf{H}_m\mathbf{s}\right\}\odot\mathbf{s}^*\right].
\end{split}
```
where $\beta_m=\frac{2}{(\mathbf{s}^H \mathbf{G}_m \mathbf{s})^2}\in\mathbb{C}$.


Using the *exact descent* direction of the *sum-of-reciprocal* SINR on the feasible set, *i.e.*, $`-\nabla_\boldsymbol{\phi} f_m(\boldsymbol{\phi})`$, we update $`\boldsymbol{\phi}`$ following $`\boldsymbol{\phi}^{t+1}=\boldsymbol{\phi}^{t}-\rho^t\nabla_\boldsymbol{\phi} f(\boldsymbol{\phi})`$ where $`t`$ is the iteration in the $`\phi`$ update, and $`\rho^t`$ is the step size at $`t^\text{th}`$ iteration. This iterative algorithm is dubbed as SRED. SRED ensures convergence to a feasible local because it updates the phase of the CMC waveforms directly without any relaxation.
