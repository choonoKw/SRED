# SRED
As we solve \eqref{eq:prob_sinr_s} with fixed $\mathbf{w}_m\ \forall m$, we simplify $\mathbf{G}_m=\mathbf{G}_m(\mathbf{w}_m)$ and $\mathbf{H}_m=\mathbf{H}_m(\mathbf{w}_m)$ for brevity from this section.
To enforce a waveform to satisfy the \gls{cmc} in \eqref{eq:prob_sinr_s}, we aim to optimize the phase variable $\bsphi$ such that $\bsphi=[\phi_1,...,\phi_{N_tN}]^T$ and $e^{j\phi_i}=\sqrt{N_tN}s_i$ for $i=1,...,N_tN$. Therefore, any change in $\bsphi$ still guarantees the \gls{cmc}. Additionally, we reformulate the problem in \eqref{eq:prob_sinr_s} as following with a lemma:
\begin{lemma}\label{lemma:minimax}
    The local minima of the following problem such that $\bs^H\mathbf{G}_m\bs\neq 0$ and $\bs^H\mathbf{H}_m\bs\neq 0$ are equivalent to the ones of the problem in \eqref{eq:prob_sinr_s}.
    \begin{equation}\label{eq:prob_minimax}
        \min_{\bs\in{\cS^{N_tN}}}\ \max_{\forall m}\ \frac{1}{\sinr_m(\bs,\cdot)}
        =\frac{\bs^H\mathbf{G}_m\bs}{\bs^H\mathbf{H}_m\bs}.
    \end{equation}
\end{lemma}

\begin{proof}
    See \cref{sec:proof_sum_of_reciprocal} of \cite{kweon2023technical}.
\end{proof}

However, minimax problems, especially in nonconvex setting, is hard to solve directly \cite{razaviyayn2020nonconvex, zhang2022sapd+,du1995minimax}. To address the minimax problem in \cref{eq:prob_sinr_s}, we define a new cost function $f(\bsphi)$ as the \textit{sum-of-reciprocals} of \glspl{sinr}, thereby, reformulate the minimax problem into the following minimization problem:
% shown in \eqref{eq:prob_sinr_s}:
\begin{equation}
\begin{split}
    \min_\bsphi f(\bsphi)=\sum_{m=1}^Mf_m(\bsphi)
    &=\sum_{m=1}^M\frac{1}{\sinr_m\left(\bs(\bsphi),\cdot\right)}\\
    &=\sum_{m=1}^M\frac{\bs^H(\bsphi)\mathbf{G}_m\bs(\bsphi)}
    {\bs^H(\bsphi)\mathbf{H}_m\bs(\bsphi)}.
\end{split}
\label{eq:fphi}   
\end{equation}
where $f_m(\bsphi)$ is the reciprocal of $\sinr_m(\bs(\bsphi),\cdot)$. The choice of this new cost function is motivated from the efficiency of utilizing its exact descent (negative gradient) to maximize the worst-case \gls{sinr}, which is described in the following section.

\subsection{Exact-Gradient of the Sum-of-Reciprocals}

The motivations of formulating \eqref{eq:fphi} is 1) minimizing $f(\bsphi)$ can lead maximization of overall \gls{sinr}s, and 2) the gradient of the cost function can guide enhancing the worst case of \gls{sinr} among $K$ targets analytically. Expressing $\bs=\bs(\bsphi)$ for brevity, the gradient of the cost function becomes 
\begin{equation}\label{eq:grad_f_phi}
\begin{split}
    &\nabla_\bsphi f(\bsphi)=\sum_{m=1}^M \nabla_\bsphi f_m(\bsphi)\\
    &=\sum_{m=1}^M\frac{-1}{\{\sinr_m(\bs,\mathbf{w}_m)\}^2}
    \nabla_\bsphi \sinr_m(\bs,\mathbf{w}_m),
\end{split}
\end{equation}
which is weighted-some of $\nabla_\bsphi \sinr_m(\cdot)$ for $m=1,...,M$. Notably, the net descent direction is dominated by $\nabla_\bsphi \sinr_{m'}(\cdot)$ such that $\sinr_{m'}(\cdot)\geq\sinr_m(\cdot)$ for $m'\neq m\ \forall m$. 

Following the chain rule of the partial derivatives, we can deduce the following.
\begin{equation*}
    \nabla_\bsphi f_m(\cdot) = 
    \frac{\bs^H\mathbf{H}_m\bs\left\{\nabla_\bsphi \bs^H\mathbf{G}_m\bs\right\}
    - \bs^H\mathbf{G}_m\bs\left\{\nabla_\bsphi \bs^H\mathbf{H}_m\bs\right\}}
    {\left\{\bs^H\mathbf{H}_m\bs\right\}^2},
\end{equation*}
and we show the following lemma about the first exact gradient of the \gls{sinr} with respect to phase-code to the best of our knowledge.

\begin{lemma}\label{lemma:grad}
    The exact gradient of $f_m(\bsphi)$ with respect to the phase-code vector $\bsphi$ is given by:
    \begin{equation}\label{eq:grad_f_phi_m}
    \begin{split}
        &\nabla_\bsphi f_m(\bsphi)\\
        &=\beta_m\imag\left[\left\{
        (\bs^H \mathbf{H}_m \bs)\mathbf{G}_m\bs-(\bs^H \mathbf{G}_m \bs)\mathbf{H}_m\bs\right\}\odot\bs^*\right].
    \end{split}
    \end{equation}
    where $\beta_m=\frac{2}{(\bs^H \mathbf{G}_m \bs)^2}\in\bbC$.
\end{lemma}

\begin{proof}
    See \cref{sec:derive_grad} of \cite{kweon2023technical} for the detailed derivation of the gradient.
\end{proof}

Using the \textit{exact descent} direction of the \textit{sum-of-reciprocal} \gls{sinr} on the feasible set, \ie, $-\nabla_\bsphi f_m(\bsphi)$, we update $\bsphi$ following $\bsphi^{t+1}=\bsphi^{t}-\rho^t\nabla_\bsphi f(\bsphi)$ where $t$ is the iteration in the $\phi$ update, and $\rho^t$ is the step size at $t\th$ iteration. This iterative algorithm is dubbed as \gls{sred}. \gls{sred} ensures convergence to a feasible local because it updates the phase of the \gls{cmc} waveforms directly without any relaxation. 
