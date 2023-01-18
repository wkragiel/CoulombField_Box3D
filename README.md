# Method of calculating Coulomb potential on lattice
Charge distribution $e\rho(\mathbf{r})$:
$$\nabla^2 \Phi(\mathbf{r})=4 \pi e^2 \rho(\mathbf{r})$$
$$\Phi(\mathbf{r})=\int d^3 r^{\prime} \frac{e^2 \rho\left(\mathbf{r}^{\prime}\right)}{\left|\mathbf{r}-\mathbf{r}^{\prime}\right|}$$
<ins>Note:</ins> $\Phi$ is defined as $e\Phi$. \
After Fourier transform one gets:
$$\Phi(\mathbf{r})=\int \frac{d^3 k}{(2 \pi)^3} \frac{e^2 \rho(\vec{k})}{k^2} \exp (i \vec{k} \cdot \mathbf{r})$$
However while working on the lattice, above formula generates undesirable interaction with neighbouring cells. To avoid that problem, one should define the modified potential:

$$f(r) = \begin{cases}
\dfrac{1}{r}, & \text{if r }  < \sqrt{L_x^2+L_y^2+L_z^2} \\
0, & \text{if otherwise}
\end{cases}$$

Where $L_i = N_i\Delta x$, $i = x, y, z$, $N_i$ denote number of equidistant lattice points in each direction, $\Delta x$ is lattice constant. \
Fourier transform of given potential is:
$$f(k)=4 \pi \frac{1-\cos \left(k \sqrt{\left.L_x^2+L_y^2+L_z^2\right)}\right.}{k^2}$$
and one gets:
$$\Phi(\mathbf{r})=\int \frac{d^3 k}{(2 \pi)^3} \frac{e^2 \rho(\vec{k})}{k^2} \exp (i \vec{k} \cdot \mathbf{r})=\frac{1}{27 N_x N_y N_z} \sum_{\vec{k} \in L_x L_y L_z} e^2 \rho(\vec{k}) f(k) \exp (i \vec{k} \cdot \mathbf{r})$$
Where $\rho(\vec{k})$ is the Fourier transformed density on the lattice $27L_xL_yL_z$
