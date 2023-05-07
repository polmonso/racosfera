# racosfera
Ecology simulations


# General Lotka Volterra (GLV)

With all lower case as vectors and upper case for matrices. $\odot$ as the pair-wise product.

$x(t) = x_0 + x_0\odot(r + A·x_0)$

For exemple with 3 wolves, 5 rabbits, a reproduction rate of 2 for rabbits and -1 for wolves, while
wolves grow 2 per rabbit eaten and rabbits lose 1 per wolf. And imagine that the carrying capacity is 20 for each, meaning that as we approach the carrying capacity, they start killing each other.

$$
\begin{pmatrix}
           🐺_1 \\
           🐇_1
\end{pmatrix}
=
\begin{pmatrix}
           3 \\
           5
\end{pmatrix} + \begin{pmatrix}
           3 \\
           5
\end{pmatrix}\odot\left(
    \begin{pmatrix}-1 \\ 2\end{pmatrix}
    + \begin{pmatrix}1/20 & 2\\ -1 & 1/10\\\end{pmatrix}\begin{pmatrix}3 \\ 5\end{pmatrix}
\right)

$$

$x(t+1) = x(t) + x(t)\odot(r + A·x(t))$

The differential representation of the GLV is

$\dfrac{x(t+dt)-x(t)}{dt} = \dfrac{dx}{dt} = 1 - r + A·x$





$r$ is the intrinsic growth/death rate of a species. It is logistically bound by $a_{ii}$ like so

$x(t+1) = x(t)(r + a_{ii})$

where $a_{ii} < 0$ and it might also represented as $-1/Ki$ with $K$ the carrying capacity aka maximum population sustainable by the space or whatever reason not related to the other species nor intrinsic reproduction rate, which are both embodied by $a_{ij}$ and $r_i$.
