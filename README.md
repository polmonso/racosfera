# racosfera
Ecology simulations

A lot of what you're going to see is inspired by [Stefano Allesina's lecture](https://stefanoallesina.github.io/Sao_Paulo_School/intro.html)


# running

`streamlit run ecohome.py`


A running streamlit app [here](https://polmonso-streamlit-racosfera-ecohome-okep7q.streamlit.app)



# General Lotka Volterra (GLV)

With all lower case as vectors and upper case for matrices. $\odot$ as the pair-wise product.

$x(t+1) = x(t) + x(t)\odot(r + A·x(t))$

For exemple with 3 wolves, 5 rabbits, a reproduction rate of 2 for rabbits and -1 for wolves, while
wolves grow 2 per rabbit eaten and rabbits lose 1 per wolf. And imagine that the carrying capacity is 20 for each, meaning that as we approach the carrying capacity, they start killing each other.

$$ \begin{pmatrix}
   a & b & c \\
   c & e & f \\
   g & h & i \\
   \end{pmatrix} $$

$$ \begin{bmatrix}
           🐺_1 \\
           🐇_1 \\
\end{bmatrix} = \begin{pmatrix}
           3 \\
           5 \\
\end{pmatrix} + \begin{pmatrix}
           3 \\
           5 \\
\end{pmatrix}\odot\left(
    \begin{pmatrix}
    -1 \\
    2 \\
    \end{pmatrix}
    + \begin{pmatrix}
    1/20 & 2 \\
     -1 & 1/10 \\
     \end{pmatrix}
     \begin{pmatrix}
     3 \\ 5 \\
     \end{pmatrix}
\right) $$

$x(t+1) = x(t) + x(t)\odot(r + A·x(t))$

The differential representation of the GLV is

$\dfrac{dx_i}{dt} = x_i(\bm{r} + \bm{Ax})$

$r$ is the intrinsic growth/death rate vector of the N species ($i \in [1..N]$). See the [thoughts](docs/thoughts.md) to extend interpretations.
\bm{A} is the iner and intra-species impact on one another. In general $a_ii < 0$ so that in abscence of prey the species goes to extintion.
