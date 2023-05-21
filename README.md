# Racosfera

Ecology simulations

This is a streamlit app that simulates a generalized lotke volterra ecology model. A lot of what you're going to see is inspired by [Stefano Allesina's lecture](https://stefanoallesina.github.io/Sao_Paulo_School/intro.html)

# running

You have a running streamlit app [here](https://polmonso-streamlit-racosfera-ecohome-okep7q.streamlit.app)

Alternativelly, you can run it locally. Install with `poetry install`, run with `streamlit run ecohome.py`

## multispecies

The multispecies version of the app is running here [here](https://polmonso-streamlit-racosfera-multiecohome-vteq5l.streamlit.app)

Run locally with `streamlit run multiecohome.py`


# General Lotka Volterra (GLV)

With all lower case as vectors and upper case for matrices. $\odot$ as the pair-wise product.

$$x(t+1) = x(t) + x(t)\odot(r + A·x(t))$$

For exemple with 3 wolves, 5 rabbits, a reproduction rate of 2 for rabbits and -1 for wolves, while
wolves grow 2 per rabbit eaten and rabbits lose 1 per wolf. And imagine that the carrying capacity is 20 for each, meaning that as we approach the carrying capacity, they start killing each other.

$$ \begin{pmatrix}
   w_1 \\
   r_1 \\
   \end{pmatrix}  = \begin{pmatrix}
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
    3 \\
    5 \\
    \end{pmatrix}
   \right) $$

where $w_1$ and $r_1$ is the density of wolves and rabbits respectively after one interation. The general case for one unit of time would be written as follows

$$x(t+1) = x(t) + x(t)\odot(r + A·x(t))$$

The differential representation of the GLV is

$$\dfrac{dx_i}{dt} = x_i(\mathbf{r} + \mathbf{Ax})$$

$r$ is the intrinsic growth/death rate vector of the N species ($i \in [1..N]$) — See the [thoughts](docs/thoughts.md) to extend interpretations —.
$\mathbf{A}$ is the iner and intra-species impact on one another. In general $a_{ii} < 0$ so that in abscence of prey the species goes to extintion.
