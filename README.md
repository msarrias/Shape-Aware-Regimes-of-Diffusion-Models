### How Data Representations Change and Evolve in Diffusion Models
**Goal:** 
- Use Shape-Aware Graph Distance (SAGD) to quantify how representations change during the reverse process.
- Motivation: The speciation time is found by averaging; with SAGD, we account for the data geometry, providing more understanding.
- Explore if the three dynamical regimes introduced in [[ref](https://www.nature.com/articles/s41467-024-54281-3)] generalize beyond the Ornstein-Uhlenbeck (OU) process. Test it on a DDPM framework.

**Approach:** 
- For a subset of time steps $t$, build a graph of the data where nodes are the samples, generated at time $t$, and the edges are the  local connections between them assigned using the $k$-nearest neighbors algorithm.
- Use SASNE [[ref](https://link.springer.com/article/10.1186/s12859-022-05028-8)] to embed the SAGD matrix representation and project into a lower-dimensional space.

### Test case:

#### Ornstein-Uhlenbeck process:

 **2D Bimodal Gaussian:**
- Small d: Reproduce the experiment setting used to generate Fig.1 and use SAGD to see if it allows us to see the speciation stage. If so, move to large values of d.
- Evaluate the regimes on a 3-class Gaussian dataset with density overlap.
- _Speciation time and cloning experiment:_ Do the scores define the fate of the data trajectory? Explore the role of the noise in relation to the class barrier.

#### DDPM:
- **Gaussian Data / Real data**
