# Optimisers

Optimisers

## Usage

``` r
optimizer_adam(
  learning_rate = 0.001,
  beta_1 = 0.9,
  beta_2 = 0.999,
  eps = 1e-08,
  weight_decay = 0,
  amsgrad = FALSE
)

optimizer_adamw(
  learning_rate = 0.001,
  beta_1 = 0.9,
  beta_2 = 0.999,
  eps = 1e-08,
  weight_decay = 0.01
)

optimizer_sgd(
  learning_rate = 0.001,
  momentum = 0,
  nesterov = FALSE,
  weight_decay = 0
)

optimizer_rmsprop(
  learning_rate = 0.001,
  alpha = 0.99,
  momentum = 0,
  eps = 1e-08,
  weight_decay = 0
)

optimizer_adagrad(
  learning_rate = 0.001,
  lr_decay = 0,
  eps = 1e-10,
  weight_decay = 0
)

optimizer_adadelta(
  learning_rate = 0.001,
  rho = 0.9,
  eps = 1e-06,
  weight_decay = 0
)

optimizer_adamax(
  learning_rate = 0.001,
  beta_1 = 0.9,
  beta_2 = 0.999,
  eps = 1e-08,
  weight_decay = 0
)

optimizer_nadam(
  learning_rate = 0.001,
  beta_1 = 0.9,
  beta_2 = 0.999,
  eps = 1e-08,
  weight_decay = 0
)
```

## Arguments

- learning_rate, weight_decay:

  Standard.

- beta_1, beta_2, eps, amsgrad, momentum, nesterov, alpha, rho,
  lr_decay:

  Per optimiser.

## Value

An optimiser specification.

## Examples

``` r
optimizer_adam(learning_rate = 1e-4)
#> $name
#> [1] "adam"
#> 
#> $learning_rate
#> [1] 1e-04
#> 
#> $beta_1
#> [1] 0.9
#> 
#> $beta_2
#> [1] 0.999
#> 
#> $eps
#> [1] 1e-08
#> 
#> $weight_decay
#> [1] 0
#> 
#> $amsgrad
#> [1] FALSE
#> 
```
