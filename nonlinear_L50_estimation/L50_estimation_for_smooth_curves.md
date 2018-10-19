L50 estimation for GAMs
================
Eric Pedersen
September 17, 2018

This tutorial will focus on how to estimate L50 values for observational fisheries length-at-maturity data when the relationship between length and maturity is more complex than a logistic curve (so simple linear methods of calculating L50 values don't work). L50 values, in fisheries ecology, correspond to the length (or age) at which 50% of the individuals in a population have gone through some transition (e.g. have become sexually mature, or are producing eggs, or have undergone a sex transition).

For this tutorial, I'll use the following packages:

``` r
library(dplyr)   #for working with data frames
library(tidyr)   #for working with data frames
library(ggplot2) #for generating plots
library(mgcv)    #for fitting nonlinear (generalized additive) models

set.seed(1)
```

I'll use the following generated data set. I'll be assuming that we have survey values sampled from a spatial grid (indexed by `x` and `y`) and that length at maturity follows a non-linear function of length, with the intercept depending on `x` and `y`:

``` r
dat = crossing(x = 1:5,  #x and y occur across a 5x5 grid
               y = 1:5, 
               #2 mm length bins, spanning from 10 to 120 mm.
               length = seq(10,120, by = 2), 
               #10 replicates at each size bin.
               rep = 1:10) %>%
  mutate(
    #create values on the link (logit) scale. In this data, maturation is
    #fastest in the middle of the range (x=3,y=3), and the length at maturity
    #curve is based off a two-part logistic function that changes slope at L = 45 mm
    prob_logit = 2 -(x-3)^2 - (y-3)^2 + ifelse(length<45, 0.3*(length-45),0.1*(length-45)),
    prob = plogis(prob_logit),
    condition = rbinom(n = n(), prob = prob,size = 1))
```

This is what the data looks like:

``` r
head(dat)
```

    ## # A tibble: 6 x 7
    ##       x     y length   rep prob_logit         prob condition
    ##   <int> <int>  <dbl> <int>      <dbl>        <dbl>     <int>
    ## 1     1     1     10     1      -16.5 0.0000000683         0
    ## 2     1     1     10     2      -16.5 0.0000000683         0
    ## 3     1     1     10     3      -16.5 0.0000000683         0
    ## 4     1     1     10     4      -16.5 0.0000000683         0
    ## 5     1     1     10     5      -16.5 0.0000000683         0
    ## 6     1     1     10     6      -16.5 0.0000000683         0

And here's a plot of the true length-at-maturity curves, with points showing raw data, the black curve indicating the true probability at length for a given `x,y` combination, and the horizontal dashed line indicating the L50 :

``` r
raw_data_plot = ggplot(dat, aes(x = length, y = condition))+ 
  facet_grid(y~x, labeller = label_both) + 
  geom_point(size=0.1)+
  geom_line(aes(y=prob), size=1)+
  geom_hline(yintercept = 0.5,linetype=2)+
  theme_bw()+
  theme(panel.grid = element_blank())
  

raw_data_plot
```

![](figures/L50-plotraw-1.png)

In logistic regression, the probability of some `condition`, ![p](https://latex.codecogs.com/png.latex?p "p"), is modelled with a generalized linear model (GLM) with a binomial distribution for`condition` being in one state or another, and the logit-transformed value ![logit(p) = ln(p/(1-p))](https://latex.codecogs.com/png.latex?logit%28p%29%20%3D%20ln%28p%2F%281-p%29%29 "logit(p) = ln(p/(1-p))") being modelled as a linear combination of the variables (**x**) of interest:

![logit(p) = \\beta\_0 + \\beta\_1\*x\_1 + \\beta\_2\*x\_2 + ...](https://latex.codecogs.com/png.latex?logit%28p%29%20%3D%20%5Cbeta_0%20%2B%20%5Cbeta_1%2Ax_1%20%2B%20%5Cbeta_2%2Ax_2%20%2B%20... "logit(p) = \beta_0 + \beta_1*x_1 + \beta_2*x_2 + ...")

where ![\\beta\_0](https://latex.codecogs.com/png.latex?%5Cbeta_0 "\beta_0") corresponds to an intercept, and ![\\beta\_1](https://latex.codecogs.com/png.latex?%5Cbeta_1 "\beta_1"), ![\\beta\_2](https://latex.codecogs.com/png.latex?%5Cbeta_2 "\beta_2"), etc. correspond to slopes for each variable of interest. In **R** code, this would be described by:

``` r
glm(condition ~ 1 + x_1 + x_2 + ..., data = dat, family=binomial(link = "logit"))
```

Where I have explicitly included the intercept should be included (the term `1`) and specified that the model should use a logit link.

In length-at-maturity analyses where length at maturity can vary with covariates, the equation would be:

![logit(p) = \\beta\_0 + \\beta\_{length}\*L + \\beta\_1\*x\_1 + \\beta\_2\*x\_2 + ...](https://latex.codecogs.com/png.latex?logit%28p%29%20%3D%20%5Cbeta_0%20%2B%20%5Cbeta_%7Blength%7D%2AL%20%2B%20%5Cbeta_1%2Ax_1%20%2B%20%5Cbeta_2%2Ax_2%20%2B%20... "logit(p) = \beta_0 + \beta_{length}*L + \beta_1*x_1 + \beta_2*x_2 + ...")

And the corresponding **R** model is:

``` r
glm(condition ~ 1 + length + x_1 + x_2 + ..., data = dat, family=binomial(link = "logit"))
```

For the sample data, the linear R model would be:

``` r
l_mat_linear = glm(condition ~ 1 + length + x + y, data = dat, family=binomial(link = "logit"))
```

Estimating L50: the linear approach
-----------------------------------

For these models, L50 value for a given set of covariates is the the value of `length` that corresponds to ![logit(p)=0](https://latex.codecogs.com/png.latex?logit%28p%29%3D0 "logit(p)=0") when the other covariates are held constant. In a standard GLM model, this can be found by solving a pretty straightforward linear equation:

![
L\_{50} = -\\frac{\\beta\_0 + \\beta\_1\*x\_1 + \\beta\_2\*x\_2 + ...}{\\beta\_L}\\qquad (1)
](https://latex.codecogs.com/png.latex?%0AL_%7B50%7D%20%3D%20-%5Cfrac%7B%5Cbeta_0%20%2B%20%5Cbeta_1%2Ax_1%20%2B%20%5Cbeta_2%2Ax_2%20%2B%20...%7D%7B%5Cbeta_L%7D%5Cqquad%20%281%29%0A "
L_{50} = -\frac{\beta_0 + \beta_1*x_1 + \beta_2*x_2 + ...}{\beta_L}\qquad (1)
")

This can be found in R pretty simply. First we'll create new data at the values we want to predict at. We have to create a dummy value of length to create predictions at, but we won't be using this value to calculate L50 values here:

``` r
dat_pred = crossing(x = unique(dat$x),
                    y = unique(dat$y),
                    length = 0, #dummy value of length. Set to zero so it will not affect the predictions
                    condition = 1 #dummy value of condition; will not affect predicted L50 values, but it is needed for building the model matrix.
                    )
```

Next we'll generate a *model matrix* from this, that creates the appropriate variables in the right order

``` r
linear_model_matrix = model.matrix(l_mat_linear$formula, dat_pred)
```

Now we can multiply this by the coefficient values from the model, divided by the coefficient value for length:

``` r
linear_coef = coef(l_mat_linear)

L50_numerator = linear_model_matrix%*%linear_coef #matrix multiplication of 
L50_denominator = linear_coef[["length"]]

dat_pred$L50_linear = -L50_numerator/L50_denominator
```

Let's see how well this does at estimating the true L50, by adding the predicted curve as a blue line, and the L50 values as a red vertical line to the previous plot.

``` r
linear_l50_plot = raw_data_plot +
  geom_line(aes(y=fitted(l_mat_linear)),color="blue")+
  geom_vline(data =dat_pred, aes(xintercept = L50_linear), linetype=1, color="red")
  

linear_l50_plot
```

![](figures/L50-plotlinear-1.png)

Note that this does a really awful job of predicting the L50 values. Part of this is that we didn't model the quadratic relationship with space; we can add that in easily:

``` r
l_mat_quad = glm(condition ~ 1 + length + x +I(x^2) + y + I(y^2), data = dat, family=binomial(link = "logit"))

quad_model_matrix = model.matrix(l_mat_quad$formula, dat_pred)
quad_coef = coef(l_mat_quad)

L50_numerator = quad_model_matrix%*%quad_coef 
L50_denominator = quad_coef[["length"]]

dat_pred$L50_quad = -L50_numerator/L50_denominator

quad_l50_plot = raw_data_plot +
  geom_line(aes(y=fitted(l_mat_quad)),color="blue")+
  geom_vline(data =dat_pred, aes(xintercept = L50_quad), linetype=1, color="red")
  

quad_l50_plot
```

![](figures/L50-quad-1.png)

This works a fair bit better, but it's still not doing a great job for, e.g. `x=3, y=3`, and going by the blue lines, it seems to be because the estimated logistic curve doesn't capture the true size-at-maturity curve. I could add a quadratic term for length, or a length-location interaction term, but in that case, I couldn't use equation (1) to find the L50 values. In fact, equation (1) only works when we assume that the relationship between length and logit-probability is linear. When we move into the realm of nonlinear relationships, we have to find L50 using nonlinear solvers.

Estimating L50: the nonlinear approach
--------------------------------------

First, let's fit this model using a GAM:

``` r
l_mat_gam = gam(condition ~ 1 + s(length) +s(x, y, k=5), data = dat, family=binomial(link = "logit"))
```

Here the term `s(length)` denotes a smoother for length, and the term `s(x,y)` denotes a 2D smoother for x and y. This model does a much better job of capturing the length-at-maturity relationship:

``` r
gam_plot = raw_data_plot +
  geom_line(aes(y=fitted(l_mat_gam)),color="blue")

gam_plot
```

![](figures/L50-gam_plot-1.png)

However, now how do we get the L50 values for this model? It's conceptually similar to how I did it for the linear model; basically, I'll find the value of `length` that results in the link equalling zero. However, the equation for the finding L50 is now:

![
0 = \\beta\_0 + f(length) + g(x,y) \\qquad (2)
](https://latex.codecogs.com/png.latex?%0A0%20%3D%20%5Cbeta_0%20%2B%20f%28length%29%20%2B%20g%28x%2Cy%29%20%5Cqquad%20%282%29%0A "
0 = \beta_0 + f(length) + g(x,y) \qquad (2)
")

where `f(length)` and `g(x,y)` are nonlinear equations. Now there is no easy linear equation like equation (1) to solve equation (2). However, R has good nonlinear optimizer codes that can easily solve this type of equation. Conceptually, we just want to, for a given set of `x` and `y` values, search through all length values to find one that solves equation (2).

I'll first define a function that can take a length, a vector of coefficients, and a data frame of covariates that we want predictions over, and returns the square of the link function for this data. I want the square value as the optimization functions in R assume I am minimizing a function, and the square of the link function will always have a minimum at zero (where equation (2) is satisfied)[1].

``` r
get_link_sqrt = function(length, coefs, covar, model){
  covar$length = length
  #this returns the linear predictors corresponding to the new data. 
  lp_matrix = predict(model,newdata = covar, type="lpmatrix") 
  
  #to get predictions on the link scale, we'll use matrix multiplication to
  #multiply the lp_matrix times the vector of coefficients.
  link_pred = lp_matrix%*%coefs
  return(link_pred^2)
}
```

We can now use optimize to solve this equation for a single row of `dat_pred`. We have to set the interval that we want to optimize over (basically, the values we consider to be valid L50 values). Here we use 0 to 150.

``` r
test_l50 = optimize(get_link_sqrt,interval = c(0,150),
                    coefs = coef(l_mat_gam),
                    covar = dat_pred[1,], 
                    model = l_mat_gam)
print(test_l50)
```

    ## $minimum
    ## [1] 102.3863
    ## 
    ## $objective
    ##           [,1]
    ## 1 2.003679e-16

This returns a minimum value (i.e. the L50) and an objective value. If the function is properly minimizing, this will be very close to zero.

Now to get the L50 values for all sets of covariates, we can loop over the predicted data[2]:

``` r
dat_pred$L50_gam = 0 #creating an empty value here
#we also want to keep track of the objective function. If this deviates away
#from 0, it's a sign that the optimizer didn't,well, optimize.
dat_pred$objective = 0 

for(i in 1:nrow(dat_pred)){
  current_L50 = optimize(get_link_sqrt,interval = c(0,150),
                    coefs = coef(l_mat_gam),
                    covar = dat_pred[i,], 
                    model = l_mat_gam)
  dat_pred$L50_gam[i] = current_L50$minimum
  dat_pred$objective[i] = current_L50$objective
}
```

We can plot this to see how well it does:

``` r
gam_l50_plot = raw_data_plot +
  geom_line(aes(y=fitted(l_mat_gam)),color="blue")+
  geom_vline(data =dat_pred, aes(xintercept = L50_gam), linetype=1, color="red")
  

gam_l50_plot
```

![](figures/L50-plotgam-1.png)

And we can see that all of the estimates have converged properly, as the optimal criteria is very close to zero for all L50 values:

``` r
dat_pred$objective
```

    ##  [1] 2.003679e-16 2.911857e-13 1.482868e-15 3.066515e-13 2.565257e-16
    ##  [6] 2.891587e-13 5.516502e-13 1.116130e-13 5.973890e-13 3.413932e-13
    ## [11] 1.505867e-15 1.125319e-13 1.253650e-12 2.042231e-13 1.056623e-15
    ## [16] 3.013204e-13 5.914159e-13 2.296293e-13 6.967090e-13 3.142159e-13
    ## [21] 2.469959e-16 3.384411e-13 1.117079e-15 3.157200e-13 2.225995e-16

Note that, as a nonlinear solver, this isn't guaranteed to find an optimum, and as this function is nonlinear, it's possible to have multiple L50 values! This should be pretty rare in actual data, but can happen with sampling issues, or when trying extrapolate predicted L50 values for unobserved covariate combinations. Note also that if you give a maximum or minimum value of length outside the range of the data, `gam` will happily give you a linear extrapolation out to whatever length value you give it, and `optimize` will happily try to find L50 values out there. I would be very careful on relying on any L50 predictions outside the range of your data. Note that this can be a problem with the linear estimate of L50 too...

The nice thing about this approach is that it also allows us to come up with estimates of uncertainty in the L50 point as well. This is a bit more complicated, but it boils down to:

1.  Simulate a bunch of new coefficient values that are also considered feasible, by drawing coefficients from a multivariate normal distribution with a mean of the observed coefficient vector, and an variance-covariance matrix given by the model.
2.  Repeat the L50 estimation for each new coefficient vector for each value of covariates that you want a prediction.
3.  Calculate a standard deviation for those L50 estimates.

Let's demonstrate this:

``` r
dat_pred$L50_gam_sd = 0 #creating an empty value here

#We'll generate 50 random coefficient vectors. This can take a while to run,
#so it's not a bad idea to keep n_sample low.
n_samples = 50
coefs = rmvn(n_samples, mu = coef(l_mat_gam), V = vcov(l_mat_gam))
L50_samples = rep(0, times = n_samples)


for(i in 1:nrow(dat_pred)){
  for(j in 1:n_samples){
    current_L50 = optimize(get_link_sqrt,interval = c(0,150),
                    coefs = coefs[j,],
                    covar = dat_pred[i,], 
                    model = l_mat_gam)
    L50_samples[j] = current_L50$minimum
  }
  dat_pred$L50_gam_sd[i] = sd(L50_samples)
  
}
```

We can plot the length-at-maturity curves with the 95% CIs now:

``` r
gam_l50_sd_plot = raw_data_plot +
  geom_line(aes(y=fitted(l_mat_gam)),color="blue")+
  geom_rect(data = dat_pred,
            aes(xmin = L50_gam - 1.96*L50_gam_sd,
                xmax = L50_gam + 1.96*L50_gam_sd,
                ymin = -Inf,
                ymax = Inf),fill = "red", alpha=0.25)+
  geom_vline(data =dat_pred, aes(xintercept = L50_gam), linetype=1, color="red")
  

gam_l50_sd_plot
```

![](figures/L50-plotgam_sd-1.png)

It's likely also possible to calculate the standard deviations with a bit less computing time, using something like the delta method, but I don't have time to dig into the math on that here.

Update: applying this approach using lme4
-----------------------------------------

I got a question about how to calculate L50 values using **lme4**, instead of **mgcv**. The general approach here is conceptually pretty similar to the nonlinear approach described above, but it differs in a few details.

The big difference between **lme4** and **mgcv** here is that **lme4** treats random effects as different from other types of parameters, so it can be difficult to get the same type of variance-covariance matrix. There is also a conceptual issue when estimating uncertainty about the L50 for a random effects model: do we want to know the estimate (and uncertainty) for the L50 value for that particular site (i.e. hold the random effects constant) or for a different site but with similar characteristics (covariates). The first estimate is what you want when when the L50 value for a particular location is of interest (i.e. when random effects are being used as a method to reduce overfitting when you've got a large number of categorical variables, but are still interested in individual random effect values). This might be the case if you are fitting L50 values for fish from a wide range of lakes, and using lake as a random effect to prevent overfit, but still want to make inferences for the observed lakes. The second question is interesting when you want to extrapolate outside of the sample, to make inferences including uncertainty for L50 values for unsampled sites (treating site as a nuisance parameter).

Before we look at either type of uncertainty or estimate, though, we should fit the model of interest. Here I'm going to create some new sample data, where the linear assumption is true, but there is inter-site variation, then use the `glmer` function from **lme4** to fit the new data with a logistic regression:

``` r
dat_lme4= crossing(x = 1:5,  #x and y occur across a 5x5 grid
               y = 1:5, 
               #2 mm length bins, spanning from 10 to 120 mm.
               length = seq(10,120, by = 2), 
               #10 replicates at each size bin.
               rep = 1:10) %>%
  group_by(x,y)%>% #going to create a random effect for each site (combination of x and y)
  mutate(
    site =paste(x,y, sep = "_"),
    site_intercept = rnorm(1,0,0.2), #intercept drawn from a normal distribution w/ std. dev. of 0.2
    site_slope  = rnorm(1,0, 0.02), #site-specific slope drawn from a normal dist. w/ std. dev. of 0.02
    #create values on the link (logit) scale. In this data, maturation is
    #a linear function of length with site specific intercepts and slopes
    prob_logit = 2 + (x-3) + (y-3) + site_intercept + (0.1+site_slope)*(length-65),
    prob = plogis(prob_logit),
    condition = rbinom(n = n(), prob = prob,size = 1)
    )%>%
  ungroup()%>%
  mutate(    
    #lme4 often has problems with unscaled variables, so we will center length around its mean value
    #we will store the mean value of length in case we need to apply that to new data sets in the future
    length_mean = mean(length),
    length_centered = length - length_mean)
    


library(lme4)
```

    ## Loading required package: Matrix

    ## 
    ## Attaching package: 'Matrix'

    ## The following object is masked from 'package:tidyr':
    ## 
    ##     expand

    ## 
    ## Attaching package: 'lme4'

    ## The following object is masked from 'package:nlme':
    ## 
    ##     lmList

``` r
library(boot) #needed for the bootstrap we'll do after this
l_mat_glmer = glmer(condition ~ 1 + length_centered + x+y + (1+length_centered| site), 
                    data = dat_lme4, 
                    family=binomial(link = "logit"))
```

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : Model failed to converge with max|grad| = 0.00505009
    ## (tol = 0.001, component 1)

I think it's possible to do something similar to what I did with mgcv for getting model variance-covariance matrices, and calculating the posterior, but it's a lot more complicated than for the **mgcv** case. Instead, I'm going to use a parametric boostrap method. The parametric bootstrap is based on simulating new data multiple times from the fitted model, then re-fitting the model to that simulated data, and extracting the parameter of interest. This will give a range of values that could also have come from the same data (see `?lme4::bootMer` for more information).

### L50 values for glmer models for the same sites (constant random effects)

First we'll look at the case where we're holding the site constant (so only uncertainty in the estimate of the random effects matters, but inter-random effect variability will be propagated).

We have to first write a function that takes a fitted model, and returns our statistic of interest. In this case, the statistic is a vector of L50 values, one for each site in the data. This function has to be self-contained (have no other arguments besides the model), so I am writing it to contain the predicted data that we expect it to have to make the predictions. I'll have to write a few helper functions first, that the final fitting function will call:

``` r
#This function is a modified version of the above function that works for lme4
get_lme4_link_sqrt = function(length_centered, covar, model){
  covar$length_centered = length_centered
  
  #this returns the predictions for the model on the link scale 
  link_pred = predict(model,newdata = covar, type="link") 
  
  #Since we're modifying the code to take a vector of predictors, this returns
  #the sum of squared deviations from zero, instead of just the square deviation
  #of one value from zero
  return(sum(as.numeric(link_pred)^2))
}

find_l50_lme4 = function(model) {
  #we have to create the test data, which should have a row for every covariate
  #we want to test
  dat_pred = crossing(x = 1:5, y = 1:5)%>%
    mutate(site = paste(x,y, sep="_"))
  
  #note: since we're using centered length values here, the lower and upper
  #bounds have to be adjusted to take that into account!! Here I'm using ad-hoc
  #values (I know mean length is 65) but this should be adjusted for your data
  #I'm also initializing the parameter at zero (i.e. the mean length)
  L50_fit = optim(par = rep(0, times = nrow(dat_pred)), 
                            get_lme4_link_sqrt,
                            lower = 0-65, upper = 150-65,
                    covar = dat_pred, 
                    model = model,
                  method = "L-BFGS-B")
  L50_values = L50_fit$par

  
  return(L50_values)
}

#this function will get CI values for all fitted from a bootstrap result.
#Taken from ?mgcv::bootMer
bCI.tab = function(b,ind=length(b$t0), type="perc", conf=0.95) {
        btab0 = t(sapply(as.list(seq(ind)),
                          function(i)
            boot::boot.ci(b,index=i,conf=conf, type=type)$percent))
        
        btab = btab0[,4:5]
        btab = cbind(b$t0, btab)
        
        colnames(btab) = c("fit", "lower","upper")
        return(btab)
}
```

The final step is to create parametric bootstrap samples from our model. As this takes a really long time (both the model-refitting step and the optimization step are slow here), we're going to keep the number of samples low (~150) but it's probably better to use more bootstrap samples for a real analysis to get a better estimate of the uncertainty. I'm also going to use the fact that my computer has several cores to parallize this a bit. Note that I'm passing it a seed (to make sure the analysis is repeatable) and setting the number of cores. The `use.u` argument specificies that we want to keep the original random effect estimates instead of resampling them.

``` r
boot_glmer_condtional = bootMer(l_mat_glmer, 
                                find_l50_lme4, 
                                use.u = TRUE, 
                                nsim = 150, 
                                seed = 2, 
                                parallel = "multicore", 
                                ncpus = 6)


#we can get the confidence intervals out with the bCI.tab function:

boot_glmer_condtional_CI = bCI.tab(boot_glmer_condtional)

#we'll add back in the mean length value, so it's on the same scale as the data:
boot_glmer_condtional_CI = boot_glmer_condtional_CI + dat_lme4$length_mean[1]


dat_pred_glmer = crossing(x = 1:5, y = 1:5)%>%
    mutate(site = paste(x,y, sep="_"),
           length = 0,
           condition = 0)

dat_pred_glmer_conditional = dat_pred_glmer %>%
  bind_cols(as_data_frame(boot_glmer_condtional_CI))
```

Let's see how well this fits the data:

``` r
glmer_conditional_plot = ggplot(dat_lme4, aes(x = length, y = condition))+ 
  facet_grid(y~x, labeller = label_both) + 
  geom_point(size=0.1)+
  geom_line(aes(y=prob), size=1)+
  geom_hline(yintercept = 0.5,linetype=2)+
  geom_vline(data= dat_pred_glmer_conditional, aes(xintercept = fit))+
  geom_rect(data = dat_pred_glmer_conditional,
            aes(xmin = lower,
                xmax = upper,
                ymin = -Inf,
                ymax = Inf),
            fill = "red", 
            alpha=0.25)+
  theme_bw()+
  theme(panel.grid = element_blank())
  

glmer_conditional_plot
```

![](figures/L50-plotglmer-1.png)

Note that for values like x=5 / y=5, the confidence interval is small, and clustered at the bottom of the range. This is implying that the model estimates that the true L50 value is likely very near the bottom of the boundary we set. This is because we picked some pretty unrealistic data for the model, but it does highlight what a boundary effect has on this.

### L50 values for glmer models for new sites (sampling new random effects)

Finally, we'll repeat this with new random effects. Note that this code would likely take a fair bit of adaptation for a real example. This is just a basic demo.

The procedure is basically the same, except we set `use.u` equal to false. If you wanted to get uncertainty estimates for new covariate values, though, keep in mind you'd have to fully re-write the find\_l50\_lme4 function. In a real analysis, you'd wrap all of this in one bigger function that would let you pass the new covariate values you want CIs for.

``` r
boot_glmer_uncondtional = bootMer(l_mat_glmer, 
                                find_l50_lme4, 
                                use.u = FALSE, 
                                nsim = 150, 
                                seed = 2, 
                                parallel = "multicore", 
                                ncpus = 6)


#we can get the confidence intervals out with the bCI.tab function:

boot_glmer_uncondtional_CI = bCI.tab(boot_glmer_uncondtional)

#we'll add back in the mean length value, so it's on the same scale as the data:
boot_glmer_uncondtional_CI = boot_glmer_uncondtional_CI + dat_lme4$length_mean[1]


dat_pred_glmer = crossing(x = 1:5, y = 1:5)%>%
    mutate(site = paste(x,y, sep="_"),
           length = 0,
           condition = 0)

dat_pred_glmer_unconditional = dat_pred_glmer %>%
  bind_cols(as_data_frame(boot_glmer_uncondtional_CI))


glmer_unconditional_plot = ggplot(dat_lme4, aes(x = length, y = condition))+ 
  facet_grid(y~x, labeller = label_both) + 
  geom_point(size=0.1)+
  geom_line(aes(y=prob), size=1)+
  geom_hline(yintercept = 0.5,linetype=2)+
  geom_vline(data= dat_pred_glmer_unconditional, aes(xintercept = fit))+
  geom_rect(data = dat_pred_glmer_unconditional,
            aes(xmin = lower,
                xmax = upper,
                ymin = -Inf,
                ymax = Inf),
            fill = "red", 
            alpha=0.25)+
  theme_bw()+
  theme(panel.grid = element_blank())
  

glmer_unconditional_plot
```

![](figures/L50-glmer_boot_unconditional-1.png)

Note that the big change is that the CIs are substantially wider. Values at the end of the range still have very tight CIs though. Also note that the fitted value is not centered on the CI; this is likely due to the fact that I just used the `t0` statistic for the mean, which is directly drawn from the fitted model (i.e. uses the original random effects to estimate it). Any observeration where the true random effect in the data is large will tend to result in the mean value being toward one side of the CI. Here, we're effectively conditioning on the observed data for the mean, but simulating new random effects for the interval. In an actual data analysis, I would probably use the something like the mean or 50th percentile (median) of the bootstrap replicates as the midpoint; However, this takes long enough to re-run that I'm not going to worry about that for this tutorial.

[1] For more information on using the linear predictor matrix (lpmatrix) to get values from GAMs, see [this blogpost by Gavin Simpson](!https://www.fromthebottomoftheheap.net/2014/06/16/simultaneous-confidence-intervals-for-derivatives/).

[2] This is also possible with `apply` functions, but here I'm using `for` loops to make it clear what I'm doing, and later because it'll be easier to get standard errors for L50 using the `for` loop approach.
