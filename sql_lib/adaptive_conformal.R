# SOURCE: https://github.com/isgibbs/AdaptiveConformal/blob/main/ACCode.R
# Commented with the help of ChatGPT for readability

library(quantreg)
library(rugarch)

### Main method for forming election night predictions of county vote totals as in Figure 2
runElectionNightPred <- function(Y,X,alpha,gamma,tinit = 500,splitSize = 0.75,updateMethod="Simple",momentumBW=0.95){
  T <- length(Y)
  ## Initialize data storage variables
  alphaTrajectory <- rep(alpha,T-tinit)
  adaptErrSeq <-  rep(0,T-tinit)
  noAdaptErrorSeq <-  rep(0,T-tinit)
  alphat <- alpha
  for(t in tinit:T){
    ### Split data into training and calibration set
    trainPoints <- sample(1:(t-1),round(splitSize*(t-1)))
    calpoints <- (1:(t-1))[-trainPoints]
    Xtrain <- X[trainPoints,]
    Ytrain <- Y[trainPoints]
    XCal <- X[calpoints,]
    YCal <- Y[calpoints]
    
    ### Fit quantile regression on training setting
    lqrfitUpper <- rq(Ytrain ~ Xtrain, tau=1-alpha/2)
    lqrfitLower <- rq(Ytrain ~ Xtrain, tau=alpha/2)
    
    ### Compute conformity score on calibration set and on new data example
    predLowForCal <- cbind(rep(1,nrow(XCal)),XCal)%*%coef(lqrfitLower)
    predUpForCal <- cbind(rep(1,nrow(XCal)),XCal)%*%coef(lqrfitUpper)
    scores <- sapply(1:length(YCal),function(x){max(YCal[x]-predUpForCal[x],predLowForCal[x]-YCal[x])})
    qUp <- sum(coef(lqrfitUpper)*c(1,X[t,]))
    qLow <- sum(coef(lqrfitLower)*c(1,X[t,]))
    newScore <- max(Y[t]-qUp,qLow-Y[t])
    
    ## Compute errt for both methods
    confQuantNaive <- quantile(scores,1-alpha)
    noAdaptErrorSeq[t-tinit+1] <- as.numeric(confQuantNaive < newScore)
    
    if(alphat >=1){
      adaptErrSeq[t-tinit+1] <- 1
    }else if (alphat <=0){
      adaptErrSeq[t-tinit+1] <- 0
    }else{
      confQuantAdapt <- quantile(scores,1-alphat)
      adaptErrSeq[t-tinit+1] <- as.numeric(confQuantAdapt < newScore)
    }
    
    ## update alphat
    alphaTrajectory[t-tinit+1] <- alphat
    if(updateMethod=="Simple"){
      alphat <- alphat + gamma*(alpha-adaptErrSeq[t-tinit+1])
    }else if(updateMethod=="Momentum"){
      w <- rev(momentumBW^(1:(t-tinit+1)))
      w <- w/sum(w)
      alphat <- alphat + gamma*(alpha - sum(adaptErrSeq[1:(t-tinit+1)]*w))
    }

    if(t %% 100 == 0){
      print(sprintf("Done %i time steps",t))
    }
  }
  return(list(alphaTrajectory,adaptErrSeq,noAdaptErrorSeq))
}

### Main method for forming volatility predictions as in Figure 1
garchConformalForcasting <- function(returns, alpha, gamma, lookback=1250, garchP=1, garchQ=1, startUp=100, verbose=FALSE, updateMethod="Simple", momentumBW=0.95) {
  
  T <- length(returns) # Total number of returns
  startUp <- max(startUp, lookback) # Ensure startUp is at least as large as lookback
  
  # Specify GARCH(1,1) model with no mean term, normal errors
  garchSpec <- ugarchspec(
    mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
    variance.model = list(model = "sGARCH", garchOrder = c(garchP, garchQ)),
    distribution.model = "norm"
  )
  
  alphat <- alpha  # Initialize dynamic alpha
  ### Initialize data storage vectors
  errSeqOC <- rep(0, T - startUp + 1) # Online conformal error sequence
  errSeqNC <- rep(0, T - startUp + 1) # Naive conformal error sequence
  alphaSequence <- rep(alpha, T - startUp + 1) # Track evolving alpha values
  scores <- rep(0, T - startUp + 1) # Track conformity scores
  
  # Main loop: walk forward through time
  for (t in startUp:T) {
    if (verbose) {
      print(t)
    }
    
    ### 1. Fit GARCH model on lookback window
    garchFit <- ugarchfit(garchSpec, returns[(t - lookback + 1):(t - 1)], solver="hybrid")
    
    ### 2. Forecast next-period volatility
    sigmaNext <- sigma(ugarchforecast(garchFit, n.ahead=1))
    
    ### 3. Compute conformity score: normalized squared residual
    scores[t - startUp + 1] <- abs(returns[t]^2 - sigmaNext^2) / sigmaNext^2
    
    ### 4. Collect recent past scores for quantile estimation
    recentScores <- scores[max(t - startUp + 1 - lookback + 1, 1):(t - startUp)]
    
    ### 5. Update error sequences
    # Online Conformal: using current dynamic alpha
    errSeqOC[t - startUp + 1] <- as.numeric(scores[t - startUp + 1] > quantile(recentScores, 1 - alphat))
    # Naive Conformal: using fixed alpha
    errSeqNC[t - startUp + 1] <- as.numeric(scores[t - startUp + 1] > quantile(recentScores, 1 - alpha))
    
    ### 6. Update alphat based on past errors
    alphaSequence[t - startUp + 1] <- alphat
    
    if (updateMethod == "Simple") {
      # Simple online update rule
      alphat <- alphat + gamma * (alpha - errSeqOC[t - startUp + 1])
      
    } else if (updateMethod == "Momentum") {
      # Momentum-based update rule
      w <- rev(momentumBW^(1:(t - startUp + 1))) # exponential decay weights
      w <- w / sum(w)
      alphat <- alphat + gamma * (alpha - sum(errSeqOC[1:(t - startUp + 1)] * w))
    }
    
    ### Optional: Print progress every 100 steps
    if (t %% 100 == 0) {
      print(sprintf("Done %g steps", t))
    }
  }
  
  # Return the results
  return(list(alphaSequence, errSeqOC, errSeqNC))
}
