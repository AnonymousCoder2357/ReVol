# ReVol
Repository for the paper "Mitigating Distribution Shift in Stock Price Data via Return-Volatility Normalization for Accurate Prediction"

![](./ReVol.png)

# Abstract
How can we address distribution shifts in stock price data to improve stock price prediction accuracy? Stock price prediction has attracted attention from both academia and industry, driven by its potential to uncover complex market patterns and enhance decisionmaking. However, existing methods often fail to handle distribution shifts effectively, focusing on scaling or representation adaptation without fully addressing distributional discrepancies and shape misalignments between training and test data. In this paper, we propose ReVol (Return-Volatility Normalization for Mitigating Distribution Shift in Stock Price Data), a robust method for stock price prediction that explicitly addresses the distribution shift problem. ReVol leverages three key strategies to mitigate these shifts: (1) normalizing price features to remove samplespecific characteristics, including return, volatility, and price scale, (2) employing an attention-based module to estimate these characteristics accurately, thereby reducing the influence of market anomalies, and (3) reintegrating the sample characteristics into the predictive process, restoring the traits lost during normalization. Additionally, ReVol combines geometric Brownian motion for longterm trend modeling with neural networks for short-term pattern recognition, unifying their complementary strengths. Extensive experiments on real-world datasets demonstrate that ReVol enhances the performance of the state-of-the-art backbone models in most cases, achieving an average improvement of more than 0.03 in IC and over 0.7 in SR across various settings. These results underscore the importance of directly confronting distribution shifts and highlight the efficacy of our integrated approach for stock price prediction.

# How to Run
```bash
cd src
python run.py --data={dataset_name}