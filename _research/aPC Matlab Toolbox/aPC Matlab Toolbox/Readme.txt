aPC Matlab Toolbox: Data-driven Arbitrary Polynomial Chaos Expansion

AUTHOR: 
Sergey Oladyshkin

AFFILIATION: 
Stuttgart Research Centre for Simulation Technology, 
Department of Stochastic Simulation and Safety Research for Hydrosystems, 
Institute for Modelling Hydraulic and Environmental Systems, 
University of Stuttgart, Pfaffenwaldring 5a, 70569 Stuttgart

CONTACT INFORMATION: 
E-mail: Sergey.Oladyshkin@iws.uni-stuttgart.de
Phone: +49-711-685-60116
Fax: +49-711-685-51073
Website: http://www.iws.uni-stuttgart.de

SCIENTIFIC LITERATURE:
Oladyshkin S. and Nowak W. Data-driven uncertainty quantification using the arbitrary polynomial chaos expansion. Reliability Engineering & System Safety, Elsevier, V. 106, P. 179–190, 2012.
Oladyshkin S. and Nowak W. Incomplete statistical information limits the utility of high-order polynomial chaos expansions. Reliability Engineering & System Safety, 169, 137-148, 2018.
Oladyshkin S., de Barros F. P. J. and Nowak W. Global sensitivity analysis: a flexible and efficient framework with an example from stochastic hydrogeology. Advances in Water Resources 37 (2012): 10-22.

GENERAL INFORMATION:
Polynomial chaos expansion (PCE) introduced by Norbert Wiener in 1938. PCE can be seen, intuitively, as a mathematically optimal way to construct and obtain a model response surface in the form of a high-dimensional polynomial in uncertain model parameters. Recently the polynomial chaos expansion received a generalization towards the arbitrary polynomial chaos expansion (aPC: Oladyshkin S. and Nowak W., 2012), which is a so-called data-driven generalization of the PCE. Like all polynomial chaos expansion techniques, aPC approximates the dependence of simulation model output on model parameters by expansion in an orthogonal polynomial basis. The aPC generalizes chaos expansion techniques towards arbitrary distributions with arbitrary probability measures, which can be either discrete, continuous, or discretized continuous and can be specified either analytically (as probability density/cumulative distribution functions), numerically as histogram or as raw data sets. The aPC at finite expansion order only demands the existence of a finite number of moments and does not require the complete knowledge or even existence of a probability density function. This avoids the necessity to assign parametric probability distributions that are not sufficiently supported by limited available data. Alternatively, it allows modellers to choose freely of technical constraints the shapes of their statistical assumptions. Investigations indicate that the aPC shows an exponential convergence rate and converges faster than classical polynomial chaos expansion techniques. The aPC Matlab Toolbox have been developed in the year 2010 for scientific purpose and now it is available for the Matlab community. 

SPECIFIC INFORMATION:
The aPC Toolbox is based on the aPC definition in the papers by Oladyshkin and Nowak and you could find an example how to use it in the script MainRun_aPC.m. In order to use the aPC Toolbox, please, initialize the aPC object and its desired parameters as following: 
•	Object Arbitrary Polynomial Chaos aPC=ArbitraryPolynomialChaos 
•	Data Driven Input Distribution: aPC.InputDistribution 
•	Number of Input Parameters: aPC.NumberOfInputs 
•	Expansion Degree: aPC.ExpansionDegree 
•	Initialization of Arbitrary Polynomial Chaos: Initialization(aPC) 
In order to use the aPC Toolbox as model reduction approach (surrogate model, response surface, etc.), please, construct the Input Training Data Set using the command GaussainCollocaiton(aPC,'PCM') specifying the training strategy: 'FT'- Full Tensor Grid or 'PCM' - Probabilistic Collocation Method. Then, construct the Output Training Data Set evaluating the original physical model for Input Training Data Set as specified in the script The MainRun_aPC.m. The MainRun_aPC.m call the original model as a Black Box model determined in PhysicalModel.m that could be replaced by any other models or an external software.  Alternately, you could use the aPC Toolbox as conventional Machine Learning approach loading the Input Training Data Set and corresponding Output the Input Training Data Set omitting any specification of the physical model. 
The aPC Toolbox constructs the aPC expansion according to equation (3) in the paper by Oladyshkin and Nowak (2012). The script aPC_OrthonormalBasis.m solves the equation (14) from the paper by Oladyshkin and Nowak (2012) and construct the data-driven orthonormal polynomial basis.  Please, use the following functionality of aPC Toolbox or training, prediction, uncertainty quantification and 
•	Training of Arbitrary Polynomial Chaos: aPC=train(aPC,TrainingInput,TrainingOutput)
•	Prediction using Arbitrary Polynomial Chaos: PredictionOutput=predict(aPC,aPC.InputDistribution)
•	Uncertainty Quantification using Arbitrary Polynomial Chaos [OutputMean, OutputVariance] = UQ(aPC)
•	Global sensitivity analysis using Arbitrary Polynomial Chaos: [SortSobolIndices, SobolTotal] = GSA(aPC)

VIZUALIZATION:
Visualisation of uncertainty quantitation via mean value and standard deviation you could find in the Visualization_UQ.m file. Visualisation of Global sensitivity analysis based on aPC introduced in the paper Oladyshkin, de Barros and Nowak (2012) you could find in the Visualization_GSA.m file. Additional, the script Visualization_ResponceSurf.m shows how to visualize the response surface using 2 parameters.
