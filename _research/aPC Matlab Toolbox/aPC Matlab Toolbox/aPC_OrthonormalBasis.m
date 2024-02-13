%% aPC Matlab Toolbox
% Data-driven Arbitrary Polynomial Chaos Expansion
% Author: Sergey Oladyshkin
% Developed year: 2009
% Stuttgart Center for Simulation Science
% Department of Stochastic Simulation and Safety Research for Hydrosystems,
% Institute for Modelling Hydraulic and Environmental Systems
% University of Stuttgart, Pfaffenwaldring 5a, 70569 Stuttgart
% E-mail: Sergey.Oladyshkin@iws.uni-stuttgart.de
% Phone: +49-711-685-60116
% Fax: +49-711-685-51073
% http://www.iws.uni-stuttgart.de
% The current aPC Matlab Toolbox is using definition of aPC that is presented in the following manuscripts: 
% Oladyshkin S. and Nowak W. Data-driven uncertainty quantification using the arbitrary polynomial chaos expansion. Reliability Engineering & System Safety, Elsevier, V. 106, P. 179–190, 2012.
% Oladyshkin S. and Nowak W. Incomplete statistical information limits the utility of high-order polynomial chaos expansions. Reliability Engineering & System Safety, 169, 137-148, 2018.
% Oladyshkin S., de Barros F. P. J. and Nowak W. Global sensitivity analysis: a flexible and efficient framework with an example from stochastic hydrogeology. Advances in Water Resources 37 (2012): 10-22.

%% Construction of Data-driven Arbitrary Orthonormal Polynomial Basis
% OrthonormalBasis = aPC_OrthonormalBasis(Data, d)
% Input:
% Data - raw data array
% d - degree of the orthonormal polynomial basis
% Output:
% OrthonormalBasis - orthonormal polynomial basis 