%% Very simple example for non-linear dynamic system
%Remark: you can replace it or call an external software
function ModelResponse = PhysicalModel(t,P)

if size(P,2)==1 
    P(2)=0; 
end

ModelResponse=(P(1).^2+P(2)-1).^2+P(1).^3+0.5*P(1)*exp(P(2))-sqrt(t)*P(1);
for i=3:length(P)
    ModelResponse=ModelResponse+P(i);
end