%% Initialize
clear, close all


% Create geometry of boundary
% Creates a circular domain with radius cr=1
% For other shapes, see decsg documentation
cr=1;
gd=[4; 0.0; 0.0; cr; cr; 0.0];
dl=decsg(gd);

% Create initial mesh
[p,e,t]=initmesh(dl,'hmax',0.5);

% Refine the mesh three times; this will be the coarse mesh
% used for the inverse problem (given the charges at the boundary, 
% find the conductivities inside the 'cell')
[p,e,t]=refinemesh(dl,p,e,t);
[p,e,t]=refinemesh(dl,p,e,t);
[p,e,t]=refinemesh(dl,p,e,t);

% These are the indices of the boundary lines; they're where the electrodes go
boundInd=e(1,:);
nodes1=length(p);

% Create the fine mesh used for the forward problem 
% (given the conductivities inside the 'cell', find the charges at the boundary)
[p1,e1,t1]=refinemesh(dl,p,e,t);
% Additional refinement can be added here if needed
% [p1,e1,t1]=refinemesh(dl,p1,e1,t1);

% Values
nodes=length(p1);
tris=length(t1);


% Display mesh (figure hidden for batch processing)
figure('visible', 'off'), pdemesh(p1,e1,t1), axis off

% High-conductivity regions referred to as 'cysts'

% The radius of the cyst
radCore=.25;
% radSmall = 0.1;  % Alternative smaller radius option

% These are the sigma values for conductivity; the conductivities don't change
inArea=5;   % Conductivity inside cyst (high conductivity)
outArea=1;  % Conductivity outside cyst (background)

% The number of training datasets we're going to generate
numInstances = 3;

% The number of cysts in the cell (typically 1 to 4)
numCysts = 3;

% Are we allowing overlap between cysts?
allowOverlap = 0;

% Are we allowing a random number of cysts?
randCysts = 0;

% Random seed for number generation (for reproducibility)
seed = 222;
rng(seed);

% Initialize arrays to store the datapoints to be saved
sigCoarseSave = [];  % Coarse mesh conductivities
phiCoarseSave = [];  % Coarse mesh boundary measurements
sigFineSave = [];    % Fine mesh conductivities
phiFineSave = [];    % Fine mesh boundary measurements
uNSave = [];         % Full solution vectors


for waaInst = 1:numInstances
    % Start by resetting the conductivity values
    % sigTrue is the fine mesh conductivity (used for forward problem)
    % sigTrue1 is the coarse mesh conductivity (used for inverse problem)
    sigTrue=ones(nodes,1);   % Fine mesh: initialize all to background conductivity
    sigTrue1=ones(nodes1,1); % Coarse mesh: initialize all to background conductivity
    
    % Generate cyst centers within the radius of the circle
    % (checked against Shyla's code for correctness)

    if randCysts
        numCysts = randi([1,2]);  % Randomly choose 1 or 2 cysts
    end
    
    % Fine mesh processing
    if allowOverlap
        % Generate the radius based on a core of radCore with small variation
        radius = radCore * ones(numCysts,1) - 0.05*randn(numCysts,1);

        % Randomly generate the angle (uniform from [0, 2*pi])
        randAngle = 2*pi * rand(numCysts,1);

        % Randomly generate the radius of the circle where the cyst will be located
        % It has to be between 0 and (0.95 - radius) to keep cyst inside domain
        randRad = (0.95 - radius) .* rand(numCysts,1);

        xTemp = randRad.*cos(randAngle);
        yTemp = randRad.*sin(randAngle);

        for waaRad = 1:numCysts
            % Set conductivity to inArea for all nodes within cyst radius (fine mesh)
            for i=1:nodes
                if (((p1(1,i)-xTemp(waaRad))^2+(p1(2,i)-yTemp(waaRad))^2)^.5<=radius(waaRad))
                    sigTrue(i)=inArea;
                end
            end %end fine mesh loop

            % Set conductivity to inArea for all nodes within cyst radius (coarse mesh)
            for i=1:nodes1
                if (((p(1,i)-xTemp(waaRad))^2+(p(2,i)-yTemp(waaRad))^2)^.5<=radius(waaRad))
                    sigTrue1(i)=inArea;
                end
            end %end coarse mesh loop

        end %end numCysts loop

    else % We are not allowing overlap between cysts

        % Store the cysts we need to compare against to prevent overlap
        prevCysts = [];  % Previous cyst centers [x, y]
        currCyst = [];   % Current cyst center
        prevRads = [];   % Previous cyst radii

        waaRad = 1;
        while waaRad <= numCysts
            % Dynamically generate the radius based on a core of radCore with small variation
            radius = radCore * ones(1) - 0.05*randn(1);

            % Randomly generate the angle (uniform from [0, 2*pi])
            randAngle = 2*pi * rand(1,1);

            % Randomly generate the radius of the circle where the cyst will be located
            % It has to be between 0 and (0.95 - radius) to keep cyst inside domain
            randRad = (0.95 - radius) .* rand(1,1);

            xTemp = randRad.*cos(randAngle);
            yTemp = randRad.*sin(randAngle);
            overlap = 0;
            currCyst = [xTemp, yTemp];
            
            if waaRad > 1
                % Check for overlap with previously placed cysts
                prevIdx = 1;
                while prevIdx <= size(prevCysts,1)
                    % For each cyst, check it against the previous cysts
                    % There's overlap if the centers are within the sum of the radii
                    overlap = norm(currCyst - prevCysts(prevIdx,:)) <= radius + prevRads(prevIdx);
                    if overlap
                        % Overlap detected, don't generate this cyst
                        break
                    end
                    prevIdx = prevIdx + 1;
                end %while checking previous cysts
                
                if ~overlap % There's no overlap, so we can generate this cyst
                    % Set conductivity to inArea for all nodes within cyst radius (fine mesh)
                    for i=1:nodes
                        if (((p1(1,i)-xTemp)^2+(p1(2,i)-yTemp)^2)^.5<=radius)
                            sigTrue(i)=inArea;
                        end
                    end %end fine mesh loop
                    
                    % Set conductivity to inArea for all nodes within cyst radius (coarse mesh)
                    for i=1:nodes1
                        if (((p(1,i)-xTemp)^2+(p(2,i)-yTemp)^2)^.5<=radius)
                            sigTrue1(i)=inArea;
                        end
                    end %end coarse mesh loop
                end %check for overlap
            else % We have not added a cyst yet, so we should add the first one
                % Set conductivity to inArea for all nodes within cyst radius (fine mesh)
                for i=1:nodes
                    if (((p1(1,i)-xTemp)^2+(p1(2,i)-yTemp)^2)^.5<=radius)
                        sigTrue(i)=inArea;
                    end
                end %end fine mesh loop
                
                % Set conductivity to inArea for all nodes within cyst radius (coarse mesh)
                for i=1:nodes1
                    if (((p(1,i)-xTemp)^2+(p(2,i)-yTemp)^2)^.5<=radius)
                        sigTrue1(i)=inArea;
                    end
                end %end coarse mesh loop
            end %have we added one cyst?
            
            % After computing and adding the cyst, add currCyst to prevCysts
            if ~overlap
                prevCysts = [prevCysts; currCyst];
                prevRads = [prevRads; radius];
                waaRad = waaRad + 1;
            end

        end %while numCysts loop

    end %are we allowing overlap?

    % Plot and save conductivity distribution (fine mesh)
    figure('Visible','off')
    pdeplot(p1,e1,t1,'xydata',sigTrue,'mesh','off')
    colormap(jet)
    title('sig True Fine')
    
    % Directory name for saving figures
    dirName = 'ThreeCystsFigs\';
    [status, msgSuppressed] = mkdir(dirName);
    filename = [[dirName, 'fineSample'] , num2str(waaInst), '.png'];
    saveas(gcf, filename)

    % Plot and save conductivity distribution (coarse mesh)
    figure('Visible','off')
    pdeplot(p,e,t,'xydata',sigTrue1,'mesh','off')
    colormap(jet)
    title('sig True Coarse')
    filename = [[dirName, 'coarseSample'] , num2str(waaInst), '.png'];
    saveas(gcf, filename)

    %% Simulate Forward Problems
    % Set Neumann boundary conditions
    numSource=1;  % Number of source patterns
    nvec=cell(numSource,1);
    nvec{1}='sin(1*atan(y./x)+((sign(x)-1)/2)*pi*1)';
    % Additional source patterns can be added:
    % nvec{2}='sin(2*atan(y./x)+((sign(x)-1)/2)*pi*2)';
    % nvec{3}='sin(3*atan(y./x)+((sign(x)-1)/2)*pi*3)';
    % nvec{4}='sin(4*atan(y./x)+((sign(x)-1)/2)*pi*4)';
    % nvec{5}='sin(5*atan(y./x)+((sign(x)-1)/2)*pi*5)';

    % Create boundary condition matrices             
    bmat=cell(numSource,1);
    for i=1:numSource
        bTemp=[1 0 1 length(nvec{i}) '0' nvec{i}]';
        bmat{i} = repmat(bTemp,1,4);
    end

    % Matrix of boundary measurements
    phiMat=zeros(length(boundInd),numSource);

    % Simulate all experiments using fine mesh
    [K,M,F]=assema(p1,t1,pdeintrp(p1,t1,sigTrue),0,0);

    % For each source, get solution uN
    for i=1:numSource
        b=bmat{i};
        [Q,G,H,R]=assemb(b,p1,e1);
        [KN,FN]=assempde(K,M,F,Q,G,H,R);
            
        % Enforce boundary sum to zero (for uniqueness of solution)
        vec = zeros(1,nodes);
        vec(boundInd) = 1/length(boundInd);
        KN=[KN;vec];
        FN=[FN;0];
        uN=KN\FN;
        
        uNSave = [uNSave, uN];  % Store full uN vector for each instance

        % Optional plotting (commented out)
        % pdeplot(p,e,t,'xydata',uN,'mesh','on')
        % colormap(jet)
        % pause;
        
        % Grab boundary measurements
        phi=uN(boundInd); 
            
        % Store boundary values for each source
        phiMat(:,i)=phi;
    end

    % Add noise to boundary measurements to simulate realistic measurement noise
    noise=.01*max(max(abs(phiMat)))*randn(size(phiMat));
    phiMat=phiMat+noise;
    
    %%
    % Solve forward problem on coarse mesh (for inverse problem)
    phiMatN=EITForwardSolveRes(sigTrue1,p,e,t,bmat,boundInd);

    % Solve forward problem on fine mesh (for forward problem)
    boundInd1=e1(1,:);
    phiMatN1 = EITForwardSolveRes(sigTrue,p1,e1,t1,bmat,boundInd1);

    % Get coordinates of boundary points
    boundaryPoints = p(:, boundInd);    % Coarse mesh boundary points
    boundaryPoints1 = p1(:, boundInd1); % Fine mesh boundary points

    % Save off sigTrue and phiMatN
    % Rows are the points, columns are the nth instance/epoch
    sigCoarseSave = [sigCoarseSave, sigTrue1];
    phiCoarseSave = [phiCoarseSave, phiMatN];

    sigFineSave = [sigFineSave, sigTrue];
    phiFineSave = [phiFineSave, phiMatN1];
    
    disp(waaInst)  % Display progress
end %outer epoch loop (numInstances)

% Save all generated data to .mat files
dirName = 'ThreeCystsSamples\';
[status, msgSuppressed] = mkdir(dirName);
save([dirName,'CoarseGridPoints.mat'], 'boundaryPoints', "p")
save([dirName,'FineGridPoints.mat'], 'boundaryPoints1', "p1")
save([dirName,'EIT_CoarseSamples.mat'], 'sigCoarseSave',"phiCoarseSave")
save([dirName,'EIT_FineSamples.mat'], 'sigFineSave',"phiFineSave")
save([dirName,'EIT_uN_Samples.mat'], 'uNSave');

