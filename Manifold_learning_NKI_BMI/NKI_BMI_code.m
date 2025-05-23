%% Prepare data
clear;clc; close all;

HeadPath = '/camin1/yrjang/NKI_BMI';
DataPath = '/camin2/Database/eNKI/DATA';

ParcelName = 'Schaefer_200';
if strcmp(ParcelName, 'Schaefer_100') == 1
    NumROI = 100;
elseif strcmp(ParcelName, 'Schaefer_200') == 1
    NumROI = 200;
elseif strcmp(ParcelName, 'Schaefer_300') == 1
    NumROI = 300;
elseif strcmp(ParcelName, 'Schaefer_400') == 1
    NumROI = 400;
end
OutPath = strcat(HeadPath,'/analysis_NKI/',ParcelName);
if exist(OutPath) == 0
    mkdir(OutPath);
end

[~,~,raw] = xlsread(strcat(HeadPath,'/Enhanced_NKI.xlsx'),'data');
subID = raw(3:end,1);
age = cell2mat(raw(3:end,4));
sex = cell2mat(raw(3:end,5));
bmi = cell2mat(raw(3:end,6));
whr = cell2mat(raw(3:end,7));
wc = cell2mat(raw(3:end,10));
edeq = cell2mat(raw(3:end,21:24));
tfeq = cell2mat(raw(3:end,25:27));
NumSubj = length(subID);

SC_tot = zeros(14+NumROI, 14+NumROI, NumSubj);
SC_EL_tot = zeros(14+NumROI, 14+NumROI, NumSubj);
FC_tot = zeros(NumROI, NumROI, NumSubj);
for ns = 1:NumSubj
    disp(strcat(['ns = ',int2str(ns), ' -- ', subID{ns}]));

    load(strcat(DataPath,'/',subID{ns},'/DIFF_137/dwi/connectome/dwi.mat'));       % sctx(14) + ctx
    eval(['SC_tot(:,:,ns) = SC.SctxCtx.schaefer_',int2str(NumROI),';']);
    eval(['SC_EL_tot(:,:,ns) = SC_EL.SctxCtx.schaefer_',int2str(NumROI),';']);

    load(strcat(DataPath,'/',subID{ns},'/REST_645/Results/Surface/ts_ctx.mat'));   % ctx
    eval(['R = corr(ts_ctx.schaefer_',int2str(NumROI),''');']);
    Z = atanh(R);
    Z(1:NumROI+1:end) = 0;
    FC_tot(:,:,ns) = Z;
end
save(strcat(OutPath,'/data.mat'),'SC_tot','SC_EL_tot','FC_tot');

%% Generate gradients & gradient eccentricity & gradient differentiation
clear;clc; close all;

load('/camin1/public/caminsoft/matlab_toolbox/default/colormap/coolwhitewarm.mat');

HeadPath = '/camin1/yrjang/NKI_BMI';
DataPath = '/camin2/Database/eNKI/DATA';

ParcelName = 'Schaefer_200';
if strcmp(ParcelName, 'Schaefer_100') == 1
    NumROI = 100;
elseif strcmp(ParcelName, 'Schaefer_200') == 1
    NumROI = 200;
elseif strcmp(ParcelName, 'Schaefer_300') == 1
    NumROI = 300;
elseif strcmp(ParcelName, 'Schaefer_400') == 1
    NumROI = 400;
end
OutPath = strcat(HeadPath,'/analysis_NKI/',ParcelName);
if exist(OutPath) == 0
    mkdir(OutPath);
end

[~,~,raw] = xlsread(strcat(HeadPath,'/Enhanced_NKI.xlsx'),'data');
subID = raw(3:end,1);
age = cell2mat(raw(3:end,4));
sex = cell2mat(raw(3:end,5));
bmi = cell2mat(raw(3:end,6));
whr = cell2mat(raw(3:end,7));
wc = cell2mat(raw(3:end,10));
edeq = cell2mat(raw(3:end,21:24));
tfeq = cell2mat(raw(3:end,25:27));
NumSubj = length(subID);

% load data
load(strcat(OutPath,'/data.mat'));

% load gradient template
load(strcat('/camin1/public/caminsoft/HCP_gradients/SC_LRcomb_schaefer_',int2str(NumROI),'.mat'));
load(strcat('/camin1/public/caminsoft/HCP_gradients/FC_schaefer_',int2str(NumROI),'.mat'));

% load surfaces
[surf_lh, surf_rh] = load_conte69();
labeling = load_parcellation('schaefer',NumROI);
exp_tmp = ['parc = labeling.schaefer_',int2str(NumROI),';'];
eval(exp_tmp);

% generate gradients
ncompo = 5;
ncompo_select = 3;

% template gradients
sp_SC = 0;
sp_FC = 90;

% SC
EL = SC_EL_tot(15:end,15:end,:);
EL(EL==0) = nan; % edgeLengths_40M.txt
el_group = nanmean(EL(1:NumROI,1:NumROI,:),3);
el_group(isnan(el_group)) = 0;
el_group(1:size(el_group)+1:end) = 0;
hemiid = [ones(1, NumROI/2) ones(1, NumROI/2)*2]';
temp = SC_tot(15:end,15:end,:);
G = fcn_distance_dependent_threshold(temp, el_group, hemiid, 1);
SC_mean = G .* nanmean(temp,3); % weight by group average

temp_gm = gradSC_HCP;
W = SC_mean;
zeroidx = find(sum(W) == 0);
if isempty(zeroidx) ~= 1
    in_mat = W;   in_mat(zeroidx,:) = [];   in_mat(:,zeroidx) =[];
    temp = temp_gm; temp(zeroidx,:) = [];
    gm = GradientMaps('kernel', 'na', 'approach', 'dm', 'alignment', 'pa', 'n_components', ncompo);
    gm = gm.fit(in_mat, 'sparsity', sp_SC, 'reference', temp, 'niterations',100);
    gm_align = gm.aligned{1};
    for zi = 1:length(zeroidx)
        insert0 = zeros(1,ncompo);
        gm_align = [gm_align(1:zeroidx(zi)-1,:); insert0; gm_align(zeroidx(zi):end,:)];
    end
else
    in_mat = W;
    gm = GradientMaps('kernel', 'na', 'approach', 'dm', 'alignment', 'pa', 'n_components', ncompo);
    gm = gm.fit(in_mat, 'sparsity', sp_SC, 'reference', temp_gm, 'niterations',100);
    gm_align = gm.aligned{1};
end
gm_temp_SC = gm_align;


% FC
FC_mean = mean(FC_tot,3);

temp_gm = gradFC_HCP;
W = FC_mean;
zeroidx = find(sum(W) == 0);
if isempty(zeroidx) ~= 1
    in_mat = W;   in_mat(zeroidx,:) = [];   in_mat(:,zeroidx) =[];
    temp = temp_gm; temp(zeroidx,:) = [];
    gm = GradientMaps('kernel', 'na', 'approach', 'dm', 'alignment', 'pa', 'n_components', ncompo);
    gm = gm.fit(in_mat, 'sparsity', sp_FC, 'reference', temp, 'niterations',100);
    gm_align = gm.aligned{1};
    for zi = 1:length(zeroidx)
        insert0 = zeros(1,ncompo);
        gm_align = [gm_align(1:zeroidx(zi)-1,:); insert0; gm_align(zeroidx(zi):end,:)];
    end
else
    in_mat = W;
    gm = GradientMaps('kernel', 'na', 'approach', 'dm', 'alignment', 'pa', 'n_components', ncompo);
    gm = gm.fit(in_mat, 'sparsity', sp_FC, 'reference', temp_gm, 'niterations',100);
    gm_align = gm.aligned{1};
end
gm_temp_FC = gm_align;

LT = {'SC-G1','SC-G2','SC-G3'};
W = [gm_temp_SC(:,1), gm_temp_SC(:,2), gm_temp_SC(:,3)];
obj = plot_hemispheres(W, {surf_lh,surf_rh}, 'labeltext', LT,'views','lm','parcellation',parc);
obj.colorlimits([-0.1 0.1]);
cmap = coolwhitewarm; cmap(1,:) = [0 0 0];
obj.colormaps(cmap)
obj.labels('FontSize',10)

LT = {'FC-G1','FC-G2','FC-G3'};
W = [gm_temp_FC(:,1), gm_temp_FC(:,2), gm_temp_FC(:,3)];
obj = plot_hemispheres(W, {surf_lh,surf_rh}, 'labeltext', LT,'views','lm','parcellation',parc);
obj.colorlimits([-0.15 0.15]);
cmap = coolwhitewarm; cmap(1,:) = [0 0 0];
obj.colormaps(cmap)
obj.labels('FontSize',10)



% individual gradients
gm_ind_SC = cell(NumSubj,1);
gm_ind_FC = cell(NumSubj,1);
for ns = 1:NumSubj
    disp(strcat(['ns = ',int2str(ns)]));
    
    % SC
    temp_gm = gm_temp_SC;
    W = SC_tot(15:end,15:end,ns);
    
    zeroidx = find(sum(W) == 0);
    if isempty(zeroidx) ~= 1
        in_mat = W;   in_mat(zeroidx,:) = [];   in_mat(:,zeroidx) =[];
        temp = temp_gm; temp(zeroidx,:) = [];
        gm = GradientMaps('kernel', 'na', 'approach', 'dm', 'alignment', 'pa', 'n_components', ncompo);
        gm = gm.fit(in_mat, 'sparsity', sp_SC, 'reference', temp, 'niterations',100);
        gm_align = gm.aligned{1};
        for zi = 1:length(zeroidx)
            insert0 = zeros(1,ncompo);
            gm_align = [gm_align(1:zeroidx(zi)-1,:); insert0; gm_align(zeroidx(zi):end,:)];
        end
    else
        in_mat = W;
        gm = GradientMaps('kernel', 'na', 'approach', 'dm', 'alignment', 'pa', 'n_components', ncompo);
        gm = gm.fit(in_mat, 'sparsity', sp_SC, 'reference', temp_gm, 'niterations',100);
        gm_align = gm.aligned{1};
    end
    gm_ind_SC{ns,1} = gm_align;


    % FC
    temp_gm = gm_temp_FC;
    W = FC_tot(:,:,ns);
    
    zeroidx = find(sum(W) == 0);
    if isempty(zeroidx) ~= 1
        in_mat = W;   in_mat(zeroidx,:) = [];   in_mat(:,zeroidx) =[];
        temp = temp_gm; temp(zeroidx,:) = [];
        gm = GradientMaps('kernel', 'na', 'approach', 'dm', 'alignment', 'pa', 'n_components', ncompo);
        gm = gm.fit(in_mat, 'sparsity', sp_FC, 'reference', temp, 'niterations',100);
        gm_align = gm.aligned{1};
        for zi = 1:length(zeroidx)
            insert0 = zeros(1,ncompo);
            gm_align = [gm_align(1:zeroidx(zi)-1,:); insert0; gm_align(zeroidx(zi):end,:)];
        end
    else
        in_mat = W;
        gm = GradientMaps('kernel', 'na', 'approach', 'dm', 'alignment', 'pa', 'n_components', ncompo);
        gm = gm.fit(in_mat, 'sparsity', sp_FC, 'reference', temp_gm, 'niterations',100);
        gm_align = gm.aligned{1};
    end
    gm_ind_FC{ns,1} = gm_align;
end
save(strcat(OutPath,'/gm_ind.mat'), 'gm_temp_SC', 'gm_temp_FC', 'gm_ind_SC', 'gm_ind_FC');

LT = {'SC-G1','SC-G2','SC-G3'};
W = [gm_ind_SC{1}(:,1), gm_ind_SC{1}(:,2), gm_ind_SC{1}(:,3)];
obj = plot_hemispheres(W, {surf_lh,surf_rh}, 'labeltext', LT,'views','lm','parcellation',parc);
obj.colorlimits([-0.1 0.1]);
cmap = coolwhitewarm; cmap(1,:) = [0 0 0];
obj.colormaps(cmap)
obj.labels('FontSize',10)

LT = {'FC-G1','FC-G2','FC-G3'};
W = [gm_ind_FC{1}(:,1), gm_ind_FC{1}(:,2), gm_ind_FC{1}(:,3)];
obj = plot_hemispheres(W, {surf_lh,surf_rh}, 'labeltext', LT,'views','lm','parcellation',parc);
obj.colorlimits([-0.15 0.15]);
cmap = coolwhitewarm; cmap(1,:) = [0 0 0];
obj.colormaps(cmap)
obj.labels('FontSize',10)


% gradient eccentricity
template_center_SC = mean(gradSC_HCP(:,1:ncompo_select));
template_center_FC = mean(gradFC_HCP(:,1:ncompo_select));

ge_ind_SC = zeros(NumSubj, NumROI);
ge_ind_FC = zeros(NumSubj, NumROI);
for ns = 1:NumSubj
    dist = gm_ind_SC{ns,1}(:,1:ncompo_select) - repmat(template_center_SC,[NumROI,1]);
    ge_ind_SC(ns,:) = sqrt(sum(dist.^2,2));

    dist = gm_ind_FC{ns,1}(:,1:ncompo_select) - repmat(template_center_FC,[NumROI,1]);
    ge_ind_FC(ns,:) = sqrt(sum(dist.^2,2));
end
save(strcat(OutPath,'/ge_ind.mat'), 'ge_ind_SC', 'ge_ind_FC');

LT = {'SC-mean GE','FC-mean GE'};
W = [mean(ge_ind_SC)', mean(ge_ind_FC)'];
obj = plot_hemispheres(W, {surf_lh,surf_rh}, 'labeltext', LT,'views','lm','parcellation',parc);
obj.colorlimits([0 0.08; 0 0.18]);
cmap = coolwhitewarm; cmap(1,:) = [0 0 0];
obj.colormaps(cmap)
obj.labels('FontSize',10)


% gradient differentiation
gd_ind_SC = zeros(NumSubj, NumROI, NumROI);
gd_ind_FC = zeros(NumSubj, NumROI, NumROI);
for ns = 1:NumSubj
    % SC
    differ = zeros(NumROI, NumROI);
    for nr1 = 1:NumROI
        for nr2 = 1:NumROI
            seed = gm_ind_SC{ns,1}(nr1,1:ncompo_select);
            targ = gm_ind_SC{ns,1}(nr2,1:ncompo_select);
            differ(nr1,nr2) = norm(seed-targ);
        end
    end
    gd_ind_SC(ns,:,:) = differ;

    % FC
    differ = zeros(NumROI, NumROI);
    for nr1 = 1:NumROI
        for nr2 = 1:NumROI
            seed = gm_ind_FC{ns,1}(nr1,1:ncompo_select);
            targ = gm_ind_FC{ns,1}(nr2,1:ncompo_select);
            differ(nr1,nr2) = norm(seed-targ);
        end
    end
    gd_ind_FC(ns,:,:) = differ;
end
save(strcat(OutPath,'/gd_ind.mat'), 'gd_ind_SC', 'gd_ind_FC');

figure; imagesc(squeeze(gd_ind_SC(1,:,:))); colorbar; caxis([0 0.15]); colormap(coolwhitewarm);
figure; imagesc(squeeze(gd_ind_FC(1,:,:))); colorbar; caxis([0 0.25]); colormap(coolwhitewarm);

%% Association with BMI (eccentricity)
clear;clc; close all;
addpath('/camin1/public/caminsoft/matlab_toolbox');
load('/camin1/public/caminsoft/matlab_toolbox/default/colormap/coolwhitewarm.mat');

HeadPath = '/camin1/yrjang/NKI_BMI';
DataPath = '/camin2/Database/eNKI/DATA';

ParcelName = 'Schaefer_200';
if strcmp(ParcelName, 'Schaefer_100') == 1
    NumROI = 100;
elseif strcmp(ParcelName, 'Schaefer_200') == 1
    NumROI = 200;
elseif strcmp(ParcelName, 'Schaefer_300') == 1
    NumROI = 300;
elseif strcmp(ParcelName, 'Schaefer_400') == 1
    NumROI = 400;
end
OutPath = strcat(HeadPath,'/analysis_NKI/',ParcelName);
if exist(OutPath) == 0
    mkdir(OutPath);
end

[~,~,raw] = xlsread(strcat(HeadPath,'/Enhanced_NKI.xlsx'),'data');
subID = raw(3:end,1);
age = cell2mat(raw(3:end,4));
sex = cell2mat(raw(3:end,5));
bmi = cell2mat(raw(3:end,6));
whr = cell2mat(raw(3:end,7));
wc = cell2mat(raw(3:end,10));
edeq = cell2mat(raw(3:end,21:24));
tfeq = cell2mat(raw(3:end,25:27));
NumSubj = length(subID);

% Add anxiety/depression, FD
anx_d = cell2mat(raw(3:end,38));
dep = cell2mat(raw(3:end,20));
load(strcat(HeadPath,'/FD.mat'));
FD = FD(:);

% load data
load(strcat(OutPath,'/ge_ind.mat'));    % NumSubj x NumROI  

% load surfaces
[surf_lh, surf_rh] = load_conte69();
labeling = load_parcellation('schaefer',NumROI);
exp_tmp = ['parc = labeling.schaefer_',int2str(NumROI),';'];
eval(exp_tmp);


%%% Association with BMI
% prepare model
Age_term = term(age);
temp = zeros(NumSubj,2); temp(find(sex == 'M'),1) = 1; temp(find(sex == 'F'),2) = 1;
Sex_term = term(temp, {'M','F'});
BMI_term = term(bmi,{'BMI'});
anx_term = term(anx_d,{'Anx'});
dep_term = term(dep,{'Dep'});
fd_term = term(FD,{'FD'});

% fit the model
M = 1 + Age_term + Sex_term + BMI_term;
M1 = 1 + Age_term + Sex_term + BMI_term + dep_term + anx_term;
M2 = 1 + Age_term + Sex_term + BMI_term + fd_term;

%%% SC
W = ge_ind_SC;
slm_SC = SurfStatLinMod(W,M2);
slm_SC_org = SurfStatLinMod(W,M);
slm_SC_ = SurfStatLinMod(W, M1);

% correlation
slm_SC = SurfStatT(slm_SC, bmi);

%slm_SC_org = SurfStatT(slm_SC_org,bmi);
%slm_SC_fd = SurfStatT(slm_SC_fd,bmi);

pval_SC = 1-tcdf(slm_SC.t, slm_SC.df);
idx = find(pval_SC>0.5);
pval_SC(idx) = 1 - pval_SC(idx);
pcor_SC = pval_adjust(pval_SC, 'fdr');
sig_idx_SC = find(pcor_SC<0.05);
sig_t_SC = zeros(NumROI,1);    sig_t_SC(sig_idx_SC) = slm_SC.t(sig_idx_SC);

sig_idx_SC_pos = find(sig_t_SC>0);
sig_idx_SC_neg = find(sig_t_SC<0);

disp(slm_SC.t);

LT = {'totT-SC','sigT-FC'};
W = [slm_SC.t', sig_t_SC];
obj = plot_hemispheres(W, {surf_lh,surf_rh}, 'labeltext', LT,'views','lm','parcellation',parc);
obj.colorlimits([-5 5; -5 5]);
cmap = coolwhitewarm; cmap1(1,:) = [1 1 1];
obj.colormaps({cmap});
obj.labels('FontSize',10);


% SC gradient eccentricity expands in the DLPFC/DMPFC for higher BMI
% the regions are % more distributed from other regions within the manifold space 
% (perhaps, more segregated)

%% Association with BMI (differentiation)
clear;clc; close all;

load('/camin1/public/caminsoft/matlab_toolbox/default/colormap/coolwhitewarm.mat');

HeadPath = '/camin1/yrjang/NKI_BMI';
DataPath = '/camin2/Database/eNKI/DATA';

ParcelName = 'Schaefer_300';
if strcmp(ParcelName, 'Schaefer_100') == 1
    NumROI = 100;
elseif strcmp(ParcelName, 'Schaefer_200') == 1
    NumROI = 200;
elseif strcmp(ParcelName, 'Schaefer_300') == 1
    NumROI = 300;
elseif strcmp(ParcelName, 'Schaefer_400') == 1
    NumROI = 400;
end
OutPath = strcat(HeadPath,'/analysis_NKI/',ParcelName);
if exist(OutPath) == 0
    mkdir(OutPath);
end

[~,~,raw] = xlsread(strcat(HeadPath,'/Enhanced_NKI.xlsx'),'data');
subID = raw(3:end,1);
age = cell2mat(raw(3:end,4));
sex = cell2mat(raw(3:end,5));
bmi = cell2mat(raw(3:end,6));
whr = cell2mat(raw(3:end,7));
wc = cell2mat(raw(3:end,10));
edeq = cell2mat(raw(3:end,21:24));
tfeq = cell2mat(raw(3:end,25:27));
NumSubj = length(subID);

% Add anxiety/depression, FD
anx_d = cell2mat(raw(3:end,38));
dep = cell2mat(raw(3:end,20));
load(strcat(HeadPath,'/FD.mat'));
FD = FD(:);

% load data
load(strcat(OutPath,'/gd_ind.mat'));

% load surfaces
[surf_lh, surf_rh] = load_conte69();
labeling = load_parcellation('schaefer',NumROI);
exp_tmp = ['parc = labeling.schaefer_',int2str(NumROI),';'];
eval(exp_tmp);



%%% Association with BMI
[~,~,sch] = xlsread(strcat('/camin1/public/caminsoft/Atlas/sch2atlas/sch2yeo7_',int2str(NumROI),'.xlsx'));
sch_net = cell2mat(sch(:,2));
NumNet = numel(unique(sch_net));

EdgeMat = triu(ones(NumNet),1);
EdgeIdx = find(EdgeMat == 1);
NumEdge = length(EdgeIdx);

gd_ind_SC_net = zeros(NumSubj, NumNet, NumNet);
gd_ind_FC_net = zeros(NumSubj, NumNet, NumNet);
for ns = 1:NumSubj
    mat_net_SC = zeros(NumNet);
    mat_net_FC = zeros(NumNet);
    for nn1 = 1:NumNet
        for nn2 = 1:NumNet
            idx1 = find(sch_net == nn1);
            idx2 = find(sch_net == nn2);

            mat = squeeze(gd_ind_SC(ns,:,:));
            mat_sub = mat(idx1,idx2);
            mat_net_SC(nn1,nn2) = mean(mat_sub(:));

            mat = squeeze(gd_ind_FC(ns,:,:));
            mat_sub = mat(idx1,idx2);
            mat_net_FC(nn1,nn2) = mean(mat_sub(:));
        end
    end
    gd_ind_SC_net(ns,:,:) = mat_net_SC;
    gd_ind_FC_net(ns,:,:) = mat_net_FC;
end


gd_ind_SC_tot = zeros(NumSubj, NumEdge);
gd_ind_FC_tot = zeros(NumSubj, NumEdge);
for ns = 1:NumSubj
    mat = gd_ind_SC_net(ns,:,:);
    gd_ind_SC_tot(ns,:) = mat(EdgeIdx);
    mat = gd_ind_FC_net(ns,:,:);
    gd_ind_FC_tot(ns,:) = mat(EdgeIdx);
end


% prepare model
Age_term = term(age);
temp = zeros(NumSubj,2); temp(find(sex == 'M'),1) = 1; temp(find(sex == 'F'),2) = 1;
Sex_term = term(temp, {'M','F'});
BMI_term = term(bmi,{'BMI'});
anx_term = term(anx_d);
dep_term = term(dep,{'Dep'});
fd_term = term(FD,{'FD'});

% fit the model
M = 1 + Age_term + Sex_term + BMI_term;
M1 = 1 + Age_term + Sex_term + BMI_term + dep_term + anx_term;
M2 = 1 + Age_term + Sex_term + BMI_term + fd_term;

%%% SC
W = gd_ind_SC_tot;
slm_SC_fd = SurfStatLinMod(W,M2);
%slm_SC_org = SurfStatLinMod(W,M);
%slm_SC = SurfStatLinMod(W, M1);

% correlation
slm_SC = SurfStatT(slm_SC_fd, bmi);

%slm_SC_org = SurfStatT(slm_SC_org,bmi);
%slm_SC_fd = SurfStatT(slm_SC_fd,bmi);

pval_SC = 1-tcdf(slm_SC.t, slm_SC.df);
idx = find(pval_SC>0.5);
pval_SC(idx) = 1 - pval_SC(idx);
pcor_SC = pval_adjust(pval_SC, 'fdr');
sig_idx_SC = find(pcor_SC<0.05);
sig_t_SC = zeros(NumEdge,1);    sig_t_SC(sig_idx_SC) = slm_SC.t(sig_idx_SC);
tot_mat_SC = zeros(NumNet);
tot_mat_SC(EdgeIdx) = slm_SC.t;
sig_mat_SC = zeros(NumNet);
sig_mat_SC(EdgeIdx) = sig_t_SC;


cmap = coolwhitewarm; cmap(1,:) = [1 1 1];

figure('Position',[100 0 1100 400]);
subplot(1,2,1); imagesc(tot_mat_SC); title({'totT-SC',''}); colormap(cmap); caxis([-5 5]); colorbar;
subplot(1,2,2); imagesc(sig_mat_SC); title({'sigT-SC',''}); colormap(cmap); caxis([-5 5]); colorbar;
set(gcf,'color','w');


% SC gradient differentiation within transmodal and between transmodal-sensory regions 
% greater differentiation in SC -> greater BMI
% individuals with high BMI show network segregation in transmodal regions


%%
% SC coupled gradient eccentricity and differentiation consistently show 
% that individuals with high BMI show more segregation of DLPFC/DMPFC 
% from other netowrks

%% Efficiency and Diffusion
clear;clc; close all;

load('/camin1/public/caminsoft/matlab_toolbox/default/colormap/coolwhitewarm.mat');

HeadPath = '/camin1/yrjang/NKI_BMI';
DataPath = '/camin2/Database/eNKI/DATA';


ParcelName = 'Schaefer_200';
if strcmp(ParcelName, 'Schaefer_100') == 1
    NumROI = 100;
elseif strcmp(ParcelName, 'Schaefer_200') == 1
    NumROI = 200;
elseif strcmp(ParcelName, 'Schaefer_300') == 1
    NumROI = 300;
elseif strcmp(ParcelName, 'Schaefer_400') == 1
    NumROI = 400;
end
OutPath = strcat(HeadPath,'/analysis_NKI/',ParcelName);
if exist(OutPath) == 0
    mkdir(OutPath);
end

[~,~,raw] = xlsread(strcat(HeadPath,'/Enhanced_NKI.xlsx'),'data');
subID = raw(3:end,1);
age = cell2mat(raw(3:end,4));
sex = cell2mat(raw(3:end,5));
bmi = cell2mat(raw(3:end,6));
whr = cell2mat(raw(3:end,7));
wc = cell2mat(raw(3:end,10));
edeq = cell2mat(raw(3:end,21:24));
tfeq = cell2mat(raw(3:end,25:27));
NumSubj = length(subID);


% % load data
% load(strcat(OutPath,'/data.mat'));
% 
% Comm.MFPT = zeros(NumSubj, NumROI, NumROI);
% Comm.PT = zeros(NumSubj, NumROI, NumROI);
% Comm.SI = zeros(NumSubj, NumROI, NumROI);
% Comm.NavPL = zeros(NumSubj, NumROI, NumROI);
% for ns = 1:NumSubj
%     disp(strcat(['ns = ',int2str(ns)]));
% 
%     W = SC_tot(15:end,15:end,ns)+0.00001;   % make a connected graph
%     L = W; E = find(W); L(E) = 1./L(E);
%     D = distance_wei(L);
% 
%     MFPT = mean_first_passage_time(W);
%     PT = path_transitivity(W,'inv');
%     SI = search_information(W,L);
%     [~, ~, NavPL, ~, ~] = navigation_wu(L,D);
% 
%     Comm.MFPT(ns,:,:) = MFPT;
%     Comm.PT(ns,:,:) = PT;
%     Comm.SI(ns,:,:) = SI;
%     Comm.NavPL(ns,:,:) = NavPL;
% end
% save(strcat(OutPath,'/SC_comm.mat'),'Comm');



%%% send-receive communication & BMI
load(strcat(OutPath,'/SC_comm.mat'));
Comm.NavPL(find(isinf(Comm.NavPL(:)) == 1)) = NaN;

[~,~,sch] = xlsread(strcat('/camin1/public/caminsoft/Atlas/sch2atlas/sch2yeo7_',int2str(NumROI),'.xlsx'));
sch_net = cell2mat(sch(:,2));
NumNet = numel(unique(sch_net));

idx_vn = find(sch_net == 1);
idx_smn = find(sch_net == 2);
idx_dan = find(sch_net == 3);
idx_van = find(sch_net == 4);
idx_lbn = find(sch_net == 5);
idx_fpn = find(sch_net == 6);
idx_dmn = find(sch_net == 7);

idx1 = idx_fpn;
idx2 = idx_dmn;
comm_send = nanmean(nanmean(Comm.SI(:,idx1,idx2),3),2);
comm_receive = nanmean(nanmean(Comm.SI(:,idx2,idx1),2),3);
[r,p] = corr(comm_send, bmi)
[r,p] = corr(comm_receive, bmi)

%% Associations with eating behaviors
clear;clc; close all;

load('/camin1/public/caminsoft/matlab_toolbox/default/colormap/coolwhitewarm.mat');

HeadPath = '/camin1/yrjang/NKI_BMI';
DataPath = '/camin2/Database/eNKI/DATA';


ParcelName = 'Schaefer_200';
if strcmp(ParcelName, 'Schaefer_100') == 1
    NumROI = 100;
elseif strcmp(ParcelName, 'Schaefer_200') == 1
    NumROI = 200;
elseif strcmp(ParcelName, 'Schaefer_300') == 1
    NumROI = 300;
elseif strcmp(ParcelName, 'Schaefer_400') == 1
    NumROI = 400;
end
OutPath = strcat(HeadPath,'/analysis_NKI/',ParcelName);
if exist(OutPath) == 0
    mkdir(OutPath);
end

[~,~,raw] = xlsread(strcat(HeadPath,'/Enhanced_NKI.xlsx'),'data');
subID = raw(3:end,1);
age = cell2mat(raw(3:end,4));
sex = cell2mat(raw(3:end,5));
bmi = cell2mat(raw(3:end,6));
whr = cell2mat(raw(3:end,7));
wc = cell2mat(raw(3:end,10));
edeq = cell2mat(raw(3:end,21:24));
tfeq = cell2mat(raw(3:end,25:27));
NumSubj = length(subID);


% Add anxiety/depression, FD
anx_d = cell2mat(raw(3:end,38));
dep = cell2mat(raw(3:end,20));
load(strcat(HeadPath,'/FD.mat'));
FD = FD(:);


% load data
load(strcat(OutPath,'/gm_ind.mat'));
load(strcat(OutPath,'/ge_ind.mat'));
load(strcat(OutPath,'/gd_ind.mat'));
load(strcat(OutPath,'/SC_comm.mat'));

% load surfaces
[surf_lh, surf_rh] = load_conte69();
labeling = load_parcellation('schaefer',NumROI);
exp_tmp = ['parc = labeling.schaefer_',int2str(NumROI),';'];
eval(exp_tmp);



sig_idx_SC_pos = [20	21	24	28	34	35	41	42	46	51	54	65	66	67	68	69	70	86	88	89	90	91	92	93	94	95	122	124	125	127	128	129	130	131	134	137	138	139	140	141	144	145	146	151	158	161	170	171	172	173	174	175	176	180	181	192	193	194	195	196	197];
sig_idx_SC_neg = [2	3	4	5	7	9	14	15	16	17	18	19	47	49	53	55	56	57	58	59	60	72	74	76	83	100	103	107	116	117	118	119	121	148	152	153	155	164	178	185	187	188	189];

sig_idx_SC_pos_anx = [20,21,23,24,26,28,34,35,41,42,43,45,46,51,54,65,66,67,68,69,70,86,88,89,90,91,92,93,94,95,122,124,125,127,128,129,130,131,134,137,138,139,140,141,142,144,145,146,151,156,158,161,166,170,171,172,173,174,175,176,180,181,192,193,194,195,196,197];
sig_idx_SC_neg_anx = [1,2,3,4,5,6,7,9,11,14,15,16,17,18,19,22,31,44,47,49,53,55,56,57,58,59,60,72,74,76,77,78,83,98,99,100,103,107,108,112,114,116,117,118,119,120,121,148,152,153,155,162,163,164,169,178,185,186,187,188,189,199];

sig_idx_SC_pos_fd = [20,21,23,24,26,28,34,35,41,42,43,45,46,51,54,65,66,67,68,69,70,86,88,89,90,91,92,93,94,95,122,124,125,127,128,129,130,131,134,137,138,139,140,141,142,144,145,146,151,156,158,161,166,170,171,172,173,174,175,176,180,181,192,193,194,195,196,197];
sig_idx_SC_neg_fd = [1,2,3,4,5,6,7,9,11,14,15,16,17,18,19,31,44,47,49,53,55,56,57,58,59,60,72,74,76,77,78,83,98,99,100,103,107,108,112,116,117,118,119,120,121,148,149,152,153,155,162,163,164,169,178,185,186,187,188,189,199];
sig_idx_SC_tot = [sig_idx_SC_pos_fd, sig_idx_SC_neg_fd];

sig_idx_SC = sig_idx_SC_tot;

eat = [tfeq, edeq];
tp = zeros(size(eat,2),3); % t, p,fdr
for e = 1:size(eat,2)
    % prepare model
    Age_term = term(age);
    temp = zeros(NumSubj,2); temp(find(sex == 'M'),1) = 1; temp(find(sex == 'F'),2) = 1;
    Sex_term = term(temp, {'M','F'});
    TFEQ_term = term(eat(:,e),{'TFEQ'});
    anx_term = term(anx_d,{'Anx'});
    dep_term = term(dep,{'Dep'});
    fd_term = term(FD,{'FD'});

    % fit the model
    M = 1 + Age_term + Sex_term + TFEQ_term + fd_term;
    

    %%% Gradients
    ncompo = 5;
    ncompo_select = 3;

    W = zeros(NumSubj, length(sig_idx_SC), ncompo_select);
    for ns = 1:NumSubj
        W(ns,:,:) = gm_ind_SC{ns}(sig_idx_SC,1:ncompo_select);
    end
    W = mean(W,2);
    slm_gm = SurfStatLinMod(W, M);

    % correlation
    slm_gm = SurfStatT(slm_gm, eat(:,e));
    pval_gm = 1-tcdf(slm_gm.t, slm_gm.df);

    tp(e,1) = slm_gm.t;
    tp(e,2) = pval_gm;
end
tp(:,3) = pval_adjust(tp(:,2),'fdr');

for e = 1:size(eat,2)
    disp(['t = ',num2str(tp(e,1),'%.3f'),', fdr = ',num2str(tp(e,3),'%.3f')]);
end



% ROI
data = zeros(NumROI,3);
data(sig_idx_SC_pos,1) = 1;
data(sig_idx_SC_neg,2) = 1;
data(sig_idx_SC_tot,3) = 1;
LT = {'pos','neg','tot'};
obj = plot_hemispheres(data, {surf_lh,surf_rh}, 'labeltext', LT,'views','lm','parcellation',parc);
obj.colorlimits([0 1]);
cmap = flipud(gray); cmap(1,:) = [1 1 1];
obj.colormaps(cmap)
obj.labels('FontSize',10)

% gradient values
data = zeros(NumROI,3);
data(sig_idx_SC_pos,:) = gm_temp_SC(sig_idx_SC_pos,1:3);
LT = {'pos-G1','pos-G2','pos-G3'};
obj = plot_hemispheres(data, {surf_lh,surf_rh}, 'labeltext', LT,'views','lm','parcellation',parc);
obj.colorlimits([-0.1 0.1]);
cmap = coolwhitewarm; cmap(1,:) = [0 0 0];
obj.colormaps(cmap)
obj.labels('FontSize',10)

data = zeros(NumROI,3);
data(sig_idx_SC_neg,:) = gm_temp_SC(sig_idx_SC_neg,1:3);
LT = {'neg-G1','neg-G2','neg-G3'};
obj = plot_hemispheres(data, {surf_lh,surf_rh}, 'labeltext', LT,'views','lm','parcellation',parc);
obj.colorlimits([-0.1 0.1]);
cmap = coolwhitewarm; cmap(1,:) = [0 0 0];
obj.colormaps(cmap)
obj.labels('FontSize',10)

data = zeros(NumROI,3);
data(sig_idx_SC_tot,:) = gm_temp_SC(sig_idx_SC_tot,1:3);
LT = {'tot-G1','tot-G2','tot-G3'};
obj = plot_hemispheres(data, {surf_lh,surf_rh}, 'labeltext', LT,'views','lm','parcellation',parc);
obj.colorlimits([-0.1 0.1]);
cmap = coolwhitewarm; cmap(1,:) = [0 0 0];
obj.colormaps(cmap)
obj.labels('FontSize',10)


