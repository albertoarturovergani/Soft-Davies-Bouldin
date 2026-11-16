root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'scripts','matlab'));
source = fullfile(root,'data','original');
names = {'cancer','crab','glass','iris','ovarian','simpleclusters','thyroid','wine'};
maxK = 5;
runs = 3;
for a = 1:numel(names)
    filename = fullfile(source,[names{a} '.mat']);
    if ~isfile(filename)
        fprintf('Skipping %s: place the original file in data/original\n',names{a});
        continue;
    end
    data = load(filename,'Data','T');
    X = data.Data;
    trueK = size(data.T,1);
    fprintf('%s: %d samples, %d features, ground-truth K=%d\n',names{a},size(X,1),size(X,2),trueK);
    for k = 2:min(maxK,size(X,1)-1)
        scores = zeros(runs,2);
        for r = 1:runs
            [centers,U] = sdb.fuzzy_c_means(X,k,2,r);
            [~,hard] = max(U,[],2);
            scores(r,1) = sdb.soft_davies_bouldin(X,centers,U);
            scores(r,2) = sdb.davies_bouldin_crisp(X,hard);
        end
        fprintf('K=%d SDB=%.6f DB-hard=%.6f\n',k,mean(scores(:,1)),mean(scores(:,2)));
    end
end
