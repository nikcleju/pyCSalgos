disp('Eig');tic;data=rand(500,500);eig(data);toc;
disp('Svd');tic;data=rand(1000,1000);[u,s,v]=svd(data);s=svd(data);toc;
disp('Inv');tic;data=rand(1000,1000);result=inv(data);toc;
disp('Det');tic;data=rand(1000,1000);result=det(data);toc;
disp('Dot');tic;a=rand(1000,1000);b=inv(a);result=a*b-eye(1000);toc;
disp('Done');
