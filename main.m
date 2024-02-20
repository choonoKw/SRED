% do simulation using customized setup: if there is setup file, load the

% setting (org: Nt=4, Nr=8, N=32)
Nt = 4;
Nr = 8;
N = 32;
%% load data 
orgPath = pwd;
cd ..;
if ~exist('data_setup.mat','file')
    make_setup_degree(Nt,Nr,N,pwd)
end

load('data_setup.mat',"struct_c","struct_m","struct_k","struct_q",...
    "aqaqh","aqhaq","bqbqh","bqhbq","Upsilon")
% constant check
if any([Nt~=struct_c.Nt,Nr~=struct_c.Nr,N~=struct_c.N])
    error('Constants are not compatible')
end

cd(orgPath)


%% Constant Setting
% system setting

Nt = struct_c.Nt;
N = struct_c.N; % 32
Nr = struct_c.Nr;

% target information
M = struct_c.M;
lm = [struct_m.lm];
delta_m = [struct_m.delta_m];

% interference information

Lj = struct_c.Lj;


% Upsilon = eye(Nr*Lj,Nr*Lj);
%%
% siumlation setting
T = 150;
TT = 5e2;
epsilon_f = 1e-5;
epsilon = 1e-2;
rho_init = 1.5e0;
gamma = 1e0;
rhoM = 0; % for momentum
tt_small = 30;
fprintf("espilon = %.0e, epsilon_f = %.0e\n",epsilon,epsilon_f)

Delta_f = 1e-5;
ttt_max = 10;
tt_rho = 50;

gcp;
n_simul = 1;
while(1)
    % initialization
    phi = 2*pi*rand(Nt*N,1);
    s = 1/sqrt(Nt*N)*exp(1j*phi); % (NtN) X 1
    % s_tilde = ; % (LjNt) X 1
    S_tilde = reshape([s.', zeros(1,Nt*(lm(M)-lm(1)))].',Nt,Lj); % Nt X Lj
    
    s0 = s;
    % S_tilde = S0_tilde;
    rho0 = rho_init;
    
    sinr_wm_record = zeros(T,M);
    sinr_m_record = zeros(T,M);
    W_m_tilde = zeros(Nr,Lj,M);
    t_measure = tic;
    for t=1:T
        S_tilde = reshape([s.', zeros(1,Nt*(lm(M)-lm(1)))].',Nt,Lj); % Nt X Lj
        
        % Sigma(s)
        Sigma = make_Sigma_parfor(struct_c,struct_k,S_tilde,bqhbq);

        % Gamma_m(s) 
        Gamma_m = make_Gamma_m(struct_c,struct_m,S_tilde,aqhaq);
        % Psi_m(s)
        Psi_m = make_Psi_m(struct_c,struct_m,S_tilde,aqhaq,Sigma,Upsilon);

        % w_mOpt (closed-form solution)
        w_mList = zeros(Lj*Nr,M);
        H_m = zeros(Nt*N,Nt*N,M);
        parfor m=1:M
            [V, D] = eig(Psi_m(:,:,m)\Gamma_m(:,:,m));
            [~,i] = max(diag(D));
            w_mList(:,m) = V(:,i)/norm(V(:,i));
            W_m_tilde(:,:,m) = reshape(w_mList(:,m),Nr,Lj);
            
            % H_m(w_m)
            H_m(:,:,m) = make_Theta(struct_c,struct_m,W_m_tilde(:,:,m),aqaqh,m);
        end
        G_m = make_Phi_m_parfor(struct_c,struct_m,struct_k,w_mList,aqaqh,bqbqh,Upsilon);
        parfor m=1:M
            sinr_wm_record(t,m)= abs(s'*H_m(:,:,m)*s)/abs((s'*G_m(:,:,m)*s));
        end
        
        sinr_tt_record = zeros(TT,M);
        sinr_tt_record(1,:) = sinr_wm_record(t,:);
        f_record = zeros(TT+1,1);
        f_record(1) = sum(1./sinr_wm_record(t,:));
        rho = rho0;
        grad_f_old = zeros(Nt*N,1);
        for tt=1:TT
            grad_f_m = zeros(Nt*N,M);
            for m=1:M
                beta_m = 2/( (s'*H_m(:,:,m)*s)^2 );
                grad_f_m(:,m) = beta_m*imag( (...
                    (s'*H_m(:,:,m)*s)*G_m(:,:,m)*s...
                     - (s'*G_m(:,:,m)*s)*H_m(:,:,m)*s ).*conj(s) );
            end
            grad_f = sum(grad_f_m,2);
    
            % sinr evaulation
            if tt<tt_rho
                % ensure rho is appropriate
                phi0 = phi;
                f_record(tt+1) = f_record(1);
                for ttt=1:ttt_max
                    phi = phi0;
                    phi = phi - rho*grad_f;
                    s = 1/sqrt(Nt*N)*exp(1j*phi);

                    f_record(tt+1) = 0;
                    for m=1:M
                        sinr = abs(s'*H_m(:,:,m)*s)/abs((s'*G_m(:,:,m)*s));
                        sinr_tt_record(tt+1,m)  = sinr;
                        f_record(tt+1) = f_record(tt+1) + 1/sinr;
                    end

                    if f_record(tt+1)+Delta_f<f_record(tt)
                        break;
                    else
                        rho = rho*0.5;
                    end
                end
            else
                % phi update
                phi = phi - rho*grad_f;
                s = 1/sqrt(Nt*N)*exp(1j*phi);

                f_record(tt+1) = 0;
                for m=1:M
                    sinr = abs(s'*H_m(:,:,m)*s)/abs((s'*G_m(:,:,m)*s));
                    sinr_tt_record(tt+1,m)  = sinr;
                    f_record(tt+1) = f_record(tt+1) + 1/sinr;
                end
                
            end

            
            if abs(f_record(tt+1)-f_record(tt))<epsilon_f
                break;
            end

            grad_f_old = grad_f;
        end
    
        sinr_m_record(t,:) = sinr_tt_record(tt+1,:);
        if t>1
            if abs(min(sinr_m_record(t,:))-min(sinr_m_record(t-1,:)))<epsilon
                break;
            end
        end
        
        if t==1
            TT_save = tt+1;

            sinr_tt_record_save = sinr_tt_record;
            figure(1); clf; hold on;
            plot(10*log10(sinr_tt_record_save(:,1)),'r','LineWidth',2)
            plot(10*log10(sinr_tt_record_save(:,2)),'b','LineWidth',2)
            plot(10*log10(sinr_tt_record_save(:,3)),'k','LineWidth',2)
            grid on;
            legend(["1";"2";"3"],'Location','best')
            
            f_record_save = f_record;
            figure(2); 
            plot(20*log10(f_record_save),'LineWidth',2)
            grid on;
        end
        
        rho_init;
    end
    sinr_m_record = sinr_m_record(1:t,:);
    
    computationTime = toc(t_measure);
    sinr_m_record = sinr_m_record(1:t,:);
    sinr_worst = min(sinr_m_record(end,:));
    fprintf("\nsinr_worst = %.2f dB within %.2f seconds...\n",...
        10*log10(sinr_worst),computationTime)

    figure(3); clf; hold on;
    plot(10*log10(sinr_tt_record(:,1)),'r','LineWidth',2)
    plot(10*log10(sinr_tt_record(:,2)),'b','LineWidth',2)
    plot(10*log10(sinr_tt_record(:,3)),'k','LineWidth',2)
    grid on;
    legend(["1";"2";"3"],'Location','best')

    figure(4); 
    plot(20*log10(f_record),'LineWidth',2)
    grid on;

    figure(5); clf; hold on;
    plot(10*log10(sinr_m_record(:,1)),'r','LineWidth',2)
    plot(10*log10(sinr_m_record(:,2)),'b','LineWidth',2)
    plot(10*log10(sinr_m_record(:,3)),'k','LineWidth',2)
    grid on;
    legend(["1";"2";"3"],'Location','best')

    if and(16<10*log10(sinr_worst),computationTime<154.6)
        sinr_worst_dB = 10*log10(min(sinr_m_record(end,:)));
        save(fullfile(pwd,strcat('data_result',num2str(n_simul,'%02d'),'.mat')),...
            "t","computationTime","sinr_worst_dB","sinr_m_record",...
            "epsilon","epsilon_f","rho_init","s0","w_mList","s",...
            'TT_save','sinr_tt_record_save','f_record_save','tt_rho')
        n_simul = n_simul + 1;
    end
    clear phi s s0
    
end