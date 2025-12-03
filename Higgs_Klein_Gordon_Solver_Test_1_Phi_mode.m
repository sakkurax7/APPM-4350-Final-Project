%% Coupled Higgs-Klein-Gordon Equations Solver using ODE45
clear all; close all; clc;

%% Constants:
% Physical
P.m = 1.0;  
P.g = 0.1;   
P.lam = 1.0; 
P.v = 1.0;  

% Numerical:
L = 40.0; % Size of domain
Nx = 512; % number of spatial points
dx = L / (Nx - 1);
P.dx = dx;
P.Nx = Nx;
x = linspace(-L/2, L/2, Nx);

% Time
stop_time = 200; 
tspan = [0 stop_time];


%% Initial Conditions:
eps = 1e-4;    % small amplitude (linear regime)

% Equilibrium 
h_star = P.v;
phi_star = 0;

n_list = [1, 2, 4, 8];   
Nk = numel(n_list);

%% Intial Conditions
figure('Name','Initial Conditions for φ','Units','normalized','Position',[0.1 0.2 0.6 0.5]);
hold on;
grid on;

% Colors for different modes
colors = lines(Nk);

% Initialize arrays
k_vals = zeros(Nk,1);
phi_init_all = zeros(Nk, Nx);
labels = cell(Nk, 1);

% Generate and plot initial conditions for each mode
for jj = 1:Nk
    n = n_list(jj);
    k0 = 2*pi*n / L;
    k_vals(jj) = k0;
    mode = cos(k0 * x);
    labels{jj} = sprintf('n = %d (k = %.3f)', n, k0);

    phi_0 = phi_star + eps * mode;
    
    phi_init_all(jj, :) = phi_0;
    
    plot(x, phi_0, 'LineWidth', 1.5, 'Color', colors(jj,:));
end

% Configure the plot
xlabel('x', 'FontSize', 12);
ylabel('\phi(x, t=0)', 'FontSize', 12);
title('Initial Conditions for \phi Field', 'FontSize', 14);
legend(labels, 'Location', 'best', 'FontSize', 10);
xlim([-L/2, L/2]);



hold off;

%% Sweep over modes and compute omega
options = odeset('RelTol', 1e-7, 'AbsTol', 1e-9);

fprintf('Starting sweep over n values: %s\n', mat2str(n_list));

% Store results
omega_num = zeros(Nk,1);
omega_theory = zeros(Nk,1);

for jj = 1:Nk
    n = n_list(jj);
    k0 = k_vals(jj);
    
    % Use stored initial conditions
    phi_0 = phi_init_all(jj, :);
    dphi_dt_0 = zeros(size(x));
    h_0 = h_star * ones(size(x));
    dh_dt_0 = zeros(size(x));
    
    U0 = [phi_0, dphi_dt_0, h_0, dh_dt_0]';

    fprintf('  Running n=%d, k0=%.4f ... ', n, k0);
    tic;
    [t, U] = ode45(@(t,U) higgsKleinGordonODE(t, U, P), tspan, U0, options);
    elapsed = toc;
    fprintf('done (%.2f s, %d steps)\n', elapsed, length(t));

    % Reconstruct phi and h arrays (Nx x Nt)
    phi_sol = U(:, 1:Nx)';          % size (Nx x Nt)
    h_sol   = U(:, 2*Nx+1:3*Nx)';

    Aphi = zeros(length(t),1);
    cos_kx = cos(k0 * x).';    % Column vector
    
    for ti = 1:length(t)
        phi_t = phi_sol(:, ti);     % also column
        a_phi = (2/L) * trapz(x, phi_t .* cos_kx);
        Aphi(ti) = a_phi;
    end

    % Estimate omega from amplitude time series
    omega_est = estimate_frequency(t, Aphi);
    omega_num(jj) = omega_est;

    % Analytical effective mass for phi-mode at equilibrium h_star
    mphi2 = P.m^2 + P.g * h_star;         % m_eff^2 for phi
    omega_theory(jj) = sqrt(k0^2 + mphi2);

    fprintf('    ω_num = %.6f, ω_theory = %.6f, rel error = %.2e\n', ...
        omega_est, omega_theory(jj), abs(omega_est-omega_theory(jj))/omega_theory(jj));
end

%% Plot numeric vs theory
figure('Name','Dispersion: ω(k)','Units','normalized','Position',[0.2 0.2 0.5 0.5]);
plot(k_vals, omega_num, 'bo-', 'LineWidth', 1.5, 'MarkerSize',8); hold on;
plot(k_vals, omega_theory, 'r--s', 'LineWidth', 1.5, 'MarkerSize',8);
xlabel('k');
ylabel('\omega(k)');
legend('Numerical \omega_{\phi, num}','Analytic \omega_{\phi, theory}','Location','best');
title('Dispersion relation: numerical vs theory  \phi-mode');
grid on;

%% Sytem of ODES for higgs Klein Gordon
function dUdt = higgsKleinGordonODE(t, U, P)
    Nx = P.Nx;
    dx = P.dx;

    % unpack state vector
    phi     = U(1:Nx);
    dphi_dt = U(Nx+1:2*Nx);
    h       = U(2*Nx+1:3*Nx);
    dh_dt   = U(3*Nx+1:4*Nx);

    % allocate
    laplacian_phi = zeros(Nx,1);
    laplacian_h   = zeros(Nx,1);

    for i = 2:Nx-1
        laplacian_phi(i) = (phi(i+1) - 2*phi(i) + phi(i-1)) / dx^2;
        laplacian_h(i)   = (h(i+1)   - 2*h(i)   + h(i-1))   / dx^2;
    end

    laplacian_phi(1)  = (phi(2)     - 2*phi(1)   + phi(2))     / dx^2;
    laplacian_phi(Nx) = (phi(Nx-1)  - 2*phi(Nx)  + phi(Nx-1))  / dx^2;

    laplacian_h(1)    = (h(2)       - 2*h(1)     + h(2))       / dx^2;
    laplacian_h(Nx)   = (h(Nx-1)    - 2*h(Nx)    + h(Nx-1))    / dx^2;

    % Higgs potential derivative
    V_prime = P.lam * h .* (h.^2 - P.v^2);

    dphi_dt2 = laplacian_phi - P.m^2 * phi - 2 * P.g * h .* phi;
    dh_dt2   = laplacian_h   - V_prime    -     P.g * phi.^2;

    dUdt = [dphi_dt; dphi_dt2; dh_dt; dh_dt2];
end


%% Plot functions
function poltICs(x, phi_0, dphi_dt_0, h_0, dh_dt_0, P)
    figure()
    tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    ax1 = nexttile;
    plot(x, phi_0, 'b-');
    hold on
    plot(x, dphi_dt_0, 'b--');
    hold off
    xlabel('x'); 
    legend('\phi','d\phi/dt')
    title('\phi initial condition');
    grid on;
    
    ax2 = nexttile;
    plot(x, h_0- P.v, 'b-');
    hold on
    plot(x, dh_dt_0, 'b--');
    hold off
    xlabel('x'); 
    ylabel('h');
    legend('h - v','dh/dt')
    title('h initial condition');
    grid on;
end

