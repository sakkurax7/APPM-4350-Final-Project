%% Coupled Higgs-Klein-Gordon Equations Solver using ODE45
clear all; close all; clc;

%% Constants:
% Physical
P.m = 1.0;  
P.g = 0.1;   
P.lam = 1.0; 
P.v = 1.0;  

%Numarical:
L = 40.0; % Size of domain
Nx = 512; % number of spatial points
dx = L / (Nx - 1);
P.dx = dx;
P.Nx = Nx;
x = linspace(-L/2, L/2, Nx);

%time
stop_time = 200; 
tspan = [0 stop_time];


%% Initial Conditions:
eps = 1e-4;    % small amplitude (linear regime)

% Equilibrium 
h_star = P.v;
phi_star = 0;

%% Sweep over modes and compute omega
options = odeset('RelTol', 1e-7, 'AbsTol', 1e-9);

% Modes to sweep
n_list = [1, 2, 4, 8];    
Nk = numel(n_list);

k_vals = zeros(Nk,1);
omega_num = zeros(Nk,1);
omega_theory = zeros(Nk,1);

fprintf('Starting sweep over n values: %s\n', mat2str(n_list));

for jj = 1:Nk
    n = n_list(jj);
    k0 = 2*pi*n / L;
    k_vals(jj) = k0;
    mode = cos(k0 * x);

    % Build initial conditions for phi-mode
    phi_0 = phi_star * ones(size(x));
    dphi_dt_0 = zeros(size(x));
    h_0 = h_star + eps * mode;
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

    Ah = zeros(length(t),1);
    cos_kx = cos(k0 * x).';
    
    for ti = 1:length(t)
        h_t = h_sol(:, ti);
        a_h = (2/L) * trapz(x, h_t .* cos_kx);
        Ah(ti) = a_h;
    end
    
    omega_est = estimate_frequency(t, Ah);
    omega_num(jj) = omega_est;

    % Analytical effective mass for phi-mode at equilibrium h_star
    m_h2 = 2 * P.lam * P.v^2;
    omega_theory(jj) = sqrt(k0^2 + m_h2);

    % (Optional) quick sanity print
    fprintf('    ω_num = %.6f, ω_\phi,theory = %.6f, rel error = %.2e\n', ...
        omega_est, omega_theory(jj), abs(omega_est-omega_theory(jj))/omega_theory(jj));
end

%% Plot numeric vs theory

figure('Name','Dispersion: ω(k)','Units','normalized','Position',[0.2 0.2 0.5 0.5]);
plot(k_vals, omega_num, 'bo-', 'LineWidth', 1.5, 'MarkerSize',8); hold on;
plot(k_vals, omega_theory, 'r--s', 'LineWidth', 1.5, 'MarkerSize',8);
xlabel('k');
ylabel('\omega(k)');
legend('Numerical \omega_{h, num}','Analytic \omega_{h, theory}','Location','best');
title('Dispersion relation: numerical vs theory h-mode');
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

    % equations of motion 
    dphi_dt2 = laplacian_phi - P.m^2 * phi - 2 * P.g * h .* phi;
    dh_dt2   = laplacian_h   - V_prime    -     P.g * phi.^2;

    dUdt = [dphi_dt; dphi_dt2; dh_dt; dh_dt2];
end


%% Plot functions
function poltICs(x, phi_0, dphi_dt_0, h_0, dh_dt_0,  P)
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


function animation(x, t, phi_sol, h_sol)
    figure();
   
    dt = 10;
    for i = 1:dt:length(t)
        subplot(1, 2, 1);
        plot(x, phi_sol(:, i), 'b-');
        ylim([min(phi_sol(:)), max(phi_sol(:))]);
        xlabel('x'); 
        ylabel('\phi');
        title(sprintf('\\phi field, t = %.2f', t(i)));
        grid on;
        
        subplot(1, 2, 2);
        plot(x, h_sol(:, i), 'r-');
        ylim([min(h_sol(:)), max(h_sol(:))]);
        xlabel('x');
        ylabel('h');
        title(sprintf('h field, t = %.2f', t(i)));
        grid on;
        
        drawnow;
    end
end

function HeatMap(x, t, phi_sol, h_sol, P)

    figure;

    tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile;
    imagesc(t, x, phi_sol);  
    set(ax1, 'YDir', 'normal');   
    axis(ax1, 'square');  
    xlabel(ax1, 't');
    ylabel(ax1, 'x');
    title(ax1, '\phi(x,t)');

    ax2 = nexttile;
    imagesc(t, x, h_sol - P.v);
    set(ax2, 'YDir', 'normal');
    axis(ax2, 'square');
    xlabel(ax2, 't');
    ylabel(ax2, 'x');
    title(ax2, 'h(x,t) - v');

    clim_min = min([phi_sol(:); h_sol(:) - P.v]);
    clim_max = max([phi_sol(:); h_sol(:)- P.v]);
    set([ax1, ax2], 'CLim', [clim_min, clim_max]);

    cb = colorbar(ax2);
    cb.Layout.Tile = 'east';    % put it on the far right of layout


end



function energy_conservation(x, t, phi_sol, h_sol, dphi_dt_sol, dh_dt_sol, P)
        total_energy = zeros(length(t), 1);
        dx = P.dx;
        for i = 1:length(t)
            phi = phi_sol(:, i);
            dphi_dt = dphi_dt_sol(:, i);
            h = h_sol(:, i);
            dh_dt = dh_dt_sol(:, i);

            dphi_dx = gradient(phi, dx);
            dh_dx = gradient(h, dx);
        
            % Kinetic energ
            kinetic_phi = 0.5 * sum(dphi_dt.^2) * dx;
            kinetic_h = 0.5 * sum(dh_dt.^2) * dx;
        
            % Gradient energy 
            gradient_phi = 0.5 * sum(dphi_dx.^2) * dx;
            gradient_h = 0.5 * sum(dh_dx.^2) * dx;
        
            % Potential energy
            potential_phi = 0.5 * P.m^2 * sum(phi.^2) * dx;
            potential_h = 0.25 * P.lam * sum((h.^2 - P.v^2).^2) * dx;
            interaction = 0.5 * P.g * sum(h .* phi.^2) * dx;

            total_energy(i) = kinetic_phi + kinetic_h + gradient_phi + gradient_h + potential_phi + potential_h + interaction;
        end

        initial_energy = total_energy(1);
        normalized_energy = total_energy/initial_energy;
        
        figure()
        
        plot(t, normalized_energy, 'k-');
        
        ylim([0.95 1.05])
        ylabel('Normilized Energy');
        title('Energy Conservation');
        grid on;
        
        
        fprintf('Final energy error: %.6f', normalized_energy(end));

end

function omega = estimate_frequency(t, A)
    A0 = A - mean(A);
    w = max(3, round(length(A0)/200));
    As = movmean(A0, w);

    peakProm = max(1e-8, 0.05 * max(abs(As)));
    [pks, locs] = findpeaks(As, t, 'MinPeakProminence', peakProm);

    if numel(locs) >= 2
        Tmean = mean(diff(locs));  % average period
        omega = 2*pi / Tmean;
        return;
    end

    Nt = length(t);
    dt = mean(diff(t));
    Y = fft(A0);
    Pspec = abs(Y(1:floor(Nt/2)));
    freqs = (0:floor(Nt/2)-1)'/(Nt*dt);
    [~, idx] = max(Pspec(2:end)); % skip DC
    freq = freqs(idx+1);
    omega = 2*pi*freq;
end




