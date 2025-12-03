%% Coupled Higgs-Klein-Gordon Equations Solver using ODE45
clear all; close all; clc;

%% Constants:
% Physical
P.m = 1.0;  
P.g = 0.4;   
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
stop_time = 32; 
tspan = [0 stop_time];

h_star = - P.m^2 / (2 * P.g);

Vprime_h = P.lam * h_star * (h_star^2 - P.v^2);

phi_star_sq = - Vprime_h / P.g;

if phi_star_sq <= 0
    error('No real stationary point exists for these parameters.')
end

phi_star = sqrt(phi_star_sq);


fprintf("\nStationary point:\n");
fprintf("h*   = %.6f\n", h_star);
fprintf("phi* = %.6f\n", phi_star);


%% Initial Conditions:
eps = 1e-6;

phi_0 = phi_star + eps * ones(size(x));
dphi_dt_0 = zeros(size(x));

h_0 = h_star + eps * ones(size(x));
dh_dt_0 = zeros(size(x));

U0 = [phi_0, dphi_dt_0, h_0, dh_dt_0]';


%% Integrate
options = odeset('RelTol',1e-7,'AbsTol',1e-9);

fprintf('Running ODE45...\n');
[t, U] = ode45(@(t,U) higgsKleinGordonODE(t,U,P), tspan, U0, options);
fprintf('ODE45 complete.\n');


%% Extract solutions
phi_sol     = U(:, 1:Nx)';
dphi_dt_sol = U(:, Nx+1:2*Nx)';
h_sol       = U(:, 2*Nx+1:3*Nx)';
dh_dt_sol   = U(:, 3*Nx+1:4*Nx)';


%% Compute A(t)

A = zeros(length(t),1);
for ti = 1:length(t)
    A(ti) = trapz(x, h_sol(:,ti) - h_star)/L;
end

%% Fit exponential growth rate with specified time window
t_min = 10.0;
t_max = 25.0;
sigma_num = fit_growth_rate_window(t, abs(A), t_min, t_max, 1);
fprintf("Numerical growth rate sigma = %.6f\n", sigma_num);

%% Theoretical growth rate 
sigma_th = theory_growth_rate(P);
fprintf("Theoretical growth rate sigma_th = %.6f\n", sigma_th);


idx_start = find(t >= t_min, 1);
A0_aligned = abs(A(idx_start)) / exp(sigma_th * t(idx_start));
A_th = A0_aligned * exp(sigma_th * t);

fprintf("Aligned theory amplitude at t=%.1f: %.6e\n", t_min, A0_aligned);

%% Plots
figure;

% Log scale plot with aligned theory
semilogy(t, abs(A), 'b', 'LineWidth', 2); hold on;
semilogy(t, abs(A_th), 'r--', 'LineWidth', 2);

% Mark the alignment point
semilogy(t(idx_start), abs(A(idx_start)), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');

xlabel('t'); ylabel('|A(t)|');
title(sprintf('Aligned Theory: σ_{num}=%.4f, σ_{th}=%.4f', sigma_num, sigma_th));
legend('Numerical', 'Theory (aligned)', 'Alignment point', 'Location', 'best');
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


%% Growth rate fitting with specified time window
function sigma = fit_growth_rate_window(t, A, t_min, t_max, doPlot)

    t = t(:);
    A = A(:);
    
    idx = (t >= t_min) & (t <= t_max);
    

    t_fit = t(idx);
    A_fit = A(idx);
    
    pos_idx = A_fit > 0;
    
    t_fit = t_fit(pos_idx);
    A_fit = A_fit(pos_idx);
    
    logA = log(A_fit);
    p = polyfit(t_fit, logA, 1);
    sigma = p(1);
    
    y_fit = polyval(p, t_fit);
    SS_res = sum((logA - y_fit).^2);
    SS_tot = sum((logA - mean(logA)).^2);
    R2 = 1 - SS_res/SS_tot;
    
    fprintf('Fit window: t = [%.3f, %.3f], %d points\n', t_min, t_max, length(t_fit));
    fprintf('Growth rate sigma = %.6f, R² = %.4f\n', sigma, R2);
    
    if doPlot
        figure;
        
        % Log plot
        subplot(1,2,1);
        semilogy(t, abs(A), 'b-', 'LineWidth', 1.5); hold on;
        semilogy(t_fit, A_fit, 'ro', 'MarkerSize', 6, 'LineWidth', 2);
        
        % Plot the fitted exponential
        A_fitted = exp(polyval(p, t_fit));
        semilogy(t_fit, A_fitted, 'r-', 'LineWidth', 2);
        
        xline(t_min, 'k--', 'LineWidth', 1, 'Label', 't_{min}');
        xline(t_max, 'k--', 'LineWidth', 1, 'Label', 't_{max}');
        
        xlabel('Time t');
        ylabel('|A(t)|');
        title(sprintf('Growth Rate Fit (Log Scale)\nσ = %.4f, R² = %.3f', sigma, R2));
        legend('Data', 'Fitting Points', 'Linear Fit', 'Location', 'best');
        grid on;
        
        % Linear plot of log(A) vs t
        subplot(1,2,2);
        plot(t_fit, logA, 'bo', 'MarkerSize', 6, 'LineWidth', 1.5); hold on;
        plot(t_fit, y_fit, 'r-', 'LineWidth', 2);
        xlabel('Time t');
        ylabel('log|A(t)|');
        title('Linear Fit in Log Space');
        legend('Data', sprintf('Fit: slope = %.4f', sigma), 'Location', 'best');
        grid on;
    end
end


function sigma_th = theory_growth_rate(P)

    h_star = -P.m^2 / (2 * P.g);

    Vprime_h = P.lam * h_star * (h_star^2 - P.v^2);

    phi_star_sq = -Vprime_h / P.g;

    if phi_star_sq <= 0
        sigma_th = 0;
        return;
    end

    phi_star = sqrt(phi_star_sq);

    Vpp = P.lam * (3*h_star^2 - P.v^2);

    A = Vpp + (P.m^2 + 2*P.g*h_star);
    B = Vpp - (P.m^2 + 2*P.g*h_star);
    C = 16 * P.g^2 * phi_star_sq;

    lambda_plus  = 0.5 * (A + sqrt(B^2 + C));
    lambda_minus = 0.5 * (A - sqrt(B^2 + C));

    if lambda_minus >= 0
        sigma_th = 0;
    else
        sigma_th = sqrt(-lambda_minus);
    end

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

    % --- Shared color scale ---
    clim_min = min([phi_sol(:); h_sol(:) - P.v]);
    clim_max = max([phi_sol(:); h_sol(:)- P.v]);
    set([ax1, ax2], 'CLim', [clim_min, clim_max]);

    % --- One colorbar on the far right ---
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


