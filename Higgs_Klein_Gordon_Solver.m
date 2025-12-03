%% Coupled Higgs-Klein-Gordon Equations Solver using ODE45
clear all; close all; clc;

%% Constants:
% Physical
P.m = 1.0;  
P.g = 2.0;   
P.lam = 1.0; 
P.v = 1.0;  

%Numarical:
L = 40.0; % Size of domain
Nx = 300; % number of spatial points
stop_time = 30; 
dx = L / (Nx - 1);

%grid
x = linspace(-L/2, L/2, Nx);
P.dx = dx;
P.Nx = Nx;


%% Intial condtions:
phi_0 = 0.4 * exp(-x.^2 / 3.0);
dphi_dt_0 = zeros(size(x));
h_0 = P.v * ones(size(x)) + 0.4 * exp(-(x+3).^2 / 5.0);
dh_dt_0 = zeros(size(x));

figure()

%Print Intial condtions

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

%Comine into a state vector
U0 = [phi_0, dphi_dt_0, h_0, dh_dt_0]';

%% Solve with ODE 45
tspan = [0 stop_time];
options = odeset('RelTol', 1e-7, 'AbsTol', 1e-9, 'Stats', 'on');

fprintf('Start ODE solver\n')
tic;

[t, U] = ode45(@(t, U) higgsKleinGordonODE(t, U, P), tspan, U0, options);

fprintf('Finished in %.2f seconds\n', toc);

%% Plot results
phi_sol = U(:, 1:Nx)';
dphi_dt_sol = U(:, P.Nx+1:2*P.Nx)';
h_sol = U(:, 2*Nx+1:3*Nx)';
dh_dt_sol = U(:, 3*P.Nx+1:4*P.Nx)';

HeatMap(x, t, phi_sol, h_sol, P)
animation(x, t, phi_sol, h_sol)
energy_conservation(x, t, phi_sol, h_sol, dphi_dt_sol, dh_dt_sol, P)



%% System of ODES for Higgs-Klein Gordon
function dUdt = higgsKleinGordonODE(t, U, P)

    Nx = P.Nx;
    dx = P.dx;

    phi = U(1:Nx);
    dphi_dt = U(Nx+1:2*Nx);
    h = U(2*Nx+1:3*Nx);
    dh_dt = U(3*Nx+1:4*Nx);

    dphi_dt2 = zeros(Nx, 1);
    dh_dt2 = zeros(Nx, 1);

    laplacian_phi = zeros(Nx, 1);
    laplacian_h = zeros(Nx, 1);

    %Second derivative using central differences
    for i = 2:Nx-1
        laplacian_phi(i) = (phi(i+1) - 2*phi(i) + phi(i-1)) / dx^2;
        laplacian_h(i) = (h(i+1) - 2*h(i) + h(i-1)) / dx^2;
    end

    % Boundary conditions Neumann:
    laplacian_phi(1) = (phi(2) - 2*phi(1) + phi(2)) / dx^2;
    laplacian_h(1) = (h(2) - 2*h(1) + h(2)) / dx^2;

    laplacian_phi(Nx) = (phi(Nx-1) - 2*phi(Nx) + phi(Nx-1)) / dx^2;
    laplacian_h(Nx) = (h(Nx-1) - 2*h(Nx) + h(Nx-1)) / dx^2;

    V_prime = P.lam * h .* (h.^2 - P.v^2);
    
    dphi_dt2 = laplacian_phi - P.m^2 * phi - 2 * P.g * h .* phi;
    dh_dt2   = laplacian_h - V_prime - P.g * phi.^2;

    dUdt = [dphi_dt; dphi_dt2; dh_dt; dh_dt2];
end



%% Plot functions
function animation(x, t, phi_sol, h_sol)
    fig = figure('Color', '#181818');
    set(fig, 'InvertHardcopy', 'off');
    fig.Position = [100 100 1400 650];

    v = VideoWriter('animation.mp4', 'MPEG-4');
    v.FrameRate = 30;
    open(v);

    dt = 10;

    for i = 1:dt:length(t)
        clf;

        ax1 = axes('Position', [0.07 0.15 0.40 0.70]);
        set(ax1, 'Color', '#181818', 'XColor', [0.9 0.9 0.9], 'YColor', [0.9 0.9 0.9]);
        hold(ax1, 'on');
        plot(x, phi_sol(:,i), 'Color', [0.3 0.7 1], 'LineWidth', 1.5);
        ylim([min(phi_sol(:)), max(phi_sol(:))]);

        xlabel('x', 'Color', [0.9 0.9 0.9]);
        ylabel('\phi', 'Color', [0.9 0.9 0.9]);
        title(sprintf('\\phi field, t = %.2f', t(i)), 'Color', [0.9 0.9 0.9]);
        grid on;

        ax2 = axes('Position', [0.55 0.15 0.40 0.70]);
        set(ax2, 'Color', '#181818', 'XColor', [0.9 0.9 0.9], 'YColor', [0.9 0.9 0.9]);
        hold(ax2, 'on');

        plot(x, h_sol(:,i), 'Color', [1 0.4 0.4], 'LineWidth', 1.5);
        ylim([min(h_sol(:)), max(h_sol(:))]);

        xlabel('x', 'Color', [0.9 0.9 0.9]);
        ylabel('h', 'Color', [0.9 0.9 0.9]);
        title(sprintf('h field, t = %.2f', t(i)), 'Color', [0.9 0.9 0.9]);
        grid on;

        frame = getframe(fig);
        writeVideo(v, frame);
    end

    close(v);
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
    cb.Layout.Tile = 'east';
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
    
        kinetic_phi = 0.5 * sum(dphi_dt.^2) * dx;
        kinetic_h = 0.5 * sum(dh_dt.^2) * dx;
    
        gradient_phi = 0.5 * sum(dphi_dx.^2) * dx;
        gradient_h = 0.5 * sum(dh_dx.^2) * dx;
    
        potential_phi = 0.5 * P.m^2 * sum(phi.^2) * dx;
        potential_h = 0.25 * P.lam * sum((h.^2 - P.v^2).^2) * dx;

        interaction = P.g * sum(h .* phi.^2) * dx;

        total_energy(i) = kinetic_phi + kinetic_h + gradient_phi + gradient_h + potential_phi + potential_h + interaction;
    end

    initial_energy = total_energy(1);
    normalized_energy = total_energy / initial_energy;
    
    figure()
    plot(t, normalized_energy, 'k-');
    
    ylim([0.95 1.05])
    ylabel('Normilized Energy');
    title('Energy Conservation');
    grid on;
    
    fprintf('Final energy error: %.6f', normalized_energy(end));

end
