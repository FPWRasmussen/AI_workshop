"""
Workshop utilities for FLoW-Net.
"""
import sys
import numpy as np
import torch
from pathlib import Path
from shapely.geometry import Point, Polygon
from shapely.affinity import scale
from models.aru_net import ARUNet
from py_wake.site._site import UniformSite
from py_wake.flow_map import HorizontalGrid
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.deficit_models.gaussian import ZongGaussianDeficit
from py_wake.superposition_models import SquaredSum
from py_wake.turbulence_models import STF2017TurbulenceModel
from src.SimpleDTU10MW import SimpleDTU10MW


def load_model(path, device='cpu'):
    """Load a saved model."""
    model = torch.load(path, map_location=device, weights_only=False)
    
    # Remove all forward and backward hooks (e.g., from wandb)
    for name, module in model.named_modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()
    
    model.to(device)
    model.eval()
    return model


# =============================================================================
# Normalization
# =============================================================================

def normalize_labels(ws, ti, ws_range=(4.0, 25.0), ti_range=(0.03, 0.25)):
    """Min-max normalize wind speed and turbulence intensity to [0,1]."""
    ws_norm = (ws - ws_range[0]) / (ws_range[1] - ws_range[0])
    ti_norm = (ti - ti_range[0]) / (ti_range[1] - ti_range[0])
    return np.array([ws_norm, ti_norm], dtype=np.float32)


def log_normalize_output(flow_map, ambient_ws, epsilon=1e-5):
    """Log-normalize flow map to [0,1]."""
    relative = np.clip(1 - flow_map / ambient_ws, epsilon, 1.0)
    log_min, log_max = np.log(epsilon), 0.0
    normalized = (np.log(relative) - log_min) / (log_max - log_min)
    return np.clip(normalized, 0, 1).astype(np.float32)


def inverse_log_normalize(normalized, ambient_ws, epsilon=1e-5):
    """Convert normalized output back to wind speeds."""
    log_min, log_max = np.log(epsilon), 0.0
    log_vals = normalized * (log_max - log_min) + log_min
    relative = np.exp(log_vals)
    return np.clip(ambient_ws * (1 - relative), 0, ambient_ws)


# =============================================================================
# Rasterization
# =============================================================================

def rasterize_turbines(positions, x_coords, y_coords):
    """Convert turbine positions to grid using bilinear interpolation."""
    nx, ny = len(x_coords), len(y_coords)
    grid = np.zeros((ny, nx), dtype=np.float32)
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    
    for x, y in positions:
        x_idx = np.argmin(np.abs(x_coords - x))
        y_idx = np.argmin(np.abs(y_coords - y))
        
        x_frac = (x - x_coords[x_idx]) / dx
        y_frac = (y - y_coords[y_idx]) / dy
        
        for di, wi in [(0, 1-abs(x_frac)), (int(np.sign(x_frac)), abs(x_frac))]:
            for dj, wj in [(0, 1-abs(y_frac)), (int(np.sign(y_frac)), abs(y_frac))]:
                xi, yj = x_idx + di, y_idx + dj
                if 0 <= xi < nx and 0 <= yj < ny:
                    grid[yj, xi] += wi * wj
    
    return grid


# =============================================================================
# Polygon-based Layout Generation (simplified from original)
# =============================================================================

def generate_polygon(n_vertices=5, farm_area=1000):
    """Generate a random polygon with given area."""
    # Random angles around circle
    angles = np.sort(np.random.uniform(0, 2*np.pi, n_vertices))
    # Random radii
    radii = 1 + np.abs(np.random.normal(0, 0.3, n_vertices))
    
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    polygon = Polygon(points)
    
    # Scale to desired area
    scale_factor = np.sqrt(farm_area / polygon.area)
    polygon = scale(polygon, xfact=scale_factor, yfact=scale_factor, origin='centroid')
    
    return polygon


def poisson_disk_sample(polygon, spacing):
    """Generate points inside polygon with minimum spacing (Poisson disk sampling)."""
    min_x, min_y, max_x, max_y = polygon.bounds
    cell_size = spacing / np.sqrt(2)
    
    grid_w = int(np.ceil((max_x - min_x) / cell_size))
    grid_h = int(np.ceil((max_y - min_y) / cell_size))
    grid = [[None] * grid_w for _ in range(grid_h)]
    
    def grid_coords(p):
        return int((p[0] - min_x) / cell_size), int((p[1] - min_y) / cell_size)
    
    def is_valid(p):
        if not polygon.contains(Point(p)):
            return False
        gx, gy = grid_coords(p)
        for i in range(max(0, gx-2), min(grid_w, gx+3)):
            for j in range(max(0, gy-2), min(grid_h, gy+3)):
                if grid[j][i] is not None:
                    if np.linalg.norm(np.array(p) - np.array(grid[j][i])) < spacing:
                        return False
        return True
    
    # Find initial point inside polygon
    for _ in range(1000):
        p = [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]
        if polygon.contains(Point(p)):
            break
    
    gx, gy = grid_coords(p)
    grid[gy][gx] = p
    active = [p]
    points = [p]
    
    while active:
        idx = np.random.randint(len(active))
        base = active[idx]
        found = False
        
        for _ in range(30):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(spacing, 2*spacing)
            new_p = [base[0] + dist*np.cos(angle), base[1] + dist*np.sin(angle)]
            
            if is_valid(new_p):
                gx, gy = grid_coords(new_p)
                if 0 <= gy < grid_h and 0 <= gx < grid_w:
                    grid[gy][gx] = new_p
                    active.append(new_p)
                    points.append(new_p)
                    found = True
                    break
        
        if not found:
            active.pop(idx)
    
    return np.array(points)


def generate_layout(farm_area=1000, spacing=6.0):
    """Generate wind farm layout using polygon + Poisson disk sampling."""
    polygon = generate_polygon(n_vertices=np.random.randint(4, 7), farm_area=farm_area)
    positions = poisson_disk_sample(polygon, spacing)
    
    # Shift so max x is at 0 (turbines upstream of observation area)
    positions[:, 0] -= positions[:, 0].max()
    
    return positions, polygon


# =============================================================================
# Data Generation
# =============================================================================

def generate_sample(
    wind_speed=10.0,
    turbulence_intensity=0.06,
    farm_area=1000,
    spacing=6.0,
    x_range=(-128, 384),
    y_range=(-128, 128),
    resolution=(256, 128)
):
    """
    Generate a wind farm sample using PyWake.
    
    Args:
        wind_speed: Ambient wind speed (m/s)
        turbulence_intensity: Turbulence intensity
        farm_area: Farm area in D² (rotor diameters squared)
        spacing: Minimum turbine spacing in D
        x_range, y_range: Domain extent in D
        resolution: Grid resolution (nx, ny)
    """
    
    # Generate layout
    positions, polygon = generate_layout(farm_area=farm_area, spacing=spacing)
    
    if len(positions) < 3:
        raise ValueError(f"Only generated {len(positions)} turbines, need at least 3")
    
    # PyWake simulation
    site = UniformSite()
    wt = SimpleDTU10MW()
    
    flow_model = PropagateDownwind(
        site=site,
        windTurbines=wt,
        wake_deficitModel=ZongGaussianDeficit(),
        superpositionModel=SquaredSum(),
        turbulenceModel=STF2017TurbulenceModel()
    )
    
    x_coords = np.linspace(x_range[0], x_range[1], resolution[0])
    y_coords = np.linspace(y_range[0], y_range[1], resolution[1])
    
    sim = flow_model(
        x=positions[:, 0],
        y=positions[:, 1],
        wd=[270],
        ws=[wind_speed],
        TI=[turbulence_intensity]
    )
    
    flow_map = sim.flow_map(HorizontalGrid(x=x_coords, y=y_coords, h=1.0))
    ws_eff = np.array(flow_map.WS_eff.squeeze())
    
    grid = rasterize_turbines(positions, x_coords, y_coords)
    
    return {
        'grid': grid,
        'flow_map': ws_eff,
        'labels': np.array([wind_speed, turbulence_intensity], dtype=np.float32),
        'x_coords': x_coords,
        'y_coords': y_coords,
        'turbine_positions': positions,
        'polygon': polygon
    }


# =============================================================================
# Training utilities
# =============================================================================

def prepare_sample(sample, normalize_output=False):
    """Convert a sample dict to tensors."""
    grid = torch.from_numpy(sample['grid']).unsqueeze(0).float()
    flow = sample['flow_map']
    ws, ti = sample['labels']
    
    if normalize_output:
        flow = log_normalize_output(flow, ws)
    
    output = torch.from_numpy(flow).unsqueeze(0).float()
    labels = torch.from_numpy(normalize_labels(ws, ti)).float()
    
    return grid, output, labels


def prepare_batch(samples, normalize_output=False):
    """Prepare a batch of samples."""
    grids, outputs, labels = [], [], []
    for s in samples:
        g, o, l = prepare_sample(s, normalize_output)
        grids.append(g)
        outputs.append(o)
        labels.append(l)
    return torch.stack(grids), torch.stack(outputs), torch.stack(labels)


def predict(model, sample, device='cpu'):
    """Run inference and denormalize output."""
    model.eval()
    grid, _, labels = prepare_sample(sample, normalize_output=False)
    
    with torch.no_grad():
        grid = grid.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        pred_norm = model(grid, labels).squeeze().cpu().numpy()
    
    return inverse_log_normalize(pred_norm, sample['labels'][0])


def compute_metrics(true, pred):
    """Calculate RMSE, MAE, R²."""
    t, p = true.flatten(), pred.flatten()
    rmse = np.sqrt(np.mean((p - t)**2))
    mae = np.mean(np.abs(p - t))
    r2 = 1 - np.sum((p - t)**2) / np.sum((t - np.mean(t))**2)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}
