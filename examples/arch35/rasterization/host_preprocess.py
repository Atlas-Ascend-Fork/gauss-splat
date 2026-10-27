import torch
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]
device = "npu"
def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm
#from gaussians splatting code
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='npu')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
#from gaussians splatting code
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def strip_symmetric(sym):
    return strip_lowerdiag(sym)
    #from gaussians splatting code
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def apply_transform_matrix(points, transform_matrix):#VIBE Coding
    """
    对点云坐标进行投影变换

    参数:
    points (torch.Tensor): 形状为 [N, 3] 的点云坐标
    transform_matrix (torch.Tensor): 形状为 [4, 4] 的变换矩阵

    返回:
    torch.Tensor: 形状为 [N, 3] 的变换后的点云坐标
    """
    # 将点云坐标转换为齐次坐标 [N, 4]
    points_homogeneous = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], dim=1)
    # 进行变换
    transformed_points = torch.matmul(points_homogeneous, transform_matrix)

    # 将齐次坐标转换回普通坐标 [N, 3]
    transformed_points = transformed_points[:, :3] / (transformed_points[:, 3].unsqueeze(1) + 1e-7)
    return transformed_points
def transformPoint4x3(mean, viewmatrix):
    # mean: (N, 3)
    # viewmatrix: (4, 4)
    # Add a column of ones to mean to make it (N, 4)
    mean_4 = torch.cat([mean, torch.ones((mean.shape[0], 1), device=mean.device)], dim=1)
    # Transform the points
    t = torch.matmul(mean_4, viewmatrix)[:, :3]
    return t

def Cov3D2Cov2D(positions, focal_x, focal_y, tanfovx, tanfovy, cov3D, viewmatrix):#VIBE CODING
    N = positions.shape[0]

    # Step 1: Transform points
    t = transformPoint4x3(positions, viewmatrix)#torch.matmul(positions, viewmatrix.T)
    #t = apply_transform_matrix(positions, viewmatrix.T)

    # Step 2: Compute limits and apply clamping
    limx = 1.3 * tanfovx
    limy = 1.3 * tanfovy
    txtz = t[:, 0] / t[:, 2]
    tytz = t[:, 1] / t[:, 2]
    t0_limit = torch.clamp(txtz, -limx, limx) * t[:, 2]
    t1_limit = torch.clamp(tytz, -limy, limy) * t[:, 2]
    #t[:, 0] = torch.clamp(txtz, -limx, limx) * t[:, 2]
    #t[:, 1] = torch.clamp(tytz, -limy, limy) * t[:, 2]

    # Step 3: Construct Jacobian matrix J
    J = torch.zeros((N, 3, 3), device=positions.device)
    J[:, 0, 0] = focal_x / t[:, 2]
    J[:, 0, 2] = -(focal_x * t0_limit) / (t[:, 2] * t[:, 2])
    J[:, 1, 1] = focal_y / t[:, 2]
    J[:, 1, 2] = -(focal_y * t1_limit) / (t[:, 2] * t[:, 2])

    # Step 4: Construct view matrix W
    W = viewmatrix[:3, :3]  # (3, 3)

    # Step 5: Compute transformation matrix T
    T = torch.matmul(W, J.transpose(1,2))  # (N, 3, 3)

    # Step 6: Construct 3D covariance matrix Vrk
    Vrk = torch.zeros((N, 3, 3), device=positions.device)
    Vrk[:, 0, 0] = cov3D[:, 0]
    Vrk[:, 0, 1] = cov3D[:, 1]
    Vrk[:, 0, 2] = cov3D[:, 2]
    Vrk[:, 1, 0] = cov3D[:, 1]
    Vrk[:, 1, 1] = cov3D[:, 3]
    Vrk[:, 1, 2] = cov3D[:, 4]
    Vrk[:, 2, 0] = cov3D[:, 2]
    Vrk[:, 2, 1] = cov3D[:, 4]
    Vrk[:, 2, 2] = cov3D[:, 5]

    # Step 7: Compute 2D covariance matrix cov
    cov = torch.matmul(torch.matmul(T.transpose(1, 2), Vrk), T)  # (N, 3, 3)
    #cov = torch.matmul(T.transpose(1,2), torch.matmul(Vrk, T))

    # Step 8: Return the result
    result = torch.zeros((N, 3), device=positions.device)
    result[:, 0] = cov[:, 0, 0]
    result[:, 1] = cov[:, 0, 1]
    result[:, 2] = cov[:, 1, 1]
    return result
def eigen_decompose_cov2D(A):
    N = A.shape[0]
    matrix = torch.zeros((N, 2, 2), device=A.device)
    matrix[:, 0, 0] = A[:, 0]
    matrix[:, 0, 1] = A[:, 1]
    matrix[:, 1, 0] = A[:, 1]
    matrix[:, 1, 1] = A[:, 2]
    return eigen_decompose_2x2(matrix)
def eigen_decompose_2x2(A):
    """
    只返回scales，最快版本
    """
    a = A[..., 0, 0]
    b = A[..., 0, 1]
    d = A[..., 1, 1]

    trace = a + d
    diff = a - d
    discriminant = diff * diff + 4 * b * b
    sqrt_disc = torch.sqrt(discriminant)

    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2

    lambda1 = torch.clamp(lambda1, min=1e-8)
    lambda2 = torch.clamp(lambda2, min=1e-8)

    scales = torch.sqrt(torch.stack([lambda1, lambda2], dim=-1))
    return scales  # (batch, 2)
def eigen_decompose_2x2_v0(A):
    """
    使用PyTorch内置函数，会同时返回特征值和特征向量
    """
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    # 对于实对称矩阵，特征值是实数，可以直接取real
    eigenvalues_real = eigenvalues.real
    return eigenvalues_real
def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result
#pos    [GS, 3]
#camera [3]
#shs    [GS, SHS, 3]
def compute_rgb_from_shs(positions, camera_pos, shs, sh_degree):
    GS = shs.shape[0]
    shs_view = shs.transpose(1, 2).view(GS, 3, -1)
    dir_pp = (positions - camera_pos.unsqueeze(0).repeat(GS, 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    return colors_precomp
def computeConicOpacity(cov2d, opacities):
    N = cov2d.shape[0]

    # Step 1: Compute the determinant
    # det = cov2d[:, 0] * cov2d[:, 2] - cov2d[:, 1] * cov2d[:, 1]
    h_var = 0.3
    cov2d[..., 0] += h_var
    cov2d[..., 2] += h_var
    det_cov_plus_h_cov = cov2d[:, 0] * cov2d[:, 2] - cov2d[:, 1] * cov2d[:, 1]
    det = det_cov_plus_h_cov
    # Step 2: Check if determinant is zero
    valid = det != 0.0
    # Step 3: Compute the inverse of the determinant
    det_inv = 1.0 / det[valid]
    # Step 4: Compute the conic parameters
    conic = torch.zeros((N, 3), device=cov2d.device)
    conic[valid, 0] = cov2d[valid, 2] * det_inv
    conic[valid, 1] = -cov2d[valid, 1] * det_inv
    conic[valid, 2] = cov2d[valid, 0] * det_inv
    # Step 5: Construct the final conic_opacity tensor
    conic_opacity = torch.zeros((N, 4), device=cov2d.device)
    conic_opacity[:, :3] = conic
    conic_opacity[:, 3] = opacities.reshape(-1)
    assert cov2d.shape[0] == conic_opacity.shape[0]
    return conic_opacity
def preprocess_gaussians(gaussians, FoVx, FoVy, world_view_transform, full_proj_transform, camera_center, H, W, sh_degree = 3):
    #FoVx, FoVy, world_view_transform, full_proj_transform, camera_center = get_camera_info(intrinsic, extrinsic, W, H)
    tanfovx = torch.tan(FoVx * 0.5)
    tanfovy = torch.tan(FoVy * 0.5)
    focal_x = W / (2.0 * (tanfovx))
    focal_y = H / (2.0 * (tanfovy))

    positions = gaussians.positions #[GS, 3]
    scales = gaussians.scales       #[GS, 3]
    rotations = gaussians.rotations #[GS, 4]
    shs = gaussians.shs             #[GS, SHS, 3]
    opacities = gaussians.opacities
    cov3D = build_covariance_from_scaling_rotation(scales, 1, rotations) #[GS, 6]

    rgb = compute_rgb_from_shs(positions, camera_center, shs, sh_degree)
    screen_positions = apply_transform_matrix(positions, full_proj_transform)[..., :2]#in NDC format
    screen_positions[..., 0] = ((screen_positions[..., 0] + 1) * W - 1) * 0.5
    screen_positions[..., 1] = ((screen_positions[..., 1] + 1) * H - 1) * 0.5

    # project2 = point_project_to_view2(intrinsic, extrinsic, positions, H, W)
    # screen_positions[..., 0] = project2[..., 1]
    # screen_positions[..., 1] = project2[..., 0]

    cov2D = Cov3D2Cov2D(positions, focal_x, focal_y, tanfovx, tanfovy, cov3D, world_view_transform)
    conic_opacities = computeConicOpacity(cov2D, opacities)
    return screen_positions, cov2D, rgb, conic_opacities
import math
def get_cov2d_scales(positions, scales, rotations, FoVx, FoVy, world_view_transform, H, W):
    #FoVx, FoVy, world_view_transform, full_proj_transform, camera_center = get_camera_info(intrinsic, extrinsic, W, H)
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)
    focal_x = W / (2.0 * (tanfovx))
    focal_y = H / (2.0 * (tanfovy))

    cov3D = build_covariance_from_scaling_rotation(scales, 1, rotations) #[GS, 6]
    cov2D = Cov3D2Cov2D(positions, focal_x, focal_y, tanfovx, tanfovy, cov3D, world_view_transform)
    return eigen_decompose_cov2D(cov2D)
    #return screen_positions, cov2D, rgb, conic_opacities