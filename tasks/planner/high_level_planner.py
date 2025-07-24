#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tasks/planner/high_level_planner.py

Utilities for orientation handling:
  - quat_to_euler / euler_to_quat
  - quat_to_rotmat / rotmat_to_quat
  - average_quaternion
  - slerp
  - orientation_error (axis, angle)
"""

import numpy as np

# 1) Quaternion → Euler (ZYX / Tait-Bryan)
def quat_to_euler(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    # roll (X-axis)
    sinr = 2*(w*x + y*z)
    cosr = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr, cosr)
    # pitch (Y-axis)
    sinp = 2*(w*y - z*x)
    pitch = np.sign(sinp)*np.pi/2 if abs(sinp)>=1 else np.arcsin(sinp)
    # yaw (Z-axis)
    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny, cosy)
    return np.array([roll, pitch, yaw], dtype=np.float32)

# 2) Euler → Quaternion
def euler_to_quat(e: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = e
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    q = np.array([w, x, y, z], dtype=np.float32)
    return q/np.linalg.norm(q)

# 3) Quaternion → Rotation matrix 3×3
def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q/np.linalg.norm(q)
    return np.array([
      [1-2*(y*y+z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
      [2*(x*y + w*z),   1-2*(x*x+z*z),   2*(y*z - w*x)],
      [2*(x*z - w*y),     2*(y*z + w*x), 1-2*(x*x+y*y)]
    ], dtype=np.float32)

# 4) Rotation matrix → Quaternion
def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]; m10, m11, m12 = R[1]; m20, m21, m22 = R[2]
    trace = m00+m11+m22
    if trace > 0:
        s = 0.5/np.sqrt(trace+1.0)
        w = 0.25/s
        x = (m21 - m12)*s
        y = (m02 - m20)*s
        z = (m10 - m01)*s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0*np.sqrt(1.0+m00-m11-m22)
            w = (m21 - m12)/s
            x = 0.25*s
            y = (m01 + m10)/s
            z = (m02 + m20)/s
        elif m11 > m22:
            s = 2.0*np.sqrt(1.0+m11-m00-m22)
            w = (m02 - m20)/s
            x = (m01 + m10)/s
            y = 0.25*s
            z = (m12 + m21)/s
        else:
            s = 2.0*np.sqrt(1.0+m22-m00-m11)
            w = (m10 - m01)/s
            x = (m02 + m20)/s
            y = (m12 + m21)/s
            z = 0.25*s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q/np.linalg.norm(q)

# 5) Average of quaternions q0, q1
def average_quaternion(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    # slerp at t=0.5
    from math import acos, sin
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1; dot = -dot
    θ = acos(np.clip(dot, -1.0, 1.0))
    if abs(θ) < 1e-6:
        return q0
    w0 = sin((1-0.5)*θ)/sin(θ)
    w1 = sin(0.5*θ)/sin(θ)
    return (w0*q0 + w1*q1)/np.linalg.norm(w0*q0 + w1*q1)

# 6) SLERP interpolation
def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0/np.linalg.norm(q0); q1 = q1/np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1; dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    θ = np.arccos(dot)
    if abs(θ) < 1e-6:
        return q0
    sinθ = np.sin(θ)
    w0 = np.sin((1-t)*θ)/sinθ
    w1 = np.sin(t*θ)/sinθ
    return w0*q0 + w1*q1

# 7) Orientation error axis-angle
def orientation_error(q_current: np.ndarray,
                      q_target:  np.ndarray) -> (np.ndarray, float):
    qc = q_current/np.linalg.norm(q_current)
    qt = q_target/np.linalg.norm(q_target)
    inv = np.array([qc[0], -qc[1], -qc[2], -qc[3]], dtype=np.float32)
    # qt ⊗ inv
    w = qt[0]*inv[0] - qt[1]*inv[1] - qt[2]*inv[2] - qt[3]*inv[3]
    x = qt[0]*inv[1] + qt[1]*inv[0] + qt[2]*inv[3] - qt[3]*inv[2]
    y = qt[0]*inv[2] - qt[1]*inv[3] + qt[2]*inv[0] + qt[3]*inv[1]
    z = qt[0]*inv[3] + qt[1]*inv[2] - qt[2]*inv[1] + qt[3]*inv[0]
    δq = np.array([w, x, y, z], dtype=np.float32)
    δq /= np.linalg.norm(δq)
    angle = 2*np.arccos(np.clip(δq[0], -1.0, 1.0))
    axis  = δq[1:]/np.sin(angle/2 + 1e-8)
    return axis, angle
