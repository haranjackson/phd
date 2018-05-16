from numpy import array, concatenate, dot, zeros
from scipy.linalg import solve

from gpr.opts import THERMAL


def stick_bcs(RL, RR, PL, PR):

    if THERMAL:
        YL = RL[11:15, :4]
        YR = RR[11:15, :4]

        xL = concatenate([PL.Σ()[0], [PL.T()]])
        xR = concatenate([PR.Σ()[0], [PR.T()]])
        yL = concatenate([PL.v, PL.J[:1]])
        yR = concatenate([PR.v, PR.J[:1]])

    else:
        YL = RL[11:14, :3]
        YR = RR[11:14, :3]

        xL = PL.Σ()[0]
        xR = PR.Σ()[0]
        yL = PL.v
        yR = PR.v

    x_ = solve(YL - YR, yR - yL + dot(YL, xL) - dot(YR, xR))
    return xL, xR, x_


def slip_bcs(RL, RR, PL, PR):

    if THERMAL:
        YL = RL[[11, 14], :4]
        YR = RR[[11, 14], :4]

        xL = array([PL.Σ()[0], PL.T()])
        xR = array([PR.Σ()[0], PR.T()])
        yL = array([PL.v[0], PL.J[0]])
        yR = array([PR.v[0], PR.J[0]])

        M = YL[:, [0, -1]] - YR[:, [0, -1]]
        x_ = solve(M, yR - yL + dot(YL, xL) - dot(YR, xR))
        x_ = array([x_[0], 0, 0, x_[1]])

    else:
        YL = RL[11, :3]
        YR = RR[11, :3]

        xL = PL.Σ()[0]
        xR = PR.Σ()[0]
        yL = PL.v[0]
        yR = PR.v[0]

        x_ = (yR - yL + dot(YL, xL) - dot(YR, xR)) / (YL - YR)[0]
        x_ = array([x_, 0, 0])

    return xL, xR, x_


def vacuum_bcs(PL, PR):

    if THERMAL:
        xL = concatenate([PL.Σ()[0], [PL.T()]])
        xR = concatenate([PR.Σ()[0], [PR.T()]])
        x_ = zeros(4)
    else:
        xL = PL.Σ()[0]
        xR = PR.Σ()[0]
        x_ = zeros(3)

    return xL, xR, x_