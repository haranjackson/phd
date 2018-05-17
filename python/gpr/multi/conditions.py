from numpy import array, concatenate, dot
from scipy.linalg import solve

from gpr.opts import THERMAL


def stick_bcs(RL, RR, PL, PR, xL, xR):

    if THERMAL:
        YL = RL[11:15, :4]
        YR = RR[11:15, :4]
        yL = concatenate([PL.v, PL.J[:1]])
        yR = concatenate([PR.v, PR.J[:1]])

    else:
        YL = RL[11:14, :3]
        YR = RR[11:14, :3]
        yL = PL.v
        yR = PR.v

    return solve(YL - YR, yR - yL + dot(YL, xL) - dot(YR, xR))


def slip_bcs(RL, RR, PL, PR, xL, xR):

    if THERMAL:
        YL = RL[[11, 14], :4]
        YR = RR[[11, 14], :4]
        yL = array([PL.v[0], PL.J[0]])
        yR = array([PR.v[0], PR.J[0]])

        M = YL[:, [0, -1]] - YR[:, [0, -1]]
        x_ = solve(M, yR - yL + dot(YL, xL) - dot(YR, xR))
        return array([x_[0], 0, 0, x_[1]])

    else:
        YL = RL[11, :3]
        YR = RR[11, :3]
        yL = PL.v[0]
        yR = PR.v[0]

        x_ = (yR - yL + dot(YL, xL) - dot(YR, xR)) / (YL - YR)[0]
        return array([x_, 0, 0])
