#!/usr/bin/env python

import numpy as np
import casadi as ca
from acados_template import AcadosModel

class GemCarModel(object):
    def __init__(self,):

        model = AcadosModel()
        constraint = ca.types.SimpleNamespace()
        length = 2.565

        # control inputs
        a = ca.SX.sym('accel')
        fai = ca.SX.sym('fai')
        controls = ca.vertcat(a, fai) # fai: front angle

        # model states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        v = ca.SX.sym('v')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta, v)

        # dynamic
        rhs = [v*ca.cos(theta), v*ca.sin(theta), v*ca.tan(fai)/length, a] # v*ca.tan(fai)/length -> v*fai/length as |fai| < 30 degree

        # function
        f = ca.Function('f', [states, controls], [ca.vcat(rhs)], ['state', 'control_input'], ['rhs'])

        # acados model
        x_dot = ca.SX.sym('x_dot', len(rhs))
        f_impl = x_dot - f(states, controls)

        # other settings
        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = []
        model.name = 'GemCarModel'

        # constraints
        constraint.a_max = 0.8
        constraint.a_min = -0.8
        constraint.theta_max = np.pi/6
        constraint.theta_min = -np.pi/6
        constraint.expr = ca.vcat([a, fai])

        self.model = model
        self.constraint = constraint