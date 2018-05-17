VISCOUS = True
THERMAL = False
REACTIVE = False
MULTI = False
LSET = 2

NV = 5 + int(VISCOUS) * 9 + int(THERMAL) * 3 + int(REACTIVE) + \
    int(MULTI) * 2 + LSET
