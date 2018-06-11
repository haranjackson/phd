VISCOUS = True
THERMAL = True
REACTIVE = False
MULTI = False
LSET = 1

NV = 5 + int(VISCOUS) * 9 + int(THERMAL) * 3 + int(REACTIVE) + \
    int(MULTI) * 2 + LSET
