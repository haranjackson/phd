VISCOUS = True
THERMAL = False
MULTI = False
REACTIVE = False
LSET = 0

NV = 5 + int(VISCOUS) * 9 + int(THERMAL) * 3 + int(MULTI) + LSET
