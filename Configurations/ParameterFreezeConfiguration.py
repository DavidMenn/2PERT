"""Just a place to run stuff, currently has some test code
usefull as it has a bunch of commonly used import statements
CURRENT CONFIGURATION: FIRST ORDER SIZINGS FOR MID AUTUMN 2022"""
import sys
sys.path.insert(1,"./")
import scipy.optimize
#from Components.ThrustChamber import ThrustChamber
from rocketcea.cea_obj import add_new_fuel, add_new_oxidizer, add_new_propellant
import numpy as np
import math
import Components.ThrustChamber as ThrustChamber
import Components.CoolingSystem as CS
from rocketcea.cea_obj_w_units import CEA_Obj
import Toolbox.RocketCEAAssister as RA
import os
import Toolbox.IsentropicEquations as IE
import difflib
import re as regex
from rocketprops.rocket_prop import get_prop
import Toolbox.Constant as const
import Analysis.DetermineOptimalMR as DOMR
import matplotlib.pyplot as plt
import Analysis.FirstOrderCalcs as FAC
import Components.ThrustChamber as ThrustChamber
import Components.StructuralApproximation as SA
import Components.CoolingSystem as CoolingSystem
import Components.Rocket as Rocket
from scipy.optimize import minimize_scalar


args = {
        'thrust': 20016.981,  # Newtons
        'time': 33.5,  # s
        # 'rho_ox' : 1141, #Kg/M^3
        # 'rho_fuel' : 842,
        'pc': 300 * const.psiToPa,
        'pe': 10 * const.psiToPa,
       # 'phi':1,
        'cr': None,
        'lstar': 1.24,
        'fuelname': 'Ethanol_75',
        'oxname': 'N2O',
        'throat_radius_curvature': .0254 *2,
        'dp': 150 * const.psiToPa,
        'impulseguess' :  495555.24828424345,
        'rc' : .11,
        'thetac' : (35*math.pi/180),
        'isp_efficiency' : .9} #623919}
configtitle = "NitroEthanol75 OOP 2_22_23 ParamFreeze"
output=True


# FIRST DETERMINE INITIAL ESTIMATES FOR IDEAL PARAMS
path=os.path.join( "Outputs",configtitle)
if output:
    os.makedirs(path,exist_ok=True)
ispmaxavg, mrideal, phiideal, ispmaxeq, ispmaxfrozen = DOMR.optimalMr(args, plot=output)
if output:
    plt.savefig(os.path.join(path, "idealisp.png"))
print(f"isp max = {ispmaxavg}, ideal mr is {mrideal}")
args['rm']=mrideal
params = FAC.SpreadsheetSolver(args)


structure = SA.StructuralApproximation(params, 12)
rocket = Rocket.Rocket(params, {"StructuralApproximation" : structure})
rocket.Equilibiurm(.25)
print(f"got to {rocket.h} at TWR {params['thrust']/9.81/structure.totalmasses} and mi = {structure.mis}")

newargs = {
    'thrust': params['thrust'],  # Newtons
    'time': params['time'],  # s
    'pc': params['pc'],
    'pe': params['pe'],
    'rm' : params['rm'],
    'rc': params['rc'],
    'lstar': params['lstar'],
    'fuelname': params['fuelname'],
    'oxname': params['oxname'],
    'throat_radius_curvature': params['throat_radius_curvature'],
    'dp': params['dp'],
    'isp_efficiency' : params['isp_efficiency']}
params = FAC.SpreadsheetSolver(newargs)
# now do it again with cr instead of rc to get all the specific values for finite combustors
newargs = {
    'thrust': params['thrust'],  # Newtons
    'time': params['time'],  # s
    'pc': params['pc'],
    'pe': params['pe'],
    'rm' : params['rm'],
    'cr': params['cr'],
    'lstar': params['lstar'],
    'fuelname': params['fuelname'],
    'oxname': params['oxname'],
    'throat_radius_curvature': params['throat_radius_curvature'],
    'dp': params['dp'],
    'isp_efficiency' : params['isp_efficiency']}
params = FAC.SpreadsheetSolver(newargs)




#conevol = math.pi*params['rc']**3*math.tan(params['thetac'])/3 - math.pi*params['rt']**3*math.tan(params['thetac'])/3
if params['thetac'] is None:
    params['thetac'] = math.pi*35/180
volfunc = lambda lc : math.pi*params['rc']**2*lc  +\
    math.pi*params['rc']**3/math.tan(params['thetac'])/3 -\
        math.pi*params['rt']**3/math.tan(params['thetac'])/3
lstarminimizer = lambda lc : volfunc(lc)/(params['rt']**2*math.pi) - params['lstar']
result = scipy.optimize.root(lstarminimizer, .05, args=(), method='hybr', jac=None, tol=None, callback=None, options=None)
params['lc']=result['x'][0]
xlist = np.linspace(0, params['lc'] + (params['rc'] - params['rt']) / math.tan(params['thetac']) + params['ln_conical'], 100)
    
TC = ThrustChamber.ThrustChamber(xlist, params['lc'], params['rc'],
                                params['lc'] + (params['rc'] - params['rt'])/(math.tan(params['thetac'])),
                                params['rt'],
                                params['lc'] + (params['rc'] - params['rt'])/(math.tan(params['thetac'])) + params['ln_conical'],
                                params['re'], params['lc']*1, 2*.0254, 2*.0254, math.pi/6, 8*math.pi/180, params['er'])  # xlist, xns, rc, xt, rt, xe, re
# xlist, xns, rc, xt, rt_sharp, xe_cone, re_cone, rcf, rtaf, rtef, thetai, thetae, ar
xlist = TC.xlist
TC.flowSimple(params)

xlistflipped = np.flip(xlist)
rlistflipped  = np.flip(TC.rlist)
chlist  = (TC.rt/rlistflipped)**.5*.003 
twlist  = (rlistflipped/TC.rt)*.001 
nlist  = np.ones(len(xlistflipped))*80
ewlist  = np.ones(len(xlistflipped))*.005
#HELICITY IS DEFINED AS 90 DEGREES BEING A STAIGHT CHANEL, 0 DEGREES BEING COMPLETILY CIRCUMFRNEITAL
helicitylist  = (rlistflipped**1.5/TC.rt**1.5)*45*math.pi/180
chanelToLandRatio = 2
for index in range(0,np.size(helicitylist )):
    if helicitylist [index]>math.pi/2:
        helicitylist [index] = math.pi/2

CS = CoolingSystem.CoolingSystem(params,TC, xlistflipped, rlistflipped, chlist, chanelToLandRatio, twlist, nlist, helicitylist,
                                dxlist=None,  material = "inconel 718", setupMethod = None)

CS.Equilibirum(initialcoolanttemp = 293, initialcoolantpressure =params['pc'] + params['pc']*.2 + 50*const.psiToPa )

CS.FOS(TC.preslist)



if output:
    # Twglist, hglist, qdotlist, Twclist, hclist, Tclist, coolantpressurelist, qdotlist, Trlist, rholist, viscositylist, Relist = CS.steadyStateTemperatures(None,TC, params, salistflipped,n, coolingfactorlist,
    #                        heatingfactorlist, xlistflipped, vlistflipped ,293, params['pc']+params['dp'][0], twlistflipped, hydraulicdiamlist)
    title = f"Factor Of Safety"
    plt.figure()
    plt.plot(xlistflipped, CS.FOSlist, 'b')
    plt.xlabel("Axial Position [m From Injector Face]")
    plt.ylabel("FOS")
    plt.title(title)

    title="ChamberTemps"
    fig, axs = plt.subplots(3,3)
    fig.suptitle(title)

    tilte = f"Chanel Geomsetries"
    plt.figure()
    plt.plot(xlistflipped,CS.chlist*1000,'r',label="Chanel Height [mm]")
    plt.plot(xlistflipped,CS.cwlist*1000,'b',label="Chanel Width [mm]")
    plt.plot(xlistflipped,CS.twlist*1000,'k',label="Wall Thickness [mm]")
    #plt.plot(xlistflipped,hydraulicdiamlist*1000,'g',label="Hydraulic Diam [mm]")
    plt.plot(xlistflipped,CS.helicitylist*180/math.pi/10,'m',label="helicity [10's of degrees]")
    plt.plot(xlistflipped,CS.vlist,'c',label="Coolant Velocity [m/s]")
    plt.legend()
    plt.xlabel("TC position, meters")
    plt.ylabel("Thicknesses, [mm]")
    plt.title(f"Geometry , nsub = {int(nlist[1])}, Chanel to Landsub = {chanelToLandRatio}")

    title = "ChamberTemps"
    fig, axs = plt.subplots(3, 3)
    fig.suptitle(title)

    axs[0, 1].plot(xlistflipped, CS.hglist, 'g')  # row=0, column=0
    axs[1, 1].plot(xlistflipped, CS.hclist, 'r')  # row=1, column=0
    axs[2, 1].plot(np.hstack((np.flip(TC.xlist), xlist)), np.hstack((np.flip(TC.rlist), -TC.rlist)), 'k')  # row=0, column=0

    axs[0, 1].set_title('hglist')
    axs[1, 1].set_title('hclist')
    axs[2, 1].set_title('Thrust Chamber Shape')

    axs[0, 0].plot(xlistflipped, CS.Twglist, 'g', label="Gas Side Wall Temp")
    axs[0, 0].plot(xlistflipped, CS.Twclist, 'r', label="CoolantSide Wall Temp")  # row=0, column=0
    axs[0, 0].plot(xlistflipped, CS.Tclist, 'b', label="Coolant Temp")  #
    axs[1, 0].plot(xlistflipped, CS.Tclist, 'r')  # row=1, column=0
    axs[2, 0].plot(xlistflipped, CS.hydraulicdiamlist, 'r')  # row=1, column=0

    axs[0, 0].set_title('Twg')
    axs[1, 0].set_title('Tc')
    axs[2, 0].set_title('hydraulicdiam')
    axs[0, 0].legend()

    axs[0, 2].plot(xlistflipped, CS.Twglist * const.degKtoR - 458.67, 'g', label="Gas Side Wall Temp, F")
    # axs[0,2].plot(xlistflipped,Tcoatinglist*const.degKtoR-458.67 , 'k', label="Opposite side of coating Temp, F")
    axs[0, 2].plot(xlistflipped, CS.Twclist * const.degKtoR - 458.67, 'r',
                   label="CoolantSide Wall Temp, F")  # row=0, column=0

    axs[0, 2].plot(xlistflipped, CS.Tclist * const.degKtoR - 458.67, 'b', label="Coolant Temp, F")  #
    axs[1, 2].plot(xlistflipped, CS.coolantpressurelist / const.psiToPa, 'k')
    axs[2, 2].plot(xlistflipped, CS.rholist, 'k')  # row=0, column=0

    axs[0, 2].set_title('Twg')
    axs[1, 2].set_title('coolantpressure (psi)')
    axs[2, 2].set_title('density of coolant')
    axs[0, 2].legend()
    plt.savefig(os.path.join(path, "temperatures.png"))
    print(f"max twg = {np.max(CS.Twglist)} in kelvin, {np.max(CS.Twglist) * const.degKtoR} in Rankine (freedom)\n max Twc ="
          f" {np.max(CS.Twclist)} in kelvin, {np.max(CS.Twclist) * const.degKtoR} in Rankine (freedom)")
    # Hide x labels and tick labels for top plots and y ticks for right plots.

    title = "Flow properties along thrust chamber"
    fig1, axs1 = plt.subplots(4, 1)

    fig1.suptitle(title)

    axs1[0].plot(TC.xlist, TC.machlist, 'g')  # row=0, column=0
    axs1[1].plot(TC.xlist, TC.preslist, 'r')  # row=1, column=0
    axs1[2].plot(TC.xlist, TC.templist, 'b')  # row=0, column=0
    axs1[3].plot(np.hstack((np.flip(TC.xlist), xlist)), np.hstack((np.flip(TC.rlist), -TC.rlist)), 'k')  # row=0, column=0

    axs1[0].set_title('Mach')
    axs1[1].set_title('Pressure')
    axs1[2].set_title('temperature')
    plt.savefig(os.path.join(path, "flowprops.png"))

    title = f"Chamber Wall Temperatures: Temp At Injector Face = {CS.Twglist[-1]}"
    plt.figure()
    plt.plot(xlistflipped, CS.Twglist, 'g', label="Gas Side Wall Temp, K")
    plt.plot(xlistflipped, CS.Twclist, 'r', label="CoolantSide Wall Temp, K")  # row=0, column=0
    plt.plot(xlistflipped, CS.Tclist, 'b', label="Coolant Temp, K")  #
    plt.xlabel("Axial Position [m From Injector Face]")
    plt.ylabel("Temperature [K]")
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(path, "ChamberTemps_LachlanFormat.png"))

    axs[0, 1].plot(xlistflipped, CS.hglist, 'g')  # row=0, column=0
    axs[1, 1].plot(xlistflipped, CS.hclist, 'r')  # row=1, column=0
    axs[2, 1].plot(np.hstack((np.flip(TC.xlist), xlist)), np.hstack((np.flip(TC.rlist), -TC.rlist)), 'k')  # row=0, column=0

    axs[0, 1].set_title('hglist')
    axs[1, 1].set_title('hclist')
    axs[2, 1].set_title('Thrust Chamber Shape')

    axs[0, 0].plot(xlistflipped, CS.Twglist, 'g', label="Gas Side Wall Temp")
    axs[0, 0].plot(xlistflipped, CS.Twclist, 'r', label="CoolantSide Wall Temp")  # row=0, column=0
    axs[0, 0].plot(xlistflipped, CS.Tclist, 'b', label="Coolant Temp")  #
    axs[1, 0].plot(xlistflipped, CS.Tclist, 'r')  # row=1, column=0
    axs[2, 0].plot(xlistflipped, CS.hydraulicdiamlist, 'r')  # row=1, column=0

    axs[0, 0].set_title('Twg')
    axs[1, 0].set_title('Tc')
    axs[2, 0].set_title('hydraulicdiam')
    axs[0, 0].legend()

    axs[0, 2].plot(xlistflipped, CS.Twglist * const.degKtoR - 458.67, 'g', label="Gas Side Wall Temp, F")
    # axs[0,2].plot(xlistflipped,Tcoatinglist*const.degKtoR-458.67 , 'k', label="Opposite side of coating Temp, F")
    axs[0, 2].plot(xlistflipped, CS.Twclist * const.degKtoR - 458.67, 'r',
                   label="CoolantSide Wall Temp, F")  # row=0, column=0

    axs[0, 2].plot(xlistflipped, CS.Tclist * const.degKtoR - 458.67, 'b', label="Coolant Temp, F")  #
    axs[1, 2].plot(xlistflipped, CS.coolantpressurelist / const.psiToPa, 'k')
    axs[2, 2].plot(xlistflipped, CS.rholist, 'k')  # row=0, column=0

    axs[0, 2].set_title('Twg')
    axs[1, 2].set_title('coolantpressure (psi)')
    axs[2, 2].set_title('density of coolant')
    axs[0, 2].legend()
    plt.savefig(os.path.join(path, "temperatures.png"))
    print(f"max twg = {np.max(CS.Twglist)} in kelvin, {np.max(CS.Twglist) * const.degKtoR} in Rankine (freedom)\n max Twc ="
          f" {np.max(CS.Twclist)} in kelvin, {np.max(CS.Twclist) * const.degKtoR} in Rankine (freedom)")

    title = f"Chamber Wall Temperatures: Temp At Injector Face = {CS.Twglist[-1]}"
    plt.figure()
    plt.plot(xlistflipped, CS.Twglist, 'g', label="Gas Side Wall Temp, K")
    plt.plot(xlistflipped, CS.Twclist, 'r', label="CoolantSide Wall Temp, K")  # row=0, column=0
    plt.plot(xlistflipped, CS.Tclist, 'b', label="Coolant Temp, K")  #
    plt.xlabel("Axial Position [m From Injector Face]")
    plt.ylabel("Wall Temperature [K]")
    plt.title(title)
    plt.savefig(os.path.join(path, "ChamberTemps_LachlanFormat.png"))

    axs[0, 1].plot(xlistflipped, CS.hglist, 'g')  # row=0, column=0
    axs[1, 1].plot(xlistflipped, CS.hclist, 'r')  # row=1, column=0
    axs[2, 1].plot(np.hstack((np.flip(TC.xlist), xlist)), np.hstack((np.flip(TC.rlist), -TC.rlist)), 'k')  # row=0, column=0

    axs[0, 1].set_title('hglist')
    axs[1, 1].set_title('hclist')
    axs[2, 1].set_title('Thrust Chamber Shape')

    axs[0, 0].plot(xlistflipped, CS.Twglist, 'g', label="Gas Side Wall Temp")
    axs[0, 0].plot(xlistflipped, CS.Twclist, 'r', label="CoolantSide Wall Temp")  # row=0, column=0
    axs[0, 0].plot(xlistflipped, CS.Tclist, 'b', label="Coolant Temp")  #
    axs[1, 0].plot(xlistflipped, CS.Tclist, 'r')  # row=1, column=0
    axs[2, 0].plot(xlistflipped, CS.hydraulicdiamlist, 'r')  # row=1, column=0

    axs[0, 0].set_title('Twg')
    axs[1, 0].set_title('Tc')
    axs[2, 0].set_title('hydraulicdiam')
    axs[0, 0].legend()

    axs[0, 2].plot(xlistflipped, CS.Twglist * const.degKtoR - 458.67, 'g', label="Gas Side Wall Temp, F")
    # axs[0,2].plot(xlistflipped,Tcoatinglist*const.degKtoR-458.67 , 'k', label="Opposite side of coating Temp, F")
    axs[0, 2].plot(xlistflipped, CS.Twclist * const.degKtoR - 458.67, 'r',
                   label="CoolantSide Wall Temp, F")  # row=0, column=0

    axs[0, 2].plot(xlistflipped, CS.Tclist * const.degKtoR - 458.67, 'b', label="Coolant Temp, F")  #
    axs[1, 2].plot(xlistflipped, CS.coolantpressurelist / const.psiToPa, 'k')
    axs[2, 2].plot(xlistflipped, CS.rholist, 'k')  # row=0, column=0

    axs[0, 2].set_title('Twg')
    axs[1, 2].set_title('coolantpressure (psi)')
    axs[2, 2].set_title('density of coolant')
    axs[0, 2].legend()
    plt.savefig(os.path.join(path, "temperatures.png"))
    print(f"max twg = {np.max(CS.Twglist)} in kelvin, {np.max(CS.Twglist) * const.degKtoR} in Rankine (freedom)\n max Twc ="
          f" {np.max(CS.Twclist)} in kelvin, {np.max(CS.Twclist) * const.degKtoR} in Rankine (freedom)")

    # plt.show()

    CS.GenerateCad(path, ewlist)
    

    """thetalist = np.arange(0, 2 * math.pi, .1)
    theta, r = np.meshgrid(thetalist, rlist - .01)
    theta, xgrid = np.meshgrid(thetalist, xlist)
    zgrid = r * np.cos(thetalist)
    ygrid = r * np.sin(thetalist)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    for index in range(0, xlistnew.shape[0]):
        ax.plot(xlistnew[index,], zlistnew[index,], ylistnew[index,], linewidth=.5)
    # chanel0 = ax.plot(xlistnew[0,], zlistnew[0,], ylistnew[0,],'r',linewidth=.5)
    # chanel1 = ax.plot(xlistnew[1,], zlistnew[1,], ylistnew[1,],'g',linewidth=.5)
    # chanel2 = ax.plot(xlistnew[2,], zlistnew[2,], ylistnew[2,],'b',linewidth=.5)
    # chanel3 = ax.plot(xlistnew[3,], zlistnew[3,], ylistnew[3,],'m',linewidth=.5)
    surf = ax.plot_surface(zgrid, ygrid, xgrid, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True, alpha=.5)

    # Customize the z axis.
    ax.set_zlim(0, np.max(xlist))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)"""
    with open(os.path.join(path,'mass_output.csv'), 'w') as f:
        for key in params.keys():
            f.write("%s,%s\n"%(key,params[key]))

    plt.show()

    print("end")

if ~output:
    params['twg_max'] = np.max(CS.Twglist)
    params['twc_max'] = np.max(CS.Twclist)
