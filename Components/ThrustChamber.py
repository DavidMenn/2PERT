"""David Menn
The intent of this class is to model everything that goes on inside the combustion chamber and nozzle.
A ThrustChamber object just stores the geometry of the combustion chamber, with xlist being the list
of x values (in m) for each array index corresponding to rlist, which is the radius of the crosssection
at that x value.

The utility of this class is that, once given input parameters from the injection, it can give you the
temp, pressure, axial velocity, and whatever else I decide of the mixture inside the combustion chamber,
which will then be stored in the object.

Should be able to spit out thrust, isp, mdot, etc.

Should have independent function that spits out data for the end of just CC portion to be used when
optimizing nozzle

Future additions will be using cantera to model the combustion and get a reasonable value for L*

Future additions will be the modelling of combustion instability and resonance (first order stuff)
"""
import sys
sys.path.insert(1,"./")
import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj
import math
import Toolbox.IsentropicEquations as Ise
from scipy import interpolate
from scipy import optimize
class ThrustChamber(object):

    def __init__(self, rlist, xlist): #This is if you already have the shape, so copying another engine
        self.rlist = rlist
        self.xlist = xlist
        self.alist = math.pi*rlist**2

        self.rc=rlist[0]

        self.rt = np.amin(rlist)
        self.xt = xlist[ np.argmin(rlist) ]
        self.at = math.pi * self.rt ** 2

        self.xns = xlist[ np.where( np.diff( rlist ) < 0)[0][0] ] #finds first instance of deviation, ns stands for nozzle start
        self.xe = xlist[-1] 

        self.cr = (rlist[0]**2)/(self.rt**2)
        self.eps = (rlist[-1]**2)/(self.rt**2)

        self.areaInterpolator = interpolate.interp1d(self.xlist,
                                                     self.alist, kind='linear')

    def __init__(self, xlist, xns, rc, xt, rt_sharp, xe_cone, re_cone, rcf, rtaf, rtef, thetai, thetae, ar): #throat fillet is defined by two sections,
    #converging "throat approach fillet radius" and diverging "throat expansion fillet radius",
    #ar = epsilon = nozzle expansion ratio
        self.rlist, self.xlist = paraRlist(xlist, xns, rc, xt, rt_sharp, xe_cone, re_cone, rcf, rtaf, rtef, thetai, thetae,
              ar)
        self.alist = math.pi * self.rlist ** 2

        self.rc = self.rlist[0]

        self.rt = np.amin(self.rlist)
        self.xt = self.xlist[np.argmin(self.rlist)]
        self.at = math.pi * self.rt ** 2

        self.xns = self.xlist[
            np.where(np.diff(self.rlist) < 0)[0][0]]  # finds first instance of deviation, ns stands for nozzle start
        self.xe = self.xlist[-1]

        self.cr = (self.rlist[0] ** 2) / (self.rt ** 2)
        self.eps = (self.rlist[-1] ** 2) / (self.rt ** 2)

        self.areaInterpolator = interpolate.interp1d(self.xlist,
                                                     self.alist, kind='linear')

    #flowsimple working on, xlist and rlist
    # define the mach number first by xlist and rlist, then use the calculated mach number to find pressure and temperature
    #Flow props are dicts in format {'name': String, 'Mdot' : int (kg/s),

    def flowSimple(self, params):
        machlist=np.zeros(self.xlist.size)
        preslist=np.zeros(self.xlist.size)
        templist = np.zeros(self.xlist.size)

        gammaC2Tlist = np.zeros(self.xlist.size)
        gammaT2Elist = np.zeros(self.xlist.size)
        
        for i in range(len(gammaC2Tlist)):
            gammaC2Tlist[i] = ((self.xlist[i] - 0)/(self.xt - 0))*(params['gamma_throat'] - params['gamma']) + params['gamma']
        for i in range(len(gammaT2Elist)):
            gammaT2Elist[i] = (((self.xlist[i] - self.xt)/(self.xe - self.xt))*(params['gamma_exit'] - params['gamma_throat']) + params['gamma_throat'])

        
        totalTCC = None
        for index in -np.arange(-np.where(self.xlist==self.xns)[0][0],1): #iterate backwards through the combustion chamber
            x=self.xlist[index]
            preslist[index]=params['pinj']+(params['pc']-params['pinj'])*x/self.xns
            machlist[index]=Ise.machFromP(params['pinj'],preslist[index],params['gamma_exit'])
            if totalTCC is None:
                totalTCC = Ise.totalT(params['temp_c'],gammaC2Tlist[index],machlist[np.where(self.xlist==self.xns)[0][0]])
            templist[index] = Ise.TFromTotalT(totalTCC,gammaC2Tlist[index],machlist[index])

        #THIS IS SIMPE BECAUSE I"M ASSUMING YOU MADE SURE IT WAS CHOKED AT THE THROAT, also assume totalT is constant which is false (no losses?!)
        for index in np.arange(np.where(self.xlist==self.xns)[0][0],np.where(self.xlist==self.xt)[0][0]): #iterate to the throat from ns
            machlist[index] = Ise.machFromArea(self.alist[index],self.at,gammaC2Tlist[index],supersonic=False)
            preslist[index]= Ise.PFromTotalP(params['pinj'],gammaC2Tlist[index],machlist[index])
            templist[index] = Ise.TFromTotalT(totalTCC,gammaC2Tlist[index],machlist[index])

        index=np.where(self.xlist==self.xt)[0][0] #throat
        machlist[index] = 1
        preslist[index] = Ise.PFromTotalP(params['pinj'], params['gamma_throat'], machlist[index])
        templist[index] = Ise.TFromTotalT(totalTCC, params['gamma_throat'], machlist[index])

        for index in np.arange(np.where(self.xlist == self.xt)[0][0]+1, self.xlist.size):  # from throat to nozzle end
            machlist[index] = Ise.machFromArea(self.alist[index], self.at, gammaT2Elist[index], supersonic=True)
            preslist[index] = Ise.PFromTotalP(params['pinj'], gammaT2Elist[index], machlist[index])
            templist[index] = Ise.TFromTotalT(totalTCC, gammaT2Elist[index], machlist[index])

        self.machlist=machlist
        self.templist=templist
        self.preslist=preslist

        self.machInterpolator = interpolate.interp1d(self.xlist,
                                             machlist, kind='linear')
        self.tempInterpolator = interpolate.interp1d(self.xlist,
                                                     templist, kind='linear')

        self.presInterpolator = interpolate.interp1d(self.xlist,
                                                     preslist, kind='linear')

        




    """ Currently calculates like this: Assume known mdot, which is found by targetting an isp and thrust. Now we know from choked flow at the throat, assuming total temperature stays
    constant throughout the nozzle, the total pressure at the throat and thus the pressure at the throat. Now we can simply work backwards to get the pressures and velocities
    """
    """
        def flow(self, fuel_props, ox_props, pcEstimate = None): #MAKE SURE TO FIX UNITS OR YOURE GONNA GET SOME WEIRD SHIT

            convert from mdot/mdot to MR using getMRforER(ERphi=None, ERr=None)
            mdot = totalmdot
            CEA=CEA_Obj(propName='', oxName='', fuelName='', fac_CR=self.cr, useFastLookup=0, makeOutput=0, make_debug_prints=False) #maybe try to set units here
            if pcEstimate is None
                pcEstimate = 400 # idk some standard value i dont think this matters???

            initialTemp = get_Temperatures(Pc=pcEstimate, MR=mr, eps=40.0, frozen=0, frozenAtThroat=0)[1]
            initalProperties = CEA.get_Temperatures(Pc=pcEstimate, MR=mr, eps=40.0, frozen=0)
            initialPt = Ise.PFromMdotAtThroat(mdot , initalProperties[1] , Ise.totalT(initialTemp,initalProperties[1],1) , self.at , Ise.gasConstant()/initalProperties[0])
            get from that to Pc using area rule (put that in the isentopic toob ox)

            Do it again like three times

            Compare results
    """


class Bezier():
    def TwoPoints(t, P1, P2):
        """
        Returns a point between P1 and P2, parametised by t.
        INPUTS:
            t     float/int; a parameterisation.
            P1    numpy array; a point.
            P2    numpy array; a point.
        OUTPUTS:
            Q1    numpy array; a point.
        """

        if not isinstance(P1, np.ndarray) or not isinstance(P2, np.ndarray):
            raise TypeError('Points must be an instance of the numpy.ndarray!')
        if not isinstance(t, (int, float)):
            raise TypeError('Parameter t must be an int or float!')

        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def Points(t, points):
        """
        Returns a list of points interpolated by the Bezier process
        INPUTS:
            t            float/int; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            newpoints    list of numpy arrays; points.
        """
        newpoints = []
        # print("points =", points, "\n")
        for i1 in range(0, len(points) - 1):
            # print("i1 =", i1)
            # print("points[i1] =", points[i1])

            newpoints += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
            # print("newpoints  =", newpoints, "\n")
        return newpoints

    def Point(t, points):
        """
        Returns a point interpolated by the Bezier process
        INPUTS:
            t            float/int; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            newpoint     numpy array; a point.
        """
        newpoints = points
        # print("newpoints = ", newpoints)
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
            # print("newpoints in loop = ", newpoints)

        # print("newpoints = ", newpoints)
        # print("newpoints[0] = ", newpoints[0])
        return newpoints[0]

    def Curve(t_values, points):
        """
        Returns a point interpolated by the Bezier process
        INPUTS:
            t_values     list of floats/ints; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            curve        list of numpy arrays; points.
        """

        if not hasattr(t_values, '__iter__'):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if len(t_values) < 1:
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if not isinstance(t_values[0], (int, float)):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")

        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            # print("curve                  \n", curve)
            # print("Bezier.Point(t, points) \n", Bezier.Point(t, points))

            curve = np.append(curve, [Bezier.Point(t, points)], axis=0)

            # print("curve after            \n", curve, "\n--- --- --- --- --- --- ")
        curve = np.delete(curve, 0, 0)
        # print("curve final            \n", curve, "\n--- --- --- --- --- --- ")
        return curve


# from Bezier import Bezier #
# import numpy as np

# example of a 5-point Bezier curve with parameter t and a numpy array of inital points points1
"""
t_points = np.arange(0, 1, 0.01) #................................. Creates an iterable list from 0 to 1.
points1 = np.array([[0, 0], [0, 8], [5, 10], [9, 7], [4, 3]]) #.... Creates an array of coordinates.
curve1 = Bezier.Curve(t_points, points1) #......................... Returns an array of coordinates.
"""


def paraRlist(xlist, xns, rc, xt, rt_sharp, xe_cone, re_cone, rcf, rtaf, rtef, thetai, thetae,
              ar):  # throat fillet is defined by two sections,
    # converging "throat approach fillet radius" and diverging "throat expansion fillet radius",
    # ar = epsilon = nozzle expansion ratio
    conslope = ((rc - rt_sharp) / (xt - xns))
    divslope_c = ((re_cone - rt_sharp) / (xe_cone - xt))  # slopes are absolute values

    # theta = converging angle, phi = diverging angle
    theta = math.atan(conslope)
    phi = math.atan(divslope_c)

    # solving for the intersection of two lines is simply m1x+b1=m2x+b2 -> x=(b1-b2)/(m2-m1)

    # rcfx and rcfy are the center of the cirlce of the chamber fillet, found at the intersction of eq1 and eq2
    rcfx = ((rc - rcf) - (rc - conslope * (- xns) - (rcf / math.cos(theta)))) / (-conslope)
    rcfy = rc - rcf

    rtfx = xt  # center of the circle of the throat approach fillet = (rtfx, rtafy), throat expansion fillet = (rtfx, rtefy)
    rtafy = rt_sharp + rtaf / math.cos(theta)
    rtefy = rtafy - (rtaf - rtef)

    # inflection point I
    ix = rtfx + (rtef * math.sin(thetai))  # center throat expansion fillet + x component
    iy = rtefy - (rtef * math.sin(thetai))

    lambdaa = 0.5 * (1 + math.cos(phi))  # theoretical correction factor
    ln = (.8 * ((ar ** 0.5 - 1) * (rt_sharp))) / math.tan(phi)  # nozzle length

    # exit point E
    exitx = xt + ln
    exity = re_cone  # ar**0.5 * (rt_sharp + rtafy - rtaf) #radius of nozzle exit

    # intersection of I and E
    c1 = iy - math.tan(thetai) * ix
    c2 = exity - math.tan(thetae) * exitx
    interx = (c2 - c1) / (math.tan(thetai) - math.tan(thetae))
    intery = (math.tan(thetai) * c2 - math.tan(thetae) * c1) / (math.tan(thetai) - math.tan(thetae))

    def chamberfillet(rcfx, rcfy, rcf, x):  # equation of circle: chamber fillet
        ycf = rcfy + (rcf ** 2 - (x - rcfx) ** 2) ** 0.5
        # change x values to list???
        return ycf

    ##def throatfillet(rtfx, rtfy, rtf, x): #equation of circle: throat fillet
    #   ytf = rtfy - (rtf ** 2 - (x - rtfx) ** 2) ** 0.5
    #   return ytf

    def throat_a_fillet(rtfx, rtafy, rtaf, x):  # equation of circle: throat approach fillet
        ytaf = rtafy - (rtaf ** 2 - (x - rtfx) ** 2) ** 0.5
        return ytaf

    def throat_e_fillet(rtfx, rtefy, rtef, x):  # equation of circle: throat expansion fillet
        ytef = rtefy - (rtef ** 2 - (x - rtfx) ** 2) ** 0.5
        return ytef

    realrt = throat_a_fillet(rtfx, rtafy, rtaf, rtfx)
    rtdiff = realrt - rt_sharp
    xtextended = xt + rtdiff / conslope

    rtfx = xtextended
    rtafy = rt_sharp + rtaf
    rtefy = rtafy - (rtaf - rtef)

    # inflection point I
    ix = rtfx + (rtef * math.sin(thetai))  # center throat expansion fillet + x component
    iy = rtefy - (rtef * math.cos(thetai))

    lambdaa = 0.5 * (1 + math.cos(phi))  # theoretical correction factor
    ln = (.8 * ((ar ** 0.5 - 1) * (rt_sharp))) / math.tan(phi)  # nozzle length

    # exit point E
    exitx = xtextended + ln
    exity = re_cone  # ar**0.5 * (rt_sharp + rtafy - rtaf) #radius of nozzle exit

    # intersection of I and E
    c1 = iy - math.tan(thetai) * ix
    c2 = exity - math.tan(thetae) * exitx
    interx = (c2 - c1) / (math.tan(thetai) - math.tan(thetae))
    intery = (math.tan(thetai) * c2 - math.tan(thetae) * c1) / (math.tan(thetai) - math.tan(thetae))

    dx = np.diff(xlist).min()
    xlistnew = np.arange(0, exitx, dx)
    point1 = [rcfx, rc]
    point2 = [rcfx + rcf * math.sin(theta), rcfy + rcf * math.cos(theta)]
    point3 = [rtfx - rtaf * math.sin(theta),
              rtafy - rtaf * math.cos(theta)]  # tangential point to throat approach fillet

    rlist = np.zeros(xlistnew.size)
    t_points = np.linspace(0, 1, 50000)
    # t_points = np.linspace(ix,exitx,int((exitx-ix)/dx))
    # t_points = np.arange(0, 1, 0.01) #................................. Creates an iterable list from 0 to 1.
    points1 = np.array([[ix, iy], [interx, intery], [exitx, exity]])  # .... Creates an array of coordinates.
    curve1 = Bezier.Curve(t_points, points1)  # ......................... Returns an array of coordinates.

    for i in range(xlistnew.size):
        x = xlistnew[i]
        if x < point1[0]:  # x before first tangential point (pt 1) on chamber fillet
            rlist[i] = rc
        elif x < point2[0]:  # x in  two tangential points (pt 1 and pt 2) of chamber fillet
            rlist[i] = chamberfillet(rcfx, rcfy, rcf, x)  # equation of circle: chamber fillet
        elif x < point3[0]:  # x before third tangential point (pt 3) on throat fillet
            rlist[i] = rc - conslope * (x - xns)
        elif x < rtfx:  # x from throat approach fillet tangent point to throat
            rlist[i] = throat_a_fillet(rtfx, rtafy, rtaf, x)  # y value of throat approach fillet
        elif x < ix:  # x from throat to inflection point
            rlist[i] = throat_e_fillet(rtfx, rtefy, rtef, x)  #: #y value of throat expansion fillet
        else:  # x from inflection point I to exit point E
            tguess = (x - .0001 - ix) / (exitx - ix)  # prevents x=exitx causing an issue
            rlist[i] = curve1[int(tguess * 50000), 1]  # parabola: Bezier curve
            # rlist[i] = curve1[np.argmin(t_points-x),1]
    # plt.plot([ix,interx,exitx],[iy,intery,exity],'r*')
    return rlist, xlistnew


